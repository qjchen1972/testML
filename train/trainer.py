"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import logging
from tqdm import tqdm
import numpy as np
import os
import torch
from contextlib import nullcontext
from torch.utils.data.dataloader import DataLoader
from train.util import set_seed, dict2str, reduce_dict
from train.util import CfgNode as CN


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ScheduledLr(object):
    
    def __init__(self, init_lr, n_warmup_steps, final_steps, n_current_steps=0):
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = n_current_steps
        self.init_lr = init_lr
        self.final_steps = final_steps
        self.min_mult = 0.1
        self.lr = self.cal_lr()
       
    def cal_lr(self):
        lr = self.init_lr
        if self.n_warmup_steps > 0:
            if self.n_current_steps < self.n_warmup_steps:
                lr_mult = float(self.n_current_steps) / float(max(1, self.n_warmup_steps))
            else:
                progress = float(self.n_current_steps - self.n_warmup_steps) / float(max(1, self.final_steps - self.n_warmup_steps))
                lr_mult = max(self.min_mult, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.init_lr * lr_mult                     
        return lr
        
    def update_lr(self, optimizer):
        self.n_current_steps += 1  
        if self.n_warmup_steps > 0:
            self.lr = self.cal_lr()       
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr


class Trainer:
    
    @staticmethod
    def get_default_config():
    
        C = CN()
        # dataloder parameters, num_workers is 0 in windows
        C.num_workers = 0
        # optimizer parameters        
        C.max_epochs = 20
        C.batch_size = 128
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        #torch.float32, torch.bfloat16,  torch.float16
        #C.ptdtype = torch.bfloat16
        C.autocast = True       
        #multi GPU
        C.warmup_tokens = 2000
        C.ckpt_path = './models'
        C.rndseed = 42
        C.gradient_accumulation_steps = 2
        C.compile = True
        return C
    
    def __init__(self, config):
    
        self.cfg = config   
        self.initEnv()        
 
    def initEnv(self):
        
        self.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        if self.ddp:
            # windows下只支持gloo, linux下使用nccl
            backend = 'nccl'
            if os.name == "nt":
                backend = 'gloo'
            torch.distributed.init_process_group(backend=backend)
            ddp_rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
            seed_offset = ddp_rank # each process gets a different seed
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.master_process = True
            seed_offset = 0
            self.world_size = 1
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if self.master_process:
            os.makedirs(self.cfg.ckpt_path, exist_ok=True)    
        set_seed(self.cfg.rndseed + seed_offset)
        
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        if self.device_type == 'cuda':
            '''        在Ampere架构的GPU上，默认启用了TF32来进行计算加速，但是并不是每一个矩阵及卷积计算都一定会使用FP32。如果想强制关闭TF32，可以通过设置环境变量：export NVIDIA_TF32_OVERRIDE=0
            allow_tf32 标志在PyTroch1.7到PyTroch1.11中默认为True，
            在PyTorch1.12及更高版本中默认为False
            '''
            torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True
        
    def initModel(self, model, resume=None):
        model = model.to(self.device) 
        if resume is not None:
            model, self.optimizer, self.start_epoch, self.best_loss =\
                self.loadPoint(model, resume)
        else:
                       
            weight_decay, learning_rate, betas, device_type = \
            self.cfg.weight_decay, self.cfg.learning_rate, self.cfg.betas, self.device_type  
            self.optimizer = model.configure_optimizers(self.cfg)
            self.start_epoch = 0
            self.best_loss = float('inf')
        '''    
        if self.ddp:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
          
        if self.cfg.compile:
            logger.info("compiling the model... (takes a ~minute)")
            #print("compiling the model... (takes a ~minute)")
            model = torch.compile(model) # requires PyTorch 2.0
        '''
        
        if self.ddp:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, 
                   device_ids=[self.ddp_local_rank]) #, find_unused_parameters=True)    
        self.model = model
        
    def setDataset(self, train_dataset, test_dataset=None):
        if self.ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                                test_dataset) if test_dataset else None
        else:
            self.train_sampler = torch.utils.data.RandomSampler(train_dataset)
                                
            self.test_sampler = torch.utils.data.SequentialSampler(test_dataset) \
                                if test_dataset else None        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
    
    def loadPoint(self, model, epoch):
        ckpt_path = os.path.join(self.cfg.ckpt_path, f'model_{epoch}.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.device)        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['model'])
        weight_decay, learning_rate, betas, device_type = \
            self.cfg.weight_decay, self.cfg.learning_rate, self.cfg.betas, self.device_type        
        optimizer = model.configure_optimizers(self.cfg)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])        
        return model, optimizer, start_epoch, best_loss        
        
    def saveModel(self, name='best.pt'):
        if self.master_process: 
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            ckpt_path = os.path.join(self.cfg.ckpt_path, name)
            logger.info(f'saving {ckpt_path}')
            #print(f'saving {ckpt_path}')
            torch.save(raw_model.state_dict(), ckpt_path)
        
    def savePoint(self, epoch):
        if self.master_process:
            ckpt_path = os.path.join(self.cfg.ckpt_path, f'model_{epoch}.pt')
            raw_model = self.model.module if hasattr(self.model, "module") else self.model        
            checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': self.optimizer.state_dict() if self.optimizer else None,
                    'epoch': epoch,
                    'best_loss': self.best_loss
                }
            logger.info(f"saving checkpoint to {ckpt_path}")
            #print(f"saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)
    
    
    def run(self, collate_fn=None):
        
        model, cfg = self.model, self.cfg
        raw_model = model.module if hasattr(self.model, "module") else model
        
        ctx = torch.autocast(device_type=self.device_type, dtype=torch.bfloat16)\
              if cfg.autocast else nullcontext()
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.autocast)
        if cfg.compile:
            logger.info("compiling the model... (takes a ~minute)")
            model = torch.compile(model) 
        
        if self.optimizer:
            epoch_iter = len(self.train_dataset) // cfg.batch_size            
            init_tokens = self.start_epoch * epoch_iter            
            final_steps = cfg.max_epochs * epoch_iter   
            schedule = ScheduledLr(init_lr=cfg.learning_rate, 
                                   n_warmup_steps=cfg.warmup_tokens,
                                   final_steps=final_steps,                                        n_current_steps=init_tokens)
        
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            (data, sampler)  = (self.train_dataset, self.train_sampler)\
                                if is_train else (self.test_dataset, self.test_sampler)         
            
            loader = DataLoader(data, pin_memory=True,
                                collate_fn = collate_fn, #lambda x: tuple(zip(*x))
                                batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers,
                                sampler=sampler)
            
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader))\
                   if is_train else enumerate(loader)
            lossName = []   
            
            micro_step = 0            
            for it, x in pbar:
                # forward the model
                with torch.set_grad_enabled(is_train and self.optimizer is not None):
                    with ctx:
                        _, loss_dict = model(x)  
                        
                    loss = loss_dict['total_loss']    
                    #得到返回loss的key,用来组装显示                    
                    if it == 0: lossName = loss_dict.keys()                    
                    loss_dict_reduced = reduce_dict(loss_dict, self.world_size)
                    losses.append([loss_dict_reduced[k].item()\
                                   for k in loss_dict_reduced.keys()])      
                                   
                if is_train:                                    
                    if self.optimizer is  None: 
                        #不需要backward,也不能用于ddp                   
                        pbar.set_description(f"epoch {epoch} iter {it}: train loss\
                                {loss.item():.5f}")
                        #用于EM算法的M步   
                        raw_model.update()        
                    else:
                        if micro_step == 0:
                            self.optimizer.zero_grad(set_to_none=True)     
                            
                        canstep = ((micro_step == cfg.gradient_accumulation_steps - 1) or
                                  (it == len(loader) - 1))
                        if self.ddp:
                            '''
                            in DDP training we only need to sync gradients at the last micro step.
                            the official way to do this is with model.no_sync() context manager, but I really dislike that this bloats the code and forces us to repeat code looking at the source of that context manager, it just toggles this variable
                            '''                            
                            model.require_backward_grad_sync = canstep                            
                        scaler.scale(loss).backward()  
                        
                        if canstep:
                            if cfg.grad_norm_clip != 0.0:
                                scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
                                
                            scaler.step(self.optimizer)
                            scaler.update() 
                            micro_step = 0
                            schedule.update_lr(self.optimizer)
                            #pbar.set_description(f"epoch {epoch} iter {it}: train loss\
                            #    {loss.item():.5f}. lr {schedule.lr:e}")
                        else:
                            micro_step += 1 
                            
                        pbar.set_description(f"epoch {epoch} iter {it}: train loss\
                                {loss.item():.5f}. lr {schedule.lr:e}")                            
                
            mean_loss = np.mean(losses, axis=0)
            dict_loss = dict(zip(lossName, mean_loss))
            if self.master_process:
                if is_train: 
                    logger.info(f'train loss: {dict2str(dict_loss, 3)}')
                else:
                    logger.info(f'test loss: {dict2str(dict_loss, 3)}')  
           
            return dict_loss['total_loss']
            

        for epoch in range(self.start_epoch, self.cfg.max_epochs):            
            if self.ddp:
                self.train_sampler.set_epoch(epoch)
            run_epoch('train')
            if self.test_sampler:
                test_loss = run_epoch('test')
                if test_loss < self.best_loss:
                    self.saveModel()
                    self.best_loss = test_loss                
            self.savePoint(epoch)
            
        if self.ddp:
            torch.distributed.destroy_process_group()    

