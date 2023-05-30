#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from train.util import set_seed

#init set log level = ERROR, in order to avoid system function show log 
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.ERROR,        
        #filename= 'dat.log',
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AdditionDataset(Dataset):
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "8550531"

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639054"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.
    """

    def __init__(self, split):
    
        self.split = split # train/test

        # split up all addition problems into either training data or test data
        ndigit = 2
        assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**ndigit)**2 # total number of possible addition problems with ndigit numbers
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)        
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]
        self.ndigit = ndigit

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        # a,b,a+b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 3*self.ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.ndigit
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # reverse c to make addition easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:ndigit*2-1] = -1 # we will only train in the output locations. -1 will mask loss to zero
        return x, y



# print an example instance of the dataset
train_dataset = AdditionDataset('train')
test_dataset = AdditionDataset('test')
x, y = train_dataset[0]
for a, b in zip(x,y):
    print(int(a),int(b))


# create a GPT instance
from model import GPT

#initial model need set random seed
set_seed(3407)

model_config = GPT.get_default_config()
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
print(model_config)
model = GPT(model_config)


# create a Trainer object
from train.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
train_config.max_epochs = 20
train_config.num_workers = 0
train_config.compile = False
trainer = Trainer(train_config)
trainer.setDataset(train_dataset, test_dataset)
trainer.initModel(model, resume=None)

trainer.run()


# In[7]:


# now let's perform some evaluation
model.eval();

# In[8]:

# helper function for the evaluation of a model
def eval_split(trainer, split, max_batches=None):
    dataset = {'train':train_dataset, 'test':test_dataset}[split]
    ndigit = dataset.ndigit
    results = []
    mistakes_printed_already = 0
    factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        # isolate the first two digits of the input sequence alone
        d1d2 = x[:, :ndigit*2]
        # let the model sample the rest of the sequence
        d1d2d3 = model.generate(d1d2, ndigit+1) # using greedy argmax, not sampling
        # isolate the last digit of the sampled sequence
        d3 = d1d2d3[:, -(ndigit+1):]
        d3 = d3.flip(1) # reverse the digits to their "normal" order
        # decode the integers from individual digits
        d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
        d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
        d3i_pred = (d3 * factors).sum(1)
        d3i_gt = d1i + d2i # manually calculate the ground truth
        # evaluate the correctness of the results in this batch
        correct = (d3i_pred == d3i_gt).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            if not correct[i] and mistakes_printed_already < 5: # only print up to 5 mistakes to get a sense
                mistakes_printed_already += 1
                print("GPT claims that %d + %d = %d but gt is %d" % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
        if max_batches is not None and b+1 >= max_batches:
            break
    rt = torch.tensor(results, dtype=torch.float)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()



# run a lot of examples from both train and test through the model and verify the output correctness
with torch.no_grad():
    train_score = eval_split(trainer, 'train', max_batches=50)
    test_score  = eval_split(trainer, 'test',  max_batches=50)


