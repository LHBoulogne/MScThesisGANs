import torch

def rescale(t):
    t.div_(0.5).add_(-1)
    return t

def to_one_hot(categories, y):
    if categories == 0 or categories is None:
        return []

    batch_size = len(y)

    y = y.cpu().view(-1,1)
    onehot = torch.FloatTensor(batch_size, categories)
    torch.zeros(batch_size, categories, out=onehot)
    onehot.scatter_(1, y, 1)
    return onehot