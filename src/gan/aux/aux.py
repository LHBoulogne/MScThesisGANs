import torch

def rescale(t):
    t.div_(0.5).add_(-1)
    return t

def to_one_hot(categories, y):
    if sum(categories) == 0 or categories is None:
        return []

    batch_size = len(y)

    onehot = torch.FloatTensor(batch_size, sum(categories))
    torch.zeros(batch_size, sum(categories), out=onehot)
    y = y.long()
    if len(y.size()) == 1:
        y=y.unsqueeze(1)

    for it in range(len(categories)):
        idcs = y[:,it].unsqueeze(1) + sum(categories[:it])
        onehot.scatter_(1, idcs, 1)
    return onehot
