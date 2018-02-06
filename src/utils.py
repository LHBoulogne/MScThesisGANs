import torch

def cuda(xs):
    if torch.cuda.is_available():
        if xs.isinstance(list):
            return [cuda(x) for x in xs]
        if xs.isinstance(tuple):
            return tuple([cuda(x) for x in xs])
        return xs.cuda()
    return xs
