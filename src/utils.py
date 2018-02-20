import torch

def cuda(xs):
    if xs is None:
        return None
        
    if torch.cuda.is_available():
        if isinstance(xs,list):
            return [cuda(x) for x in xs]
        if isinstance(xs,tuple):
            return tuple([cuda(x) for x in xs])
        return xs.cuda()
    return xs
