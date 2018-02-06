import torch

def cuda(xs):
	print('?????\n\n')
    if torch.cuda.is_available():
    	print('HIERO')
        if xs.isinstance(list):
            return [cuda(x) for x in xs]
        if xs.isinstance(tuple):
            return tuple([cuda(x) for x in xs])
        return xs.cuda()
    print('hoi')
    return xs
