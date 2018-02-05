import torch.nn as nn

def weight_init(model, mode):
	if mode == 'normal':
	    for m in model._modules:
	        normal_init(model._modules[m])
	elif mode == 'pytorch_default':
		return


def normal_init(m):
    if hasattr(m, 'weight'):
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1, 0.02)
        elif not isinstance(m, nn.InstanceNorm2d): #instance norm weight is None
            m.weight.data.normal_(0, 0.02)
        if not isinstance(m, nn.InstanceNorm2d):
            m.bias.data.zero_()
    else:
        for mod in m._modules:
            normal_init(m._modules[mod])

'''
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
    if isinstance(m, nn.Sequential):
        for mod in m._modules:
            normal_init(m._modules[mod], mean, std)
'''
