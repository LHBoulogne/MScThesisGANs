import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, inp):
        return inp

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, inp):
        return inp.view(self.shape)


class Norm2d(nn.Module):
    def __init__(self, dim, mode):
        super(Norm2d, self).__init__()
        if mode == 'batch':
            self.main = nn.BatchNorm2d(dim)
        elif mode == 'instance':
            self.main = nn.InstanceNorm2d(dim)
        elif mode == 'none':
            self.main = Identity()
        else :
            raise RuntimeError('norm argument has unknown value: ' + mode)

    def forward(self, inp):
        return self.main(inp)


class Vector2FeatureMaps(nn.Module):
    def __init__(self, vec_len, feature_maps, mode, fm_dim=4, kernel_size=4):
        super(Vector2FeatureMaps, self).__init__()
        if mode == 'convtransposed':
            self.main = nn.Sequential(
                Reshape(-1, vec_len, 1, 1),
                nn.ConvTranspose2d(vec_len, feature_maps, kernel_size, stride=1, padding=0)
                )
        elif mode == 'linear':
            self.main = nn.Sequential(
                nn.Linear(vec_len, feature_maps*fm_dim*fm_dim),
                Reshape(-1, feature_maps, fm_dim, fm_dim),
                )
        else :
            raise RuntimeError('norm argument has unknown value: ' + mode)

    def forward(self, inp):
        return self.main(inp)

class FeatureMaps2Vector(nn.Module):
    def __init__(self, feature_maps, vec_len, mode, fm_dim=4, kernel_size=4):
        super(FeatureMaps2Vector, self).__init__()
        if mode == 'conv':
            self.main = nn.Sequential(
                nn.Conv2d(feature_maps, vec_len, kernel_size, stride=1, padding=0),
                Reshape(-1, vec_len)
                )
        elif mode == 'linear':
            self.main = nn.Sequential(
                Reshape(-1, feature_maps*fm_dim*fm_dim),
                nn.Linear(feature_maps*fm_dim*fm_dim, vec_len),
                )

    def forward(self, inp):
        return self.main(inp)


