from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from celeba import *
import models_64x64
import PIL.Image as Image
import tensorboardX
import torch
from torch.autograd import grad
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
import sys

""" param """
epochs = 50
batch_size = 64
n_critic = 5
lr = 0.0002
z_dim = 100
feature_dim = 64
savename = sys.argv[1]

def grad_penalty(inp_real, inp_fake, D):
    inp_hat = ()
    for idx in range(len(inp_fake)):
        e = utils.cuda(torch.rand(batch_size, 1,1,1))

        x = inp_real[idx].data
        x_wave = inp_fake[idx].data

        x_hat = e*x + (1-e)*x_wave

        inp_hat += (utils.cuda(Variable(x_hat, requires_grad=True)),)

    out_hat = ((D(*inp_hat),),) # simulate src

    gps = ()
    for idx in range(len(out_hat)):
        gradient = grad(out_hat[idx][0], inp_hat[idx],
            grad_outputs = utils.cuda(torch.ones(out_hat[idx][0].size())), 
            create_graph = True)[0]
        gradient = gradient.view(batch_size, -1)
        gp = ((gradient.norm(p=2, dim=1) - 1)**2).mean()
        gps += (gp,)
        
    return gps

""" gpu """
# gpu_id = [2]
# utils.cuda_devices(gpu_id)

if not torch.cuda.is_available:
    print("Training on GPU")
else:
    print('CUDA not available. Training on CPU.')
    

""" data """

celeba_data = CelebA_dataset(root='../data/celeba/',
              transform=transforms.Compose([transforms.CenterCrop(160),
                                            transforms.Scale((64,64)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))

data_loader = torch.utils.data.DataLoader(celeba_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)


""" model """
D = models_64x64.DiscriminatorWGANGP(3, dim=feature_dim)
G = models_64x64.Generator(z_dim, dim=feature_dim)
utils.cuda([D, G])

d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


""" load checkpoint """
ckpt_dir = './checkpoints/' + savename
utils.mkdir(ckpt_dir)
try:
    ckpt = utils.load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    d_optimizer.load_state_dict(ckpt['d_optimizer'])
    g_optimizer.load_state_dict(ckpt['g_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0


""" run """
writer = tensorboardX.SummaryWriter('./summaries/' + savename)

z_sample = Variable(torch.randn(100, z_dim))
z_sample = utils.cuda(z_sample)
for epoch in range(start_epoch, epochs):
    for i, (imgs, _) in enumerate(data_loader):
        # step
        step = epoch * len(data_loader) + i + 1

        # set train
        G.train()

        # leafs
        imgs = Variable(imgs)
        bs = imgs.size(0)
        z = Variable(torch.randn(bs, z_dim))
        imgs, z = utils.cuda([imgs, z])

        f_imgs = G(z)

        # train D
        r_logit = D(imgs)
        f_logit = D(f_imgs.detach())

        wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
        gp = grad_penalty((imgs,), (f_imgs,), D)[0]
        d_loss = -wd + gp * 10.0

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        writer.add_scalar('D/wd', wd.data.cpu().numpy(), global_step=step)
        writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)

        if step % n_critic == 0:
            # train G
            z = utils.cuda(Variable(torch.randn(bs, z_dim)))
            f_imgs = G(z)
            f_logit = D(f_imgs)
            g_loss = -f_logit.mean()

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            writer.add_scalars('G',
                               {"g_loss": g_loss.data.cpu().numpy()},
                               global_step=step)

        if (i + 1) % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, len(data_loader)))

        if (i + 1) % 100 == 0:
            G.eval()
            f_imgs_sample = (G(z_sample).data + 1) / 2.0

            save_dir = './sample_images_while_training/'  + savename
            utils.mkdir(save_dir)
            torchvision.utils.save_image(f_imgs_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(data_loader)), nrow=10)

    utils.save_checkpoint({'epoch': epoch + 1,
                           'D': D.state_dict(),
                           'G': G.state_dict(),
                           'd_optimizer': d_optimizer.state_dict(),
                           'g_optimizer': g_optimizer.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=2)
