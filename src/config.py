import argparse
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args() :
    parser = argparse.ArgumentParser(description='GAN')
    
    ###############
    #### Main #####
    ###############
    parser.add_argument('--train', type=str2bool, default=True)

    #### Saving and loading ####
    parser.add_argument('--savefolder', type=str, default='default')
    parser.add_argument('--loadfolder', type=str, default='default')

    ##############################
    #### Snapshot params ####
    ##############################
    parser.add_argument('--visualize_training', type=str2bool, default=True)
    parser.add_argument('--snap_step', type=int, default=500)
    #### saving training images #####
    parser.add_argument('--vis_dim', type=int, default=6)
    #### plotting ####
    parser.add_argument('--combine_GANs', type=str2bool, default=False)
    parser.add_argument('--combine_gd', type=str2bool, default=False)
    parser.add_argument('--combine_rf', type=str2bool, default=True)
    parser.add_argument('--combine_sc', type=str2bool, default=True)
    

    #########################
    #### Training params ####
    #########################
    parser.add_argument('--mini_batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--labelsmoothing', type=str2bool, default=True)
    
    #### GAN type ####
    parser.add_argument('--coupled', type=str2bool, default=True)
    parser.add_argument('--auxclas', type=str2bool, default=True)

    #### Dataset params ####
    parser.add_argument('--dataname', type=str, default="MNIST")
    # Coupled
    parser.add_argument('--batches', type=int,  default=25000)
    parser.add_argument('--balance', type=str2bool, default=True)
    parser.add_argument('--colabelname', type=str, default="Male") # CelebA
    parser.add_argument('--labels1', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9]) # MNIST
    parser.add_argument('--labels2', nargs='+', type=int, default=[0,1,2,3,4,  6,7,8,9]) # MNIST
    # Class vector
    parser.add_argument('--c_len', type=int, default=10)

    #### Model params ####
    parser.add_argument('--imsize', type=int, default=64)
    parser.add_argument('--imgch', type=int, default=1)
    parser.add_argument('--categories', type=int, default=10)
    parser.add_argument('--weight_init', type=str, default='normal')
    parser.add_argument('--norm', type=str, default='batch')
    # Generator
    parser.add_argument('--generator', type=str, default='dcgan')
    parser.add_argument('--z_len', type=int, default=100)
    parser.add_argument('--z_distribution', type=str, default='uniform')
    parser.add_argument('--g_dim', type=int, default=64)
    parser.add_argument('--g_lr', type=int, default=0.0002)
    parser.add_argument('--g_b1', type=int, default=0.5)
    parser.add_argument('--g_b2', type=int, default=0.999)
    parser.add_argument('--g_extra_conv', type=str2bool, default=False)
    parser.add_argument('--g_first_layer', type=str, default='convtransposed')
    
    # Discriminator
    parser.add_argument('--discriminator', type=str, default='dcgan')
    parser.add_argument('--d_dim', type=int, default=64)
    parser.add_argument('--d_lr', type=int, default=0.0002)
    parser.add_argument('--d_b1', type=int, default=0.5)
    parser.add_argument('--d_b2', type=int, default=0.999)
    parser.add_argument('--d_last_layer', type=str, default='conv')






    config = parser.parse_args()
    config.savefolder = os.path.join('../savedata/', config.savefolder)
    config.loadfolder = os.path.join('../savedata/', config.savefolder)
    return config