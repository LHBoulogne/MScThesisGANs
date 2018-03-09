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
    parser.add_argument('--G_updates', type=int, default=1)
    parser.add_argument('--labelsmoothing', type=str2bool, default=True)
    
    #### GAN type ####
    parser.add_argument('--coupled', type=str2bool, default=True)
    parser.add_argument('--auxclas', type=str2bool, default=True)

    #### Dataset params ####
    parser.add_argument('--dataname', type=str, default="MNIST")
    parser.add_argument('--cropsize', type=int, default=160)
    # Coupled
    parser.add_argument('--batches', type=int,  default=25000)
    parser.add_argument('--balance', type=str2bool, default=True)
    parser.add_argument('--labelnames', nargs='+', type=str, default=["Smiling", "Male"]) # CelebA
    parser.add_argument('--labels1', nargs='+', type=str, default=[0,1,2,3,4,5,6,7,8,9]) # digits for MNIST, CelebA
    parser.add_argument('--labels2', nargs='+', type=str, default=[0,1,2,3,4,  6,7,8,9]) # digits for MNIST, CelebA
    parser.add_argument('--labels1_neg', nargs='+', type=str, default=[]) #  CelebA
    parser.add_argument('--labels2_neg', nargs='+', type=str, default=[]) #  CelebA
    parser.add_argument('--domainlabel', type=str, default=None) #CelebA
    parser.add_argument('--labeltype', type=str, default='bool') #CelebA
    # Class vector
    parser.add_argument('--c_len', type=int, default=10)

    #### Trainer params ####
    parser.add_argument('--algorithm', type=str, default='default')
    parser.add_argument('--gp_coef', type=int, default=10)

    #### Model params ####
    parser.add_argument('--imsize', type=int, default=64)
    parser.add_argument('--imgch', type=int, default=1)
    parser.add_argument('--weight_init', type=str, default='normal')
    parser.add_argument('--blocks', type=int, default=3)

    parser.add_argument('--norm', type=str, default='batch')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--b1', type=float, default=0.5)
    parser.add_argument('--b2', type=float, default=0.999)


    # auxclass
    parser.add_argument('--categories', type=int, default=10)
    parser.add_argument('--c_error_weight', type=float, default=1.0)
    
    # Generator
    parser.add_argument('--generator', type=str, default='dcgan')
    parser.add_argument('--z_len', type=int, default=100)
    parser.add_argument('--z_distribution', type=str, default='uniform')
    parser.add_argument('--g_dim', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=None)
    parser.add_argument('--g_b1', type=float, default=None)
    parser.add_argument('--g_b2', type=float, default=None)
    parser.add_argument('--g_weight_decay', type=float, default=0.0)
    parser.add_argument('--g_extra_conv', type=str2bool, default=False)
    parser.add_argument('--g_first_layer', type=str, default='convtransposed')
    parser.add_argument('--g_norm', type=str, default=None) #takes value of --norm if None
    parser.add_argument('--g_act', type=str, default='relu')
    
    # Discriminator
    parser.add_argument('--discriminator', type=str, default='dcgan')
    parser.add_argument('--d_dim', type=int, default=64)
    parser.add_argument('--d_lr', type=float, default=None)
    parser.add_argument('--d_b1', type=float, default=None)
    parser.add_argument('--d_b2', type=float, default=None)
    parser.add_argument('--d_weight_decay', type=float, default=0.0)
    parser.add_argument('--d_last_layer', type=str, default='conv')
    parser.add_argument('--d_norm', type=str, default=None) #takes value of --norm if None
    parser.add_argument('--d_act', type=str, default='leakyrelu')
    
    parser.add_argument('--weight_decay', type=float, default=0.0)
    config = parser.parse_args()

    config.d_weight_decay = config.weight_decay
    config.g_weight_decay = config.weight_decay
    
    if config.d_norm is None:
        config.d_norm = config.norm
    if config.g_norm is None:
        config.g_norm = config.norm


    if config.d_lr is None:
        config.d_lr = config.lr
    if config.g_lr is None:
        config.g_lr = config.lr

    if config.d_b1 is None:
        config.d_b1 = config.b1
    if config.g_b1 is None:
        config.g_b1 = config.b1

    if config.d_b2 is None:
        config.d_b2 = config.b2
    if config.g_b2 is None:
        config.g_b2 = config.b2

    if config.labels1 == ["None"]:
        config.labels1 = []
    if config.labels2 == ["None"]:
        config.labels2 = []
    if config.labels1_neg == ["None"]:
        config.labels1_neg = []
    if config.labels2_neg == ["None"]:
        config.labels2_neg = []

    if config.dataname == "MNIST":
        config.labels1 = [int(label) for label in config.labels1]
        config.labels2 = [int(label) for label in config.labels2]
    config.savefolder = os.path.join('../savedata/', config.savefolder)
    config.loadfolder = os.path.join('../savedata/', config.loadfolder)
    return config