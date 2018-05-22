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
    parser = argparse.ArgumentParser(description='Classifier')
        
    parser.add_argument('--train', type=str2bool, default=True)

    parser.add_argument('--savefolder', type=str, default='default')
    parser.add_argument('--loadfolder', type=str, default='default')
    parser.add_argument('--dloadworkers', type=int, default=3)
    parser.add_argument('--val_set_len', type=int, default=10000)

    parser.add_argument('--classifier', type=str, default='cogan')
    parser.add_argument('--mini_batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2000)
    
    parser.add_argument('--imsize', type=int, default=28)
    parser.add_argument('--imgch', type=int, default=1)

    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--dropout', type=str2bool, default=True)
    
    parser.add_argument('--categories', nargs='+', type=int, default=[10])
    config = parser.parse_args()

    config.savefolder = os.path.join('../c_savedata/', config.savefolder)
    config.loadfolder = os.path.join('../c_savedata/', config.loadfolder)
    return config