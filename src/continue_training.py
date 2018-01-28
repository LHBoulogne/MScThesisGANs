from config import parse_args
from gan.gan import *

if __name__ == "__main__":
    config = parse_args()
    gan = GAN(config)
    if config.train:
        gan.load()
        gan.train()
        gan.save()
    else: # test
    	config.loadfolder = config.savefolder
    	gan.load()
        gan.test()