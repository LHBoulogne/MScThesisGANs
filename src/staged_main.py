from config import parse_args
from gan.gan import *

if __name__ == "__main__":
    config = parse_args()
    gan = GAN(config)
    if config.train:
        gan.train()
        gan.save()

        config.loadfolder = config.savefolder
        config.savefolder = config.savefolder+'_nofives'
        config.labels2 = [1,2,3,4,  6,7,8,9,0]

        gan = GAN(config)
        gan.load()
        gan.train()
        gan.save()
    else: # test
        gan.load()
        gan.test()

        config.savefolder = config.savefolder+'_nofives'
        config.labels2 = [1,2,3,4,  6,7,8,9,0]

        gan = GAN(config)
        gan.load()
        gan.test()
