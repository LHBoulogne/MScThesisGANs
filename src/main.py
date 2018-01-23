from config import parse_args
from gan.gan import *

if __name__ == "__main__":
    config = parse_args()
    gan = GAN(config)
    if config.train:
        gan.train()
        gan.save()
    else: # test
        gan.load()
        gan.test()