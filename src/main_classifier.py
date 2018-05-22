from classifier.classifier_config import parse_args
from classifier.classifier import *

if __name__ == "__main__":
    config = parse_args()
    classifier = Classifier(config)
    if config.train:
        classifier.train()
    else: # test
        classifier.load()
        classifier.test()