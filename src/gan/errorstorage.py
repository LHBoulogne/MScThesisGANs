import os
import pickle
import copy

class ErrorStorage(): 
    def __init__(self, config):
        self.config = config
        self.error_dicts = []
        self.error_dicts += [{}]
        if self.config.coupled:
            self.error_dicts += [{}]
        for error_dict in self.error_dicts:
            self.init_error_dict(error_dict)

    def init_error_dict(self, d):
        d['G'] = {}
        d['G']['source'] = []
        
        d['D'] = {}
        if self.config.algorithm == 'default':
            d['D']['real'] = {}
            d['D']['fake'] = {}
            d['D']['real']['source'] = []
            d['D']['fake']['source'] = []
        else:
            d['D']['source'] = []
        

        if self.config.auxclas:
            d['G']['classification'] = []
            if self.config.algorithm == 'default':
                d['D']['real']['classification'] = []
                d['D']['fake']['classification'] = []
            else :
                d['D']['classification'] = []
            
    def get_error_dicts(self):
        error_dicts = []
        for error_dict in self.error_dicts:
            error_dicts += [copy.deepcopy(error_dict)]
        return error_dicts

    def save_error(self):
        filename = os.path.join(self.config.savefolder, 'error.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.error_dicts, f)

    def load_error(self):
        filename = os.path.join(self.config.loadfolder, 'error.pkl')
        with open(filename, 'rb') as f:
            self.self.error_dicts = pickle.load(f)

    def store_errors2(self, error_list, errors):
        keys = ['source', 'classification']
        for it, error in enumerate(errors):
            key = keys[it]
            error_list[key] += [error.data.numpy()]

    def store_errors1(self, model, error_fake, error_real, d):
        if model == 'generator':
            error_list_fake = d['G']
        if model == 'discriminator':
            if self.config.algorithm == 'default':
                error_list_fake = d['D']['fake']
                error_list_real = d['D']['real']
            else :
                error_list_fake = d['D']

        self.store_errors2(error_list_fake, error_fake)

        if not error_real is None:
            self.store_errors2(error_list_real, error_real)


    def store_errors(self, model, error_fake, error_real=(None,None)):
        for it in range(len(error_fake)):
            self.store_errors1(model, error_fake[it], error_real[it], self.error_dicts[it])
