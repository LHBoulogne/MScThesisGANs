import os
import pickle
import copy

class ErrorStorage(): 
    def __init__(self, config):
        self.config = config
        self.error_dicts = [{}]
        if self.config.coupled:
            self.error_dicts += [{}]
        for error_dict in self.error_dicts:
            self.init_error_dict(error_dict)

    def init_error_dict(self, d):
        d['G'] = {}
        d['G']['source'] = []
        
        d['D'] = {}
        d['D']['err1'] = {}
        d['D']['err2'] = {}
        d['D']['err1']['source'] = []
        d['D']['err2']['source'] = []        

        if self.config.auxclas:
            d['G']['classification'] = []
            d['D']['err1']['classification'] = []
            d['D']['err2']['classification'] = []
            
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

    def savable(self, error):
        return error.data.cpu().numpy()

    def store_errors(self, model, idx, src_error, class_error=None):
        error_list_fake = self.error_dicts[idx][model]
        d = self.error_dicts[idx][model]
        if model == 'D':
            d['err1']['source'] += [self.savable(src_error[0])]
            d['err2']['source'] += [self.savable(src_error[1])]
        else:
            d['source'] += [self.savable(src_error)]

        if model == 'D':
            d['err1']['classification'] += [self.savable(src_error[0])]
            d['err2']['classification'] += [self.savable(src_error[1])]
        else:
            self.error_dicts[idx][model]['classification'] += [self.savable(src_error)]
            