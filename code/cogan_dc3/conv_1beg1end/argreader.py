import sys

class ArgReader :
    def __init__(self):
        self.current_arg = 1
        self.arg_string = ''
    
    def next_arg(self):
        idx = self.current_arg
        self.current_arg += 1
        self.arg_string += '_' + sys.argv[idx]
        return sys.argv[idx]