import json

class Configuration():
    def __init__(self, args, save_path):
        with open(save_path + "/configuration.json", "w") as f:
            json.dump(args, f)
        with open(save_path + "/configuration.json", "r") as f:
            config = json.load(f)
        self.config = config
        self.data_path = config['data_path']
        self.emoji = config['emoji']
        self.lr = config['lr']
        self.maxlen = config['maxlen']
        self.batch_size = config['batch_size']
        self.gpu = config['gpu']
        self.save_path = config['save_path']
        self.device = config['device']
        self.e_dim = config['e_dim']
        self.grid = config['grid']
        self.k = config['k']
        self.topk = config['topk']
        self.threshold = config['stopping_threshold']

    def configprint(self):
        for i in self.config.keys():
            print(f'{i} : {self.config[i]}')
        print("========================================")


