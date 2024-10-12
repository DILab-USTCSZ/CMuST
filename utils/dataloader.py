import torch
import numpy as np
import os

class StandardScaler:
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    

def get_dataloaders_scaler(dataset_dir, batch_size=16, logger=None):
    
    data = {}
    datasets = {}
    dataloaders = {}
    num_samples = 0
    
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y'][...,:1]
        num_samples += data['x_' + category].shape[0]
        
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        datasets[category] = torch.utils.data.TensorDataset(torch.FloatTensor(data['x_' + category]), torch.FloatTensor(data['y_' + category]))
    
    # (num_samples, length, num_nodes, dim)
    logger.info(f"Data Length: {num_samples} Node num: {data['x_train'].shape[2]}")
    logger.info(f"Train num: {data['x_train'].shape[0]} Val num: {data['x_val'].shape[0]} Test num: {data['x_test'].shape[0]}")

    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)

    return dataloaders, scaler
