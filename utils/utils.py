import numpy as np
import torch
import copy

def save_prompt_weights(model, file_path):
    torch.save(model.prompt.data, file_path)


def load_prompt_weights(model, file_path):
    model.prompt.data = torch.load(file_path)
    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def calculate_variance(weight_snapshots):
    # Calculating variance across epochs for each weight matrix element-wise
    weight_snapshots = np.stack(weight_snapshots, axis=0)
    variance = np.var(weight_snapshots, axis=0)
    return variance


class MaskedMAELoss(torch.nn.Module):
    
    def __init__(self):
        super(MaskedMAELoss, self).__init__()

    def forward(self, v_, v):
        mask = (v != 0.0)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(v_ - v)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=30, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_checkpoint = None
        self.best_epoch = 0
        self.current_epoch = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss
        self.current_epoch += 1
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.best_checkpoint = copy.deepcopy(model.state_dict())
        self.best_epoch = self.current_epoch
        self.val_loss_min = val_loss
        