import numpy as np
import torch
import copy

from utils.utils import *
from utils.metrics import get_mae_rmse_mape

def train_epoch(model, device, dataloader, scaler, optimizer, scheduler, criterion):

    model.train()
    losses = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        preds = scaler.inverse_transform(outputs)

        loss = criterion(preds, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)
    scheduler.step()

    return avg_loss


def train(model, device, train_loader, val_loader, scaler, optimizer, scheduler, criterion, 
          max_epochs, patience, logger=None):
    
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    w_list = []
    
    early_stopping = EarlyStopping(patience=patience, trace_func=logger.info)

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, device, train_loader, scaler, optimizer, scheduler, criterion)
        train_losses.append(train_loss)
        
        parameters_copy = {name: param.clone() for name, param in model.named_parameters()}
        w_list.append(parameters_copy)

        _,_,_,val_loss = eval(model, device, val_loader, scaler, criterion)
        val_losses.append(val_loss)

        message = "Epoch: {}\tTrain Loss: {:.4f} Val Loss: {:.4f} LR: {:.4e}"
        logger.info(message.format(epoch+1, train_loss, val_loss, scheduler.get_last_lr()[0]))

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch: {epoch+1} Best at epoch {early_stopping.best_epoch}")
            break

    # load best
    model.load_state_dict(early_stopping.best_checkpoint)
    best_epoch = early_stopping.best_epoch
    
    # eval model
    train_mae, train_rmse, train_mape, _ = eval(model, device, train_loader, scaler, criterion)
    val_mae, val_rmse, val_mape, _ = eval(model, device, val_loader, scaler, criterion)
    
    train_log = "Train Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(train_log.format(train_losses[best_epoch-1], train_mae, train_rmse, train_mape))
    
    val_log = "Val Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(val_log.format(val_losses[best_epoch-1], val_mae, val_rmse, val_mape))

    return model, w_list


@torch.no_grad()
def eval(model, device, dataloader, scaler, criterion):
    
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = scaler.inverse_transform(outputs)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)         # [all_samples,12,num_nodes,dims]
    all_labels = torch.cat(all_labels, dim=0)       # [all_samples,12,num_nodes,dims]
    
    mae, rmse, mape = get_mae_rmse_mape(all_preds, all_labels)
    loss = criterion(all_preds, all_labels)

    return mae, rmse, mape, loss.item()


@torch.no_grad()
def test(model, device, test_loader, scaler, logger):
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = scaler.inverse_transform(outputs)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    test_maes = []
    test_mapes = []
    test_rmses = []
    
    output_len = all_labels.shape[1]
    for i in range(output_len):
        mae, rmse, mape = get_mae_rmse_mape(all_preds[:,i,...], all_labels[:,i,...])
        log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        logger.info(log.format(i + 1, mae, rmse, mape))
        test_maes.append(mae)
        test_rmses.append(rmse)
        test_mapes.append(mape)

    log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    logger.info(log.format(np.mean(test_maes), np.mean(test_rmses), np.mean(test_mapes)))
    