import numpy as np
import os
import torch
import torch.nn as nn
import random
import datetime
import copy

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from utils.utils import *
from utils.dataloader import *
from utils.logging import *
from utils.args import *
from engine import *
from model.models import CMuST

def get_config():
    parser = create_parser()
    args = parser.parse_args()
    
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    log_dir = 'logs/{}/'.format(args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(log_dir, __name__, '{}.log'.format(now))
    logger.info(args)
    
    return args, logger, now

if __name__ == "__main__":
    
    # configuration parameters, logger, and current time
    args, logger, now = get_config()
    
    # is random seed
    if args.seed == 0:
        seed = random.randint(1, 1000)
    else:
        seed = args.seed
        
    set_seed(seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset_dir = f"data/{args.dataset}"
    
    # built model
    model = CMuST(
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        tod_size=args.tod_size,
        obser_dim=args.obser_dim,
        output_dim=args.output_dim,
        d_obser=args.d_obser,
        d_tod=args.d_tod,
        d_dow=args.d_dow,
        d_ts=args.d_ts,
        d_s=args.d_s,
        d_t=args.d_t,
        d_p=args.d_p,
        self_atten_dim=args.self_atten_dim,
        cross_atten_dim=args.cross_atten_dim,
        ffn_dim=args.ffn_dim,
        n_heads=args.n_heads,
        dropout=args.dropout
    )
    
    # load pre-trained model weights
    # model.load_state_dict(torch.load('2024-04-18-10-18-28.pt'))
    
    # load dataset
    tasks = os.listdir(dataset_dir)
    task_dirs = [os.path.join(dataset_dir, item) for item in tasks]
    logger.info(f"Load data {task_dirs}")
    train_loaders=[]
    val_loaders=[]
    test_loaders=[]
    scalers=[]
    for task_dir in task_dirs:
        dataloaders, scaler = get_dataloaders_scaler(task_dir, args.batch_size, logger)
        train_loaders.append(dataloaders['train'])
        val_loaders.append(dataloaders['val'])
        test_loaders.append(dataloaders['test'])
        scalers.append(scaler)

    # loss function, optimizer, and scheduler
    criterion = MaskedMAELoss()
    # criterion = nn.HuberLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=args.steps,gamma=args.gamma)

    # model structure information
    total_params = sum(param.nelement() for param in model.parameters())
    logger.info(f'The number of parameters: {total_params}')
    
    # Rolling Adaption
    threshold = args.threshold
    
    # train task 1
    logger.info('Train for task 1')
    save_dir = 'checkpoints/{}/{}/{}/'.format(args.dataset,tasks[0],now)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # load prompt
    load_prompt_weights(model, os.path.join(task_dirs[0], f"prompt.pth"))
    # start train
    model, _ = train(model, device, train_loaders[0], val_loaders[0], scalers[0], 
                    optimizer, scheduler, criterion, args.max_epochs, args.patience, logger=logger)
    # test model
    test(model, device, test_loaders[0], scalers[0], logger=logger)
    # save prompt and model
    save_prompt_weights(model, os.path.join(save_dir, f"start_prompt.pth"))
    torch.save(model.state_dict(), os.path.join(save_dir, f"start.pt"))
    # update weight list
    weight_histories = {name: [] for name, param in model.named_parameters()}
    for name, param in model.named_parameters():
        weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())
    
    # train to task k
    for i, task in enumerate(tasks):
        if i == 0:
            continue
        # reset train
        optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()), 
                         lr=args.lr*0.01, weight_decay=args.weight_decay, eps=1e-8)
        scheduler = MultiStepLR(optimizer, milestones=[500], gamma=args.gamma)
        # train task 2 to k
        logger.info(f'Train for task {i+1}')
        save_dir = 'checkpoints/{}/{}/{}/'.format(args.dataset,tasks[i],now)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # load prompt
        load_prompt_weights(model, os.path.join(task_dirs[i], f"prompt.pth"))
        # start train
        model, w_list = train(model, device, train_loaders[i], val_loaders[i], scalers[i], 
                        optimizer, scheduler, criterion, args.max_epochs, args.patience, logger=logger)
        # append weight
        for w in w_list:
            for name, param in w.items():
                weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())
        # cal var & frozen
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                variances = calculate_variance(weight_histories[name])
                if 'prompt' not in name and np.all(variances < threshold):
                    param.requires_grad = False
        # print frozen params
        trainable_params = sum(param.nelement() for param in filter(lambda p: p.requires_grad, model.parameters()))
        frozen_params = total_params - trainable_params
        logger.info('Total/Frozen Parameters: {}/{}'.format(total_params, frozen_params))
        # test model
        test(model, device, test_loaders[i], scalers[i], logger=logger)
        # save prompt and model
        save_prompt_weights(model, os.path.join(save_dir, f"first_round_prompt.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, f"first_round.pt"))
        # update weight list
        weight_histories = {name: [] for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())

    # train task 1
    # reset train
    optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()), 
                        lr=args.lr*0.01, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=[500], gamma=args.gamma)
    logger.info(f'Train for task 1')
    save_dir = 'checkpoints/{}/{}/{}/'.format(args.dataset,tasks[0],now)
    # load prompt
    load_prompt_weights(model, os.path.join(save_dir, f"start_prompt.pth"))
    # start train
    model, w_list = train(model, device, train_loaders[0], val_loaders[0], scalers[0], 
                    optimizer, scheduler, criterion, args.max_epochs, args.patience, logger=logger)
    # append weight
    for w in w_list:
        for name, param in w.items():
            weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())
    # cal var & frozen
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            variances = calculate_variance(weight_histories[name])
            if 'prompt' not in name and np.all(variances < threshold):
                param.requires_grad = False
    # print frozen params
    trainable_params = sum(param.nelement() for param in filter(lambda p: p.requires_grad, model.parameters()))
    frozen_params = total_params - trainable_params
    logger.info('Total/Frozen Parameters: {}/{}'.format(total_params, frozen_params))
    # test model
    test(model, device, test_loaders[0], scalers[0], logger=logger)
    # save prompt and model
    save_prompt_weights(model, os.path.join(save_dir, f"first_round_prompt.pth"))
    torch.save(model.state_dict(), os.path.join(save_dir, f"first_round.pt"))
    
    # fine-tuning
    logger.info(f'Fine Tuning')
    weights_star_dict = copy.deepcopy(model.state_dict())
    
    for i, task in enumerate(tasks):
        # load init weights
        model.load_state_dict(weights_star_dict)
        # reset train
        optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()), 
                         lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
        scheduler = MultiStepLR(optimizer, milestones=args.steps, gamma=args.gamma)
        logger.info(f'Train for task {i+1}')
        save_dir = 'checkpoints/{}/{}/{}/'.format(args.dataset,tasks[i],now)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # load prompt
        load_prompt_weights(model, os.path.join(save_dir, f"first_round_prompt.pth"))
        # start train
        model, _ = train(model, device, train_loaders[i], val_loaders[i], scalers[i], 
                        optimizer, scheduler, criterion, args.max_epochs, args.patience, logger=logger)
        # test model
        test(model, device, test_loaders[i], scalers[i], logger=logger)
        # save prompt and model
        save_prompt_weights(model, os.path.join(save_dir, f"fine_tuning_prompt.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, f"fine_tuning.pt"))
        
    # test model
    # test(model, device, test_loader, scaler, logger=logger)
