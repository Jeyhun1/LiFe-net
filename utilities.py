import sys
import numpy as np
import torch
from torch import Tensor, ones, stack, load
from torch.autograd import grad
import pandas as pd
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy import stats
from pathlib import Path
import wandb
import time
from tesladatano import TeslaDatasetNo, TeslaDatasetNoStb

from mlp import MLP

# Set fixed random number seed
torch.manual_seed(1234)
np.random.seed(1234)


# write checkpoint
def write_checkpoint(checkpoint_path, epoch, min_mlp_loss, optimizer, model):
    checkpoint = {}
    checkpoint["epoch"] = epoch
    checkpoint["minimum_mlp_loss"] = min_mlp_loss
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["mlp_model"] = model.state_dict()
    torch.save(checkpoint, checkpoint_path)

# function for evaluating the performance on test data
def evaluate(model,idd,rel_time,diff,normalize,device):
    # import test data
    ds_test = TeslaDatasetNo(device = device, ID = idd, data = "test",rel_time = rel_time, diff = diff)

    # Prediction accuracy of the Neural Operator
    #print('Prediction accuracy of the Neural Operator (NO)')
    begin = time.time()
    pred_der = model(ds_test.x.to(device))
    pred_der = pred_der.detach().cpu().numpy()/normalize
    end = time.time()
    true_der = ds_test.y.cpu().numpy()


    # relative time
    t=ds_test.t

    #MAE
    mae_der = np.sum(np.abs(pred_der- true_der).mean(axis=None))
    #print('MAE:', mae_der)

    #MSE
    mse_der = ((true_der - pred_der)**2).mean(axis=None)
    #print('MSE:', mse_der)

    #Relative error
    rel_error_der = np.linalg.norm(pred_der - true_der) / np.linalg.norm(true_der)*100
    #print('Relative error (%):', rel_error_der)


    #print('########################################################')

    #3)Forward Euler method with fixed initial env. conditions but with updated 
    #Temperature (and rel time) from the prediction of the model at previous iteration
    #with generated temporally equidistant time steps

    #print('Forwad Euler method with fixed initial env conditions')
    rel_t = ds_test.rel_t

    # ground-truth time
    t=ds_test.t
    max_t = t.max()
    t=t.cpu().numpy()

    # Ground-truth temperature
    true_temp = ds_test.x[:,4].cpu().numpy()

    # Predicted temperature using model prediction and forward euler method
    pred_temp = np.zeros((ds_test.x.shape[0]))
    pred_temp = true_temp.copy()

    # Fixed initial conditions for all environmental conditions
    input = ds_test.x[0].detach().clone()

    # temporally equdistant time steps
    tt = np.linspace(0,max_t,ds_test.x.shape[0])
    step_size=tt[2]-tt[1]

    #ODE
    begin = time.time()

    for i in range(0, ds_test.x.shape[0] - 1):
        input[4] = torch.tensor(pred_temp[i]).detach().clone()
        if rel_time == True:
            input[5] = torch.tensor(rel_t[i]).detach().clone()
        pred = model(input.to(device))
        pred = pred.detach().cpu().numpy()/normalize
        pred_temp[i + 1] = pred_temp[i] + pred*step_size
    end = time.time()

    #print("time:", end - begin)

    #MAE
    mae = np.sum(np.abs(pred_temp- true_temp).mean(axis=None))
    #print('MAE:', mae)

    #MSE
    mse = ((true_temp - pred_temp)**2).mean(axis=None)
    #print('MSE:', mse)

    #Relative error
    rel_error = np.linalg.norm(pred_temp - true_temp) / np.linalg.norm(true_temp)*100
    #print('Relative error (%):', rel_error)


    #4)Forward Euler method with updated environmental conditions from the dataset at each iteration
    #But with updated temperature from the prediction of the model at previous iteration
    #with true step sizes
    #print('Forwad Euler method with updated env conditions from the dataset at each iteration with true step sizes')

    # time
    t=ds_test.t
    t=t.numpy()

    # max time
    max_t = t.max()

    # Ground-truth temperature
    true_temp = ds_test.x[:,4].cpu().numpy()

    # Predicted temperature using model prediction and forward euler method
    pred_temp = np.zeros((ds_test.x.shape[0]))
    pred_temp[0] = true_temp[0].copy()

    begin = time.time()
    for i in range(0, ds_test.x.shape[0] - 1):
        input = ds_test.x[i].detach().clone()
        input[4] = torch.tensor(pred_temp[i]).detach().clone()
        pred = model(input.to(device))
        pred = pred.detach().cpu().numpy()/normalize
        pred_temp[i + 1] = pred_temp[i] + pred*(t[i+1]-t[i])
    end = time.time()


    #print("time:", end - begin)
    #MAE 
    mae_upd = np.sum(np.abs(pred_temp- true_temp).mean(axis=None))
    #print('MAE:', mae_upd)

    #MSE
    mse_upd = ((true_temp - pred_temp)**2).mean(axis=None)
    #print('MSE:', mse_upd)

    # Relative error
    rel_error_upd = np.linalg.norm(pred_temp - true_temp) / np.linalg.norm(true_temp)*100
    #print('Relative error (%):', rel_error_upd)

    mae_arr = np.array([mae_der, mae, mae_upd])
    mse_arr = np.array([mse_der, mse, mse_upd])
    rel_arr = np.array([rel_error_der, rel_error, rel_error_upd])
    
    return mae_arr, mse_arr, rel_arr