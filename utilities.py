# import sys
import numpy as np
import torch
from torch import Tensor, ones, stack, load
from torch.autograd import grad
# import pandas as pd
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy import stats
import matplotlib.pyplot as plt
# from pathlib import Path
import wandb
import time
from tesladatano import TeslaDatasetNo, TeslaDatasetNoStb

from mlp import MLP

# Set fixed random number seed
torch.manual_seed(1234)
np.random.seed(1234)


# write a checkpoint
def write_checkpoint(checkpoint_path, epoch, min_mlp_loss, optimizer, model):
    checkpoint = {}
    checkpoint["epoch"] = epoch
    checkpoint["minimum_mlp_loss"] = min_mlp_loss
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["mlp_model"] = model.state_dict()
    torch.save(checkpoint, checkpoint_path)
    
# function for getting charging statistics    
def get_prediction_regression(soc, battery_temperature, soc_end = 80):
    #CS_Tessa
    c_soc = - 0.4918029
    c_bat_temp = 1.309966
    const = 44.89049

    #training_data_2020.48.35.5
    #Rsq 0.596
    c_soc = -0.7344
    c_bat_temp = 0.7670
    const = 65.1319

    #training_data_2021.4.11
    c_soc = -0.9557
    c_bat_temp = 4.0903
    const = 23.4002

    charging_speed = c_soc * soc + c_bat_temp * battery_temperature + const

    #training_data_2021.6.20
    ## prediction: charging time
    c_soc_start = -0.4505
    c_soc_end = 0.7690
    c_bat_temp = -0.6267
    const = - 6.9412
    charging_time = c_soc_start*soc + c_soc_end*soc_end + c_bat_temp*battery_temperature + const

    ## prediction: peak charging speed
    c_soc_start = -0.6213
    c_soc_end = 0.0856
    c_bat_temp = 3.8553
    const = 31.0754
    charging_speed = c_soc_start*soc + c_soc_end*soc_end + c_bat_temp*battery_temperature + const

    return charging_speed, charging_time



# function for evaluating the performance on test data
def evaluate(model1,model2,model3,idd,rel_time,diff,normalize,device):
    import warnings
    warnings.filterwarnings('ignore')
    
    ds_test = TeslaDatasetNoStb(rel_time = rel_time, diff = diff,device = device, ID = idd, data = "test")
    print('Test data ID=', idd)
    #print('test data size', ds_test.df0.shape[0])
       
    ############# model1
    # Prediction accuracy of the Neural Operator
    #print('1.Prediction accuracy of the Neural Operator (NO)')
    
    lw=5
    t=ds_test.t
    begin = time.time()
    pred_der1 = model1(ds_test.x.to(device))
    pred_der1 = pred_der1.detach().cpu().numpy()/normalize
    true_der1 = ds_test.y.numpy()
    end = time.time()
    #print("time:", end - begin)

    #MAE
    mae_der1 = np.sum(np.abs(pred_der1- true_der1).mean(axis=None))
    #print('MAE1:', mae_der1)

    #MSE
    mse_der1 = ((true_der1 - pred_der1)**2).mean(axis=None)
    #print('MSE1:', mse_der1)

    #Relative error
    rel_error_der1 = np.linalg.norm(pred_der1 - true_der1) / np.linalg.norm(true_der1)*100
    #print('Relative error (%):', rel_error_der1)
    
    ################### model2
    t=ds_test.t
    begin = time.time()
    pred_der2 = model2(ds_test.x.to(device))
    pred_der2 = pred_der2.detach().cpu().numpy()/normalize
    true_der2 = ds_test.y.numpy()
    end = time.time()
    #print("time:", end - begin)

    #MAE
    mae_der2 = np.sum(np.abs(pred_der2- true_der2).mean(axis=None))
    #print('MAE2:', mae_der2)

    #MSE
    mse_der2 = ((true_der2 - pred_der2)**2).mean(axis=None)
    #print('MSE2:', mse_der2)

    #Relative error
    rel_error_der2 = np.linalg.norm(pred_der2 - true_der2) / np.linalg.norm(true_der2)*100
    #print('Relative error2 (%):', rel_error_der2)
    ######################################### model3
    
    t=ds_test.t
    begin = time.time()
    pred_der3 = model3(ds_test.x.to(device))
    pred_der3 = pred_der3.detach().cpu().numpy()/normalize
    true_der3 = ds_test.y.numpy()
    end = time.time()
    #print("time:", end - begin)

    #MAE
    mae_der3 = np.sum(np.abs(pred_der3- true_der3).mean(axis=None))
    #print('MAE3:', mae_der3)

    #MSE
    mse_der3 = ((true_der3 - pred_der3)**2).mean(axis=None)
    #print('MSE3:', mse_der3)

    #Relative error
    rel_error_der3 = np.linalg.norm(pred_der3 - true_der3) / np.linalg.norm(true_der3)*100
    #print('Relative error3 (%):', rel_error_der3)
    #########################
    
    
    #Plot
    parameters = {'axes.labelsize': 35,
           'axes.titlesize': 35,
           'legend.title_fontsize': 15,
            'axes.labelsize':20,
            'legend.fontsize':15,
            'xtick.labelsize':15,
             'ytick.labelsize':15}
    plt.rcParams.update(parameters)

#     plt.figure(figsize = (12, 8))
#     plt.plot(t, pred_der1, '-', label='LiFe-net (baseline) prediction', linewidth=lw)
#     plt.plot(t, pred_der2, '-', label='LiFe-net (regulariser) prediction', linewidth=lw)
#     plt.plot(t, pred_der3, '-', label='LiFe-net (time-stability) prediction', linewidth=lw)
#     plt.plot(t, true_der1, '--', label='Ground-truth', linewidth=lw)
#     plt.xlabel('time (s)')
#     plt.ylabel('ΔTemp/Δt (°C/s)')
#     plt.grid()
#     plt.legend(loc='best') 
#     plt.show()

    #3)Forward Euler method with fixed initial env. conditions but with updated 
    #Temperature (and rel time) from the prediction of the model at previous iteration
    #with generated temporally equidistant time steps

    #print('3.Forwad Euler method with fixed initial env conditions')

    rel_t = ds_test.rel_t
    
    ###################### model1
    # ground-truth time
    t=ds_test.t
    max_t = t.max()
    t=t.numpy()

    # Ground-truth temperature
    true_temp1 = ds_test.x[:,4].numpy()

    # Predicted temperature using model prediction and forward euler method
    pred_tempv1_1 = np.zeros((ds_test.x.shape[0]))
    pred_tempv1_1[0] = true_temp1[0].copy()

    # Fixed initial conditions for all environmental conditions
    input = ds_test.x[0].detach().clone()

    # temporally equdistant time steps
    tt = np.linspace(0,max_t,ds_test.x.shape[0])
    step_size=tt[2]-tt[1]

    #ODE
    begin = time.time()

    for i in range(0, ds_test.x.shape[0] - 1):
        input[4] = torch.tensor(pred_tempv1_1[i]).detach().clone()
        if rel_time == True:
            input[5] = torch.tensor(rel_t[i]).detach().clone()      
        pred = model1(input.to(device))
        pred = pred.detach().cpu().numpy()/normalize
        pred_tempv1_1[i + 1] = pred_tempv1_1[i] + pred*step_size
    end = time.time()

    #print("time:", end - begin)

    #MAE
    mae1 = np.sum(np.abs(pred_tempv1_1- true_temp1).mean(axis=None))
    #print('MAE1:', mae1)

    #MSE
    mse1 = ((true_temp1 - pred_tempv1_1)**2).mean(axis=None)
    #print('MSE1:', mse1)

    #Relative error
    rel_error1 = np.linalg.norm(pred_tempv1_1 - true_temp1) / np.linalg.norm(true_temp1)*100
    #print('Relative error (%):', rel_error1)
    
    
    ###################### model2
    # ground-truth time
    t=ds_test.t
    max_t = t.max()
    t=t.numpy()

    # Ground-truth temperature
    true_temp2 = ds_test.x[:,4].numpy()

    # Predicted temperature using model prediction and forward euler method
    pred_tempv1_2 = np.zeros((ds_test.x.shape[0]))
    pred_tempv1_2[0] = true_temp2[0].copy()

    # Fixed initial conditions for all environmental conditions
    input = ds_test.x[0].detach().clone()

    # temporally equdistant time steps
    tt = np.linspace(0,max_t,ds_test.x.shape[0])
    step_size=tt[2]-tt[1]

    #ODE
    begin = time.time()

    for i in range(0, ds_test.x.shape[0] - 1):
        input[4] = torch.tensor(pred_tempv1_2[i]).detach().clone()
        if rel_time == True:
            input[5] = torch.tensor(rel_t[i]).detach().clone()      
        pred = model2(input.to(device))
        pred = pred.detach().cpu().numpy()/normalize
        pred_tempv1_2[i + 1] = pred_tempv1_2[i] + pred*step_size
    end = time.time()

    #print("time:", end - begin)

    #MAE
    mae2 = np.sum(np.abs(pred_tempv1_2- true_temp2).mean(axis=None))
    #print('MAE2:', mae2)

    #MSE
    mse2 = ((true_temp2 - pred_tempv1_2)**2).mean(axis=None)
    #print('MSE2:', mse2)

    #Relative error
    rel_error2 = np.linalg.norm(pred_tempv1_2 - true_temp2) / np.linalg.norm(true_temp2)*100
    #print('Relative error (%):', rel_error2)
    
    
    
    ###################### model3
    # ground-truth time
    t=ds_test.t
    max_t = t.max()
    t=t.numpy()

    # Ground-truth temperature
    true_temp3 = ds_test.x[:,4].numpy()

    # Predicted temperature using model prediction and forward euler method
    pred_tempv1_3 = np.zeros((ds_test.x.shape[0]))
    pred_tempv1_3[0] = true_temp3[0].copy()

    # Fixed initial conditions for all environmental conditions
    input = ds_test.x[0].detach().clone()

    # temporally equdistant time steps
    tt = np.linspace(0,max_t,ds_test.x.shape[0])
    step_size=tt[2]-tt[1]

    #ODE
    begin = time.time()

    for i in range(0, ds_test.x.shape[0] - 1):
        input[4] = torch.tensor(pred_tempv1_3[i]).detach().clone()
        if rel_time == True:
            input[5] = torch.tensor(rel_t[i]).detach().clone()      
        pred = model3(input.to(device))
        pred = pred.detach().cpu().numpy()/normalize
        pred_tempv1_3[i + 1] = pred_tempv1_3[i] + pred*step_size
    end = time.time()

    #print("time:", end - begin)

    #MAE
    mae3 = np.sum(np.abs(pred_tempv1_3- true_temp3).mean(axis=None))
    #print('MAE3:', mae3)

    #MSE
    mse3 = ((true_temp3 - pred_tempv1_3)**2).mean(axis=None)
    #print('MSE3:', mse3)

    #Relative error
    rel_error3 = np.linalg.norm(pred_tempv1_3 - true_temp3) / np.linalg.norm(true_temp3)*100
    #print('Relative error (%):', rel_error3)
    
    ###

    #Plot
#     plt.figure(figsize = (12, 8))
#     plt.plot(tt, pred_tempv1_1, '-', label='LiFe-net (baseline) prediction', linewidth=lw)
#     plt.plot(tt, pred_tempv1_2, '-', label='LiFe-net (regulariser) prediction', linewidth=lw)
#     plt.plot(tt, pred_tempv1_3, '-', label='LiFe-net (time-stability) prediction', linewidth=lw)
#     plt.plot(t, true_temp1, '--', label='Ground-truth', linewidth=lw)
#     #plt.title('Prediction vs ground-truth for drive-ID = {} (temporally equidistant step size)'.format(idd))
#     plt.xlabel('time (s)')
#     plt.ylabel('Temperature (°C)')
#     plt.grid()
#     #plt.legend(loc='lower right')
#     plt.legend(loc='best') 
#     plt.show()


    #4)Forward Euler method with updated environmental conditions from the dataset at each iteration
    #But with updated temperature from the prediction of the model at previous iteration
    #with true step sizes
    #print('4.Forwad Euler method with updated env conditions from the dataset at each iteration with true step sizes')
     
        
    ##############################model1
    # time
    t=ds_test.t
    max_t = t.max()
    t=t.numpy()

    # Ground-truth temperature
    true_temp1 = ds_test.x[:,4].numpy()

    # Predicted temperature using model prediction and forward euler method
    pred_temp1 = np.zeros((ds_test.x.shape[0]))
    pred_temp1[0] = true_temp1[0].copy()


    begin = time.time()
    for i in range(0, ds_test.x.shape[0] - 1):
        input = ds_test.x[i].detach().clone()
        input[4] = torch.tensor(pred_temp1[i]).detach().clone()
        pred = model1(input.to(device))
        pred = pred.detach().cpu().numpy()/normalize
        pred_temp1[i + 1] = pred_temp1[i] + pred*(t[i+1]-t[i])
    end = time.time()
    #print("time:", end - begin)
    
    #MAE 
    mae_upd1 = np.sum(np.abs(pred_temp1- true_temp1).mean(axis=None))
    print('MAE1:', mae_upd1)

    #MSE
    mse_upd1 = ((true_temp1 - pred_temp1)**2).mean(axis=None)
    print('MSE1:', mse_upd1)

    # Relative error
    rel_error_upd1 = np.linalg.norm(pred_temp1 - true_temp1) / np.linalg.norm(true_temp1)*100
    print('Relative error1 (%):', rel_error_upd1)

    
    ##############################model2
    # time
    t=ds_test.t
    max_t = t.max()
    t=t.numpy()

    # Ground-truth temperature
    true_temp2 = ds_test.x[:,4].numpy()

    # Predicted temperature using model prediction and forward euler method
    pred_temp2 = np.zeros((ds_test.x.shape[0]))
    pred_temp2[0] = true_temp2[0].copy()


    begin = time.time()
    for i in range(0, ds_test.x.shape[0] - 1):
        input = ds_test.x[i].detach().clone()
        input[4] = torch.tensor(pred_temp2[i]).detach().clone()
        pred = model2(input.to(device))
        pred = pred.detach().cpu().numpy()/normalize
        pred_temp2[i + 1] = pred_temp2[i] + pred*(t[i+1]-t[i])
    end = time.time()
    #print("time:", end - begin)
    
    #MAE 
    mae_upd2 = np.sum(np.abs(pred_temp2- true_temp2).mean(axis=None))
    print('MAE2:', mae_upd2)

    #MSE
    mse_upd2 = ((true_temp2- pred_temp2)**2).mean(axis=None)
    print('MSE2:', mse_upd2)

    # Relative error
    rel_error_upd2 = np.linalg.norm(pred_temp2 - true_temp2) / np.linalg.norm(true_temp2)*100
    print('Relative error2 (%):', rel_error_upd2)
    
    
    ##############################model3
    # time
    t=ds_test.t
    max_t = t.max()
    t=t.numpy()

    # Ground-truth temperature
    true_temp3 = ds_test.x[:,4].numpy()

    # Predicted temperature using model prediction and forward euler method
    pred_temp3 = np.zeros((ds_test.x.shape[0]))
    pred_temp3[0] = true_temp3[0].copy()


    begin = time.time()
    for i in range(0, ds_test.x.shape[0] - 1):
        input = ds_test.x[i].detach().clone()
        input[4] = torch.tensor(pred_temp3[i]).detach().clone()
        pred = model3(input.to(device))
        pred = pred.detach().cpu().numpy()/normalize
        pred_temp3[i + 1] = pred_temp3[i] + pred*(t[i+1]-t[i])
    end = time.time()
    #print("time:", end - begin)
    
    #MAE 
    mae_upd3 = np.sum(np.abs(pred_temp3- true_temp3).mean(axis=None))
    print('MAE3:', mae_upd3)

    #MSE
    mse_upd3 = ((true_temp3- pred_temp3)**2).mean(axis=None)
    print('MSE3:', mse_upd3)

    # Relative error
    rel_error_upd3 = np.linalg.norm(pred_temp3 - true_temp3) / np.linalg.norm(true_temp3)*100
    print('Relative error3 (%):', rel_error_upd3)
    
    
    #print('Main PLOT')
    
    #Plot
    plt.figure(figsize = (12, 8))
    plt.plot(t, pred_temp1, '-', label='LiFe-net (baseline) prediction', linewidth=lw)
    plt.plot(t, pred_temp2, '-', label='LiFe-net (regulariser) prediction', linewidth=lw)
    plt.plot(t, pred_temp3, '-', label='LiFe-net (time-stability) prediction', linewidth=lw)       
#   plt.plot(t, pred_temp, '-', label='Prediction')
    plt.plot(t, true_temp1, '--', label='Ground-truth', linewidth=lw)
    #plt.title('Prediction (with updated env. conditions) vs ground-truth for drive-ID = {} (true step size)'.format(idd))
    plt.xlabel('time (s)')
    plt.ylabel('Temperature (°C)')
    plt.grid()
    #plt.legend(loc='lower right')
    plt.legend(loc='best') 
    plt.show()

    mae_arr1 = np.array([mae_der1, mae1, mae_upd1])
    mse_arr1 = np.array([mse_der1, mse1, mse_upd1])
    rel_arr1 = np.array([rel_error_der1, rel_error1, rel_error_upd1])
    
    mae_arr2 = np.array([mae_der2, mae2, mae_upd2])
    mse_arr2 = np.array([mse_der2, mse2, mse_upd2])
    rel_arr2 = np.array([rel_error_der2, rel_error2, rel_error_upd2])
    
    mae_arr3 = np.array([mae_der3, mae3, mae_upd3])
    mse_arr3 = np.array([mse_der3, mse3, mse_upd3])
    rel_arr3 = np.array([rel_error_der3, rel_error3, rel_error_upd3])
    
    return mae_arr1, mse_arr1, rel_arr1, mae_arr2, mse_arr2, rel_arr2, mae_arr3, mse_arr3, rel_arr3,pred_tempv1_1,pred_tempv1_2,pred_tempv1_3,pred_temp1,pred_temp2,pred_temp3,true_temp1,tt