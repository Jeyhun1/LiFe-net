import sys
import numpy as np
import torch
from torch import Tensor, ones, stack, load
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
import pandas as pd
from torch.nn import Module
from torch.utils.data import DataLoader
from scipy import stats
from pathlib import Path
import wandb
import time
from utilities import *
from tesladatano import TeslaDatasetNoStb
from mlp import MLP

# Set fixed random number seed
torch.manual_seed(1234)
np.random.seed(1234)

# Use cuda if it is available, else use the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameter defaults
hyperparameter_defaults = dict(
    normalize=1000,
    batch_size=1,
    lr=1e-3,
    input_size=6,
    output_size=1,
    hidden_size=100,
    num_hidden=8,
    epochs=100,
    )

# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults, project="NO_tesla",
          name='NO_run_t-stb'
          )
# Access all hyperparameter values through wandb.config
config = wandb.config

# Create instance of the dataset
ds = TeslaDatasetNoStb(device = device, data ="train", normalize = config["normalize"], rel_time = True, diff = "fwd_diff")
ds_test = TeslaDatasetNoStb(device = device, ID = -1, data = "test",normalize = config["normalize"], rel_time = True, diff = "fwd_diff")

# trainloader
train_loader = DataLoader(ds, batch_size=config["batch_size"],shuffle=True)
validloader = DataLoader(ds_test, batch_size=1,shuffle=True)

model = MLP(input_size=config["input_size"],
            output_size=config["output_size"], 
            hidden_size=config["hidden_size"], 
            num_hidden=config["num_hidden"], 
            lb=ds.lb, 
            ub=ds.ub,
            activation = torch.relu)

model.to(device)

# Log the network weight histograms (optional)
wandb.watch(model)

# optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
criterion = torch.nn.MSELoss()


########################################
#Training of the Neural Operator based on time stability loss
########################################

min_mlp_loss = np.inf
min_valid_loss = np.inf

x_data_plot=[]
y_data_all_plot=[]
y_data_1_plot=[]
y_data_2_plot=[]

# Set fixed random number seed
torch.manual_seed(1234)

if __name__ == '__main__':
    
    begin = time.time()
    for epoch in range(config["epochs"]):
        
        print(f'Starting epoch {epoch}')
        # Set current and total loss value
        current_loss = 0.0
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0

        model.train()  
        for i, data in enumerate(train_loader,0):

            x_batch, y_batch, delta_t,rel_t = data

            x_batch = torch.squeeze(x_batch, 0)
            y_batch = torch.squeeze(y_batch, 0)
            delta_t = torch.squeeze(delta_t, 0)
            rel_t = torch.squeeze(rel_t, 0)

            # Ground-truth temperature
            true_temp = x_batch[:,4].detach().clone()

            # Initial condition
            input0 = x_batch[0].detach().clone()

            # Predicted temperature using model prediction and forward euler method
            pred_temp = torch.zeros(x_batch.shape[0])
            pred_temp[0]=true_temp[0].detach().clone().to(device)

            optimizer.zero_grad()

            for j in range(0, x_batch.shape[0] - 1):
                input0 = x_batch[j].detach().clone()
                input0[4] = torch.tensor(pred_temp[j]).detach().clone()
                pred = model(input0.to(device))/wandb.config.normalize
                pred_temp[j + 1] = pred_temp[j] + pred*delta_t[j]

            loss = criterion(pred_temp.to(device),true_temp.to(device))

            loss.backward()
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            total_loss += loss.item()

        train_loss = total_loss/(i+1)
        x_data_plot.append(epoch)
        y_data_all_plot.append(train_loss)

        # validation
        valid_loss = 0.0
        model.eval()  
        for k, data in enumerate(validloader,0):
            
            x_batch, y_batch, delta_t,rel_t = data
            x_batch = torch.squeeze(x_batch, 0)
            y_batch = torch.squeeze(y_batch, 0)
            delta_t = torch.squeeze(delta_t, 0)
            rel_t = torch.squeeze(rel_t, 0)

            # Ground-truth temperature
            true_temp = x_batch[:,4].detach().clone()

            input0 = x_batch[0].detach().clone()

            # Predicted temperature using model prediction and forward euler method
            pred_temp = torch.zeros(x_batch.shape[0])
            pred_temp[0]=true_temp[0].detach().clone().to(device)

            for l in range(0, x_batch.shape[0] - 1):
                input0 = x_batch[l].detach().clone()
                input0[4] = torch.tensor(pred_temp[l]).detach().clone()
                pred = model(input0.to(device))/wandb.config.normalize
                pred_temp[l + 1] = pred_temp[l] + pred*delta_t[l]

            loss = criterion(pred_temp.to(device),true_temp.to(device))

            # Calculate Loss
            valid_loss += loss.item()

        valid_loss_avg = valid_loss / (k+1)
        print(f'Epoch {epoch} \t Training Loss: {train_loss:.5f} \t Validation Loss avg: {valid_loss_avg:.5f} \t Validation Loss: {valid_loss:.5f}')    

        # uncomment for saving the best model and writing checkpoints during training
        # save best model
        if min_valid_loss > valid_loss_avg:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss_avg:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss_avg
            # Saving State Dict
            model_name_path = Path('nostb/best_model_stb_{}_{}.pt'.format(wandb.run.id, wandb.run.name))
            torch.save(model.state_dict(), model_name_path, _use_new_zipfile_serialization=False)

        # writing checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = Path('nostb/checkpoint_stb_{}_{}_{}.pt'.format(wandb.run.id, wandb.run.name, epoch))
            write_checkpoint(checkpoint_path, epoch, min_valid_loss, optimizer, model)


        # Log the loss and accuracy values at the end of each epoch
        wandb.log({
            "Epoch": epoch,
            "Total train Loss": train_loss,
            "Validation Loss": valid_loss_avg,
            "Min valid loss": min_valid_loss,
            })
    end = time.time()

    print("training time:", end - begin)


# Import the best model
# PATH = '/nostb/best_model_stb_{}_{}.pt'.format(wandb.run.id, wandb.run.name)
# model.load_state_dict(torch.load(PATH))
# model.eval()