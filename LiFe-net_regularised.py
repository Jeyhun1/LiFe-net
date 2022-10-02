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
from utilities import *
from tesladatano import TeslaDatasetNo, TeslaDatasetNoStb
from mlp import MLP


# Set fixed random number seed
torch.manual_seed(1234)
np.random.seed(1234)

# Use cuda if it is available, else use the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


hyperparameter_defaults = dict(
    alpha=0.1,
    normalize=1000,
    batch_size=4096,
    lr=1e-3,
    input_size=6,
    output_size=1,
    hidden_size=50,
    num_hidden=8,
    epochs=100,
    )


# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults, project="Neural_Operator_project", name='NO_reg-alpha')
# Access all hyperparameter values through wandb.config
config = wandb.config

# derivative
def derivative(x, u):

    grads = ones(u.shape, device=u.device) # move to the same device as prediction
    grad_u = grad(u, x, create_graph=True, grad_outputs=grads )[0]

    return grad_u


# Create instance of the dataset
ds = TeslaDatasetNo(diff = "fwd_diff", device = device, data ='train', normalize = config["normalize"], rel_time = True)
ds_test = TeslaDatasetNoStb(device = device, ID = -1, data = "test",normalize = config["normalize"], rel_time = True)
    
# bounds
lb = ds.lb
ub = ds.ub

# trainloader
train_loader = DataLoader(ds, batch_size=config["batch_size"],shuffle=True)
validloader = DataLoader(ds_test, batch_size=1,shuffle=True)

#model
model = MLP(input_size=config["input_size"],
                output_size=config["output_size"], 
                hidden_size=config["hidden_size"], 
                num_hidden=config["num_hidden"], 
                lb=lb, 
                ub=ub,
                activation = torch.relu
                )


model.to(device)

#Log the network weight histograms (optional)
wandb.watch(model)

# optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
criterion = torch.nn.MSELoss()

min_mlp_loss = np.inf
min_valid_loss = np.inf

x_data_plot=[]
y_data_all_plot=[]
y_data_1_plot=[]
y_data_2_plot=[]


if __name__ == '__main__':
    for epoch in range(config["epochs"]):
            # Set current and total loss value
            current_loss = 0.0
            total_loss = 0.0
            total_loss1 = 0.0
            total_loss2 = 0.0

            model.train()   # Optional when not using Model Specific layer
            for i, data in enumerate(train_loader,0):
                
                x_batch, y_batch = data
                if wandb.config.batch_size == 1:
                    x_batch=torch.squeeze(x_batch)
                    y_batch=torch.squeeze(y_batch)

                # Ground-truth temperature
                true_temp = x_batch[:,4].detach().clone()

                optimizer.zero_grad()
                x_batch.requires_grad=True #new
                pred = model(x_batch.to(device))

                u_deriv = derivative(x_batch,pred) #new
                loss1 = criterion(pred,y_batch.to(device))

                loss2 = torch.mean(u_deriv**2) * config["alpha"] 
                loss = loss1 + loss2 

                loss.backward()
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                total_loss += loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()


            train_loss = total_loss/(i+1)
            loss1 = total_loss1/(i+1) 
            loss2 = total_loss2/(i+1) 
            x_data_plot.append(epoch)
            y_data_all_plot.append(train_loss)      
            y_data_1_plot.append(loss1)
            y_data_2_plot.append(loss2)


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

                for j in range(0, x_batch.shape[0] - 1):
                    input0 = x_batch[j].detach().clone()
                    input0[4] = torch.tensor(pred_temp[j]).detach().clone()
                    pred = model(input0.to(device))/config["normalize"]
                    pred_temp[j + 1] = pred_temp[j] + pred*delta_t[j]

                loss = criterion(pred_temp.to(device),true_temp.to(device))

                # Calculate Loss
                valid_loss += loss.item()
            
            valid_loss_avg = valid_loss / (k+1)
            print(f'Epoch {epoch} \t Training Loss: {train_loss:.5f} \t Loss 1: {loss1:.5f} \t Loss 2: {loss2:.5f} \t Validation Loss: {valid_loss_avg:.5f}')    

            # uncomment for saving the best model and checkpoints during training
            # save best model
            if min_valid_loss > valid_loss_avg:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss_avg:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss_avg

                # Saving State Dict      
                model_name_path = Path('nomodel/best_model_{}_{}.pt'.format(wandb.run.id, wandb.run.name))
                torch.save(model.state_dict(), model_name_path, _use_new_zipfile_serialization=False)

            # writing checkpoint
            if (epoch + 1) % 20 == 0:
                checkpoint_path = Path('nomodel/checkpoint_{}_{}_{}.pt'.format(wandb.run.id, wandb.run.name, epoch))
                write_checkpoint(checkpoint_path, epoch, min_valid_loss, optimizer,model)

            # Log the loss and accuracy values at the end of each epoch
            wandb.log({
                "Epoch": epoch,
                "Total Loss": train_loss,
                "Loss1 (temperature)": loss1,
                "Loss2 (regulariser)": loss2,
                "Validation Loss": valid_loss_avg,
                "Min valid loss": min_valid_loss,
                })


    # Load the best model
    #PATH = 'nomodel/best_model_{}_{}.pt'.format(wandb.run.id, wandb.run.name)
