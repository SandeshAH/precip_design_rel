#########################################################################
# --------------------------------------------------------------------- #
# | Author      : Sandesh Athni Hiremath                                |
# |---------------------------------------------------------------------|
# | Description :                                                       |
# |   This script is used to train a deep neural network (DNN) model    |
# |   for the optimal design of a pH-driven precipitation process.      |
# |   The model is implemented in PyTorch, and training is handled via  |
# |   dataLoader and optimization utilities. The script also includes   |
# |   functions for plotting the results and saving the model.          |
# |---------------------------------------------------------------------|
# | Usage       : python precipitation_dnn_data_fit.py                  |
# |                --mode [train|test|cmp] [--model_type gru|ann] ...   |
# |   --mode train : Run training for the selected model (GRU or ANN)   |
# |   --mode test  : Run testing/inference for the selected model       |
# |   --mode cmp   : Run comparison mode for both GRU and ANN models    |
# |   --model_type : Specify 'gru' or 'ann' (default: gru)              |
# |   See --help for more options.                                      |
# --------------------------------------------------------------------- #
#########################################################################

 
import torch
import torch.nn as nn

from torch.utils.data import Dataset
import torch.optim as optim
from pytorch_model_summary  import summary

import numpy as np
from matplotlib import pyplot as plt
from IPython import display
import pickle
from tqdm import tqdm
import argparse
import os

import warnings
warnings.simplefilter('ignore')

import sys
sys.path.append('./')
sys.path.append('../')

from src.utils import *
from src.data_utils import *

from src.utils import getFBDiffOps1D, npSps2Torch, get1DLapOp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())

setup_figure()

mse_loss = torch.nn.MSELoss(reduction='none')
ref_par_dict = {'1':[1, 1, 1e2, .5, 1, .02, .002, 0.0005, .001],
                '2':[1, 1, 1e2, 1, 1, .015, .0075, .002, .003],
                '3':[1, 1, 1e2, .1, 1, .019, .02, 0.0001, .001],
                '4':[ 1, 1, 1e2, 1, 1, .038, .018, 0.0004, .001]}


ser_ndf1, ser_ndf2, ser_ndf3, ser_ndf4 = proc_data()
ser_ndf1.columns, ser_ndf1.shape

# dataset and dataloaders
class MultiDataFrameDataset(Dataset):
    def __init__(self, dataframes, sequence_length):
        """
        Args:
            dataframes (list of pd.DataFrame): List of dataframes to sample from.
            sequence_length (int): Length of the sequence to return.
        """
        self.dataframes = dataframes
        self.sequence_length = sequence_length
        self.total_samples = sum(int(len(df) - sequence_length) for df in dataframes)
        self.cumulative_lengths = [0]
        
        # Calculate cumulative lengths for indexing
        for df in dataframes:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(df) - sequence_length)

        #print(self.cumulative_lengths)

    def __len__(self):
        # Number of samples is the total number of rows across all dataframes
        return self.total_samples
    
    def __getitem__(self, idx):
        # Find which dataframe the index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                df_idx = i - 1
                row_idx = idx - self.cumulative_lengths[df_idx]
                break
        
        # Select the dataframe
        df = self.dataframes[df_idx]
        
        # Ensure we can sample a full sequence
        if row_idx + self.sequence_length > len(df):
            row_idx = len(df) - self.sequence_length
        
        # Extract the sequence
        sequence = df.to_numpy()[row_idx:row_idx + self.sequence_length]
        #print(sequence.dtype, sequence.shape)
        # Convert to tensor
        return torch.tensor(sequence, dtype=torch.float32)

# Model definitions
class ANN(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, output_size=2, num_layers=5, seq_len=64):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size*seq_len, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, seq_len*output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (b, N, 3)
        #x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        out = self.tanh(x)
        #out = x.view(x.size(0), -1, 3)
        return out

class GRUNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=2, num_layers=3, seq_len=64):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (b, N, 3)
        #print(x.shape)
        x, h = self.gru(x)  # out: (b, N, 2*hidden_size)
        #print(x.shape, h.shape)
        x = self.fc(x)    # out: (b, N, 2)
        #print(x.shape)
        out = self.tanh(x)
        #out = x.view(x.size(0), -1, 2)
        return out

# Function to print model info
def print_model_stats(GRU=True):

    if GRU:
        gru_model = GRUNetwork()
        input_tensor = torch.randn(32, 64, 2)  # Example input with shape (b=32, N=100, 2)
    else:
        ann_model = ANN()
        input_tensor = torch.randn(32, 64, 2).reshape(-1,64*2)  # Example input with shape (b=32, N=100, 2)


    model = gru_model if GRU else ann_model

    # Print model summary
    print(summary(model, torch.zeros_like(input_tensor), show_input=False, show_hierarchical=True))

    # Calculate total parameters and size in memory
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.element_size() * p.numel() for p in model.parameters())  # in bytes
    param_size_mb = param_size / (1024 ** 2)  # convert to MB


    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size in Memory: {param_size_mb:.2f} MB")
    output = model(input_tensor)
    print(output.shape)  # Should print: torch.Size([32, 100, 5])


# Forward propagation function
def precip_forward_solve(batch, par_vec, init_vals=None, DEBUG=False):
    ##################
    ph_exp_val = batch[:,:,1]
    dph_exp_val = torch.zeros_like(batch[:,:,1],device=batch.device);
    dph_exp_val[:,1:] = ph_exp_val[:,1:] - ph_exp_val[:,:-1]
    dph = dph_exp_val.mean(axis=(0)).to(batch.device)
    mg_ic_exp_val = batch[:,:,2]
    ca_ic_exp_val = batch[:,:,3]
    ca_exp_val = batch[:,:,4]
    exp_typ = batch[:,:,7].mean(axis=1).to(batch.device)
    exp_ca0 = batch[:,:,8].to(batch.device)
    exp_c0 = exp_ca0 - ca_ic_exp_val
    ph_shift_const = batch[:,0,9].to(batch.device)
    
    ############    
    dt      = torch.tensor(.01)
    dx      = torch.tensor(.1)  
    Nx      = 64
    x_ax    = torch.tensor(dx * torch.arange(Nx)).to(dph.device)
    f0      = (6 / (2 * np.pi)) * torch.exp(-6e0 * (1.5 - x_ax)**2)
    alp     = .1
    #print(f0.shape, f0.max().item())
    ############## Initial values for MPC style forward solve
    exp_c0, f0 = init_vals if init_vals is not None else (exp_c0, f0.to(batch.device))
    
    ############
    Dxf, Dxb, Dxfab, Dxbab = getFBDiffOps1D(Nx, True)
    D_xx = npSps2Torch(get1DLapOp(Nx))  # lhs operator
    
    #############
    C       = [exp_c0[:,0].to(dph.device)]
    Csolid  = [exp_c0[:,0].to(dph.device)]
    Ca      = [ca_ic_exp_val[:,0]]
    V       = [v0*torch.ones_like(ph_exp_val[:,0]).to(dph.device)]
    simpH   = [ph_exp_val[:,0]]
    simH    = [torch.float_power(10,-simpH[0])]
    pH      = [ph_exp_val[:,0]]
    H       = [torch.float_power(10,-(pH[0]))]
    F       = [(f0/f0.max().item()).expand(batch.size(0), -1)]
    G_arr   = [torch.zeros_like(ph_exp_val[:,0]).to(dph.device)]
    M_arr   = [torch.zeros_like(ph_exp_val[:,0]).to(dph.device)]
    
    
    #############
    par_vec_ref = torch.zeros((batch.size(0),9),device=batch.device)
    #par_vec_ref[:,:3] = torch.tensor([1,1,1e2]).view(1,3).expand(batch.size(0),3)
    exp1_batch = exp_typ==1.0; par_vec_ref[exp1_batch,:] = torch.tensor(ref_par_dict['1'],device=batch.device).view(1,9)#.expand(batch.size(0),9)
    exp2_batch = exp_typ==2.0; par_vec_ref[exp2_batch,:] = torch.tensor(ref_par_dict['2'],device=batch.device).view(1,9)#.expand(batch.size(0),9)
    exp3_batch = exp_typ==3.0; par_vec_ref[exp3_batch,:] = torch.tensor(ref_par_dict['3'],device=batch.device).view(1,9)#.expand(batch.size(0),9)
    exp4_batch = exp_typ==4.0; par_vec_ref[exp4_batch,:] = torch.tensor(ref_par_dict['4'],device=batch.device).view(1,9)#.expand(batch.size(0),9)
    p1,p2,p3,p4,p5,p6,p7,p8,p9 = par_vec_ref[:,0],par_vec_ref[:,1],par_vec_ref[:,2],par_vec_ref[:,3],par_vec_ref[:,4],par_vec_ref[:,5],par_vec_ref[:,6],par_vec_ref[:,7],par_vec_ref[:,8]

    #p1,p2,p3,p4,p5,p6,p7,p8,p9 = par_vec[:,:,0],par_vec[:,:,1],par_vec[:,:,2],par_vec[:,:,3],par_vec[:,:,4],par_vec[:,:,5],par_vec[:,:,6],par_vec[:,:,7],par_vec[:,:,8]
    
    #print(dph_exp_val.shape, p5.shape)
    U_ph_ref = dph_exp_val
    U_r_ref =  p5.view(-1,1)*torch.clamp(p6.view(-1,1)*U_ph_ref,min=-p8.view(-1,1),max=p7.view(-1,1))

    U_ph = par_vec[:,:,0]
    U_r =  1*U_ph*par_vec[:,:,1]

    #print(U_ph.shape, U_r.shape)
    
    #p1, p2, p3, p4 = 1, 1, 1e2, 10
    ###################
    sig_C      = .025;  dW1 = torch.randn_like(ph_exp_val)
    sig_Ca     = .0005; dW2 = torch.randn_like(ph_exp_val)
    sig_H      = .001;  dW3 = torch.randn_like(ph_exp_val) 
    
    for k in range(0,len(dph)-1):
        dpH_k     = U_ph[:,k].to(dph.device)
        dCa_k     = U_r[:,k].to(dph.device)
        #print(pH[-1].shape,dpH_k.shape, F[-1].shape) 
        
        preV    = V[-1] + dt*vdot_k #1e2*torch.pow(10,dpH_k-7) * dt
        ph      = torch.clamp((pH[-1])*(1 + torch.sqrt(dt) * sig_H * dW3[:,k]) + 1e2 * dpH_k * dt,min=0,max=14) #+ ph_shift_const
        preH    = torch.float_power(10,-ph)
        ph_adj  = ph + ph_shift_const
        
        #print(F[0].shape, Ca[0].shape,H[0].shape,V[0].shape)
        X_vec = F[-1], C[-1], Ca[-1], pH[-1] + ph_shift_const, V[-1]
        
        # Nucleation rate
        J_vec = N_fn(C[-1],H[-1])
        #J, dJ_dC, dJ_dH = J_vec
        
        # Growth rate
        G_vec = a_fn(C[-1],H[-1])
        Gt, dG_dC, dG_dH = G_vec

        #print(cts.shape, csk.shape, F[-1].shape)
        # Birth rate
        rhsF, drhsF_dF, drhsF_dC, drhsF_dCa, drhsF_dH = rhsF_fn(X_vec, J_vec) #(F[-1] * J.view((-1,1))).to(torch.float)
        #print(x_ax.shape, F[-1].shape)
        St      = St_fn(F[-1],x_ax)
        St_star = St_star_fn(F[-1],x_ax)
        St_vec  = (St, St_star)

        # Advection operator
        h = (dt / dx)  # Courant number (dt/self.dx)
        adOp = (0.5 * ((Dxbab - Dxb * h) - (Dxfab - Dxf * h)) * h).to(f0.device)

        # Update particle size distribution
        preF = torch.abs((F[-1].squeeze() + (adOp.matmul(F[-1].T).T * Gt.view((-1,1))) + dt * rhsF.squeeze())).to(torch.float)
        epsilon = 1e-18  # Small value to avoid division by zero
        nomralized_F = preF / (preF.max(dim=1).values.view(-1,1) + epsilon)
        #print(preF.shape, preF.max(dim=1).values.shape)

        # Update Ca concentration
        rhsCa, drhsCa_dF, drhsCa_dC, drhsCa_dCa, drhsCa_dH = rhsCa_fn(X_vec, dCa_k)
        preCa       = torch.clamp(Ca[-1]*(1 + torch.sqrt(dt)*sig_Ca*dW2[:,k]) + dt * rhsCa  ,min=0,max=1)
        
        # Update CaCO3 concentration
        rhsC, drhsC_dF, drhsC_dC, drhsC_dCa, drhsC_dH    =  rhsC_fn(X_vec, J_vec, G_vec, St_vec, dCa_k)
        noise_term = C[-1] * sig_C * dW1[:,k]
        preC = torch.max(((C[-1] + dt *  rhsC + torch.sqrt(dt) * noise_term)), 0*rhsC) 
        
        
        
        #print("preC:", preC.shape, preF.shape)
        # Collecting data
        F.append(nomralized_F)#.detach().cpu().numpy()
        C.append(preC)#.detach().cpu().numpy()
        Ca.append(preCa)
        pH.append(ph)
        H.append(preH)
        simH.append(torch.float_power(10,-simpH[-1]))
        simpH.append(simpH[-1] + dph_exp_val[:,k])
        V.append(preV)
        
        G_arr.append(Gt)
        M_arr.append(St)#.item()
    
    F = torch.stack(F).permute(1,0,2)
    C = torch.stack(C).permute(1,0)
    Ca = torch.stack(Ca).permute(1,0)
    CO3 = Ca.clone()
    pH = torch.stack(pH).permute(1,0)
    H = torch.stack(H).permute(1,0)
    simpH = torch.stack(simpH).permute(1,0)
    simH = torch.stack(simH).permute(1,0)
    V = torch.stack(V).permute(1,0)
    G_arr = torch.stack(G_arr).permute(1,0)
    M_arr = torch.stack(M_arr).permute(1,0)
    
    loss_dph = mse_loss(U_ph, dph_exp_val).sum(dim=1).mean()
    loss_ph = mse_loss(pH,simpH).sum(dim=1).mean()
    loss_H = mse_loss(H,simH).sum(dim=1).mean()
    loss_ca = mse_loss(1e3*Ca[:,::1], 1e3*ca_ic_exp_val[:,::1]).sum(dim=1).mean()
    loss_ref_clp = mse_loss(U_r[:,::1], U_r_ref[:,::1]).sum(dim=1).mean()
    
    if DEBUG:
        print(loss_dph, loss_ph, loss_H, loss_ca, loss_ref_clp)
    
    loss = 1*loss_dph + 10*loss_ph + 1*loss_ca + 1*loss_ref_clp
    
    return F, C, Ca, G_arr, CO3, pH, simpH, simH, H, U_ph, U_r, U_r_ref, loss

# Define training function
def train_model(model, dataloader, optimizer, args):
    seq_len = args.seq_len
    device = args.device
    
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        batch = batch.to(device)
        if GRU:
            input_tensor = batch[:,:,-2:].to(device)
            par_vec = model(input_tensor)
        else:
            input_tensor = batch[:,:,-2:].to(device).reshape(-1,seq_len*2)
            par_vec = model(input_tensor).reshape(-1,seq_len,2)
        F, C, Ca, Csolid, CO3, pH, simpH, simH, H, U_ph, U_r, U_r_ref, loss = precip_forward_solve(batch,par_vec)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return batch, par_vec, F, C, Ca, Csolid, CO3, pH, simpH, simH, H, U_ph, U_r, U_r_ref, total_loss / len(dataloader)

# Define validation function
def validate_model(model, dataloader, args):
    seq_len = args.seq_len
    device = args.device
    
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(device)
            if GRU:
                input_tensor = batch[:,:,-2:].to(device)
                par_vec = model(input_tensor)
            else:
                input_tensor = batch[:,:,-2:].to(device).reshape(-1,seq_len*2)
                par_vec = model(input_tensor).reshape(-1,seq_len,2)
            F, C, Ca, Csolid, CO3, pH, simpH, simH, H, U_ph, U_r, U_r_ref, loss = precip_forward_solve(batch,par_vec)

            total_loss += loss.item()
    return batch, par_vec, F, C, Ca, Csolid, CO3, pH, simpH, simH, H, U_ph, U_r, U_r_ref, total_loss / len(dataloader)

# exponential smoothing function for loss data
def exp_smth(data, alpha=0.1):
    smoothed_data = []
    smoothed_value = data[0]  # Initialize with the first value
    for value in data:
        smoothed_value = alpha * value + (1 - alpha) * smoothed_value
        smoothed_data.append(smoothed_value)
    return smoothed_data

# Plotting functions
def plot_test_results(res, GRU, losses=None):
    h = 1
    batch, par_vec, F, C, Ca, Csolid, CO3, pH, simpH, simH, H, U_ph, U_r, U_r_ref, loss = res
    time = batch[:,:,0].cpu().numpy()
    ph_exp_val = batch[:,:,1].cpu().numpy()
    ca_ic_exp_val = batch[:,:,3].cpu().numpy()
    train_loss, val_loss = losses if losses is not None else (None, None)
    
    sup = 'g' if GRU else 'a'
    s = 2 if GRU else 2
    e = 3 if GRU else 3
    s1 = s+1; e1 = e+1
    res = (time[::h,s1:e1].mean(axis=1).flatten()[::h],ph_exp_val[::h,s1:e1].mean(axis=1).flatten()[::h],ca_ic_exp_val[::h,s1:e1].mean(axis=1).flatten()[::h],\
            pH[::h,s1:e1].mean(dim=1).flatten().cpu().numpy(),  Ca[::h,s1:e1].mean(dim=1).flatten().cpu().numpy(),\
                par_vec[::h,s:e,0].mean(dim=1).flatten().cpu().numpy()[::h], par_vec[::h,s:e,1].mean(dim=1).flatten().cpu().numpy()[::h],\
                    U_r_ref[::h,s:e].mean(dim=1).flatten().cpu().numpy()[::h], train_loss, val_loss)
    
    #ph_shift_const = batch[:,:,7].cpu().numpy()
    plt.figure(figsize=(18, 8),facecolor='white')
    #plt.rcParams.update({'axes.titlesize': 20, 'axes.labelsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20})
    plt.subplot(221);
    plt.plot(time[::h,s1:e1].mean(axis=1).flatten()[::5*h]/60, ph_exp_val[::h,s1:e1].mean(axis=1).flatten()[::5*h],'-o',color='magenta',linewidth=2,mew=2,alpha=.5,mfc='none',label='$\\bar H')
    plt.plot(time[::h,s1:e1].mean(axis=1).flatten()/60, pH[::h,s1:e1].mean(dim=1).flatten().cpu().numpy(),'-',color='tab:cyan',lw=3,label=f'$\hat H^{sup}$')	
    plt.xlabel('Time $t~(\\rm min)$'); plt.ylabel(f"$\\bf H$"); plt.title("Acidity index as a function of time"); 
    plt.grid(True)
    plt.legend(loc='lower right')

    plt.subplot(223);
    plt.plot(time[::h,s1:e1].mean(axis=1).flatten()[::5*h]/60, ca_ic_exp_val[::h,s:e].mean(axis=1).flatten()[::5*h],'-o',color='magenta',linewidth=2,mew=2,alpha=.5,mfc='none',label='$\\bar Q$')
    plt.plot(time[::h,s1:e1].mean(axis=1).flatten()/60, Ca[::h,s:e].mean(dim=1).flatten().cpu().numpy(),'-',color='tab:cyan',lw=3,label=f"$\hat Q^{sup}$")
    plt.xlabel('Time $t~(\\rm min)$'); plt.ylabel(f"$\\bf Q$"); plt.title("[$Ca^{+2}$] as a function of time"); 
    plt.grid(True)
    plt.legend()

    plt.subplot(222);
    ur = 1 * (par_vec[::h,s:e,0]*par_vec[::h,s:e,1])
    plt.plot(time[::h,s:e].mean(axis=1).flatten()[::h]/60, (ur).mean(dim=1).flatten().cpu().numpy()[::h],'g-',lw=3,label=f'$\hat U^{sup}_r$')
    plt.plot(time[::h,s:e].mean(axis=1).flatten()[::h]/60, 0*par_vec[::h,s:e,1].mean(dim=1).flatten().cpu().numpy()[::h],'k--',alpha=.5,lw=5,); 
    plt.ylim([max(-.001,ur.flatten().cpu().numpy().min()), min(.001,ur.flatten().cpu().numpy().max())])
    
    plt.xlabel('Time $t~(\\rm min)$'); plt.ylabel(f"$\\bf U_r$", labelpad=-15); plt.title("Rate modulation function")
    plt.grid(True)
    plt.legend()

    if losses is not None:
        train_loss, val_loss = losses
        plt.subplot(224);
        plt.plot(train_loss, '-b', lw=3, alpha=.75, label='Train Loss')
        plt.plot(val_loss, '-', color='tab:red', lw=3, alpha=.75, label='Validation Loss')
        plt.grid(True)
        ylim = 2000 if GRU else 300
        plt.ylim([0,ylim]) 
        plt.legend()
        plt.xlabel('Epochs'); plt.ylabel("Loss value"); plt.title("Training and Validation Loss")
    
    plt.subplots_adjust(hspace=0.45,wspace=.2)
    #plt.pause(.1)
    return res

def test_chunks_in_parallel(model, np_batches, losses, args):
    model_typ = args.model_type
    model_file = args.model_file
    seq_len = args.seq_len
    device = args.device
    exp_typ = args.exp_typ
    GRU = model_typ == 'gru'
    model.eval()
    total_loss = 0.0
    h = 5
    train_loss, val_loss = losses
    setup_figure()
    with torch.no_grad():
        fig, axs = plt.subplots(2,2,figsize=(18, 8),gridspec_kw=dict(hspace=.45, wspace=.2),facecolor='white'); axs = axs.flatten()
        #plt.figure(figsize=(18, 8))
        #axs = [plt.subplot(221), plt.subplot(222), plt.subplot(223), plt.subplot(224)]
        plt.rcParams.update({'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
        batch = torch.tensor(np_batches).float().to(device)
        if GRU:
            input_tensor = batch[:,:,-2:].to(device)
            par_vec = model(input_tensor)
        else:
            input_tensor = batch[:,:,-2:].to(device).reshape(-1,seq_len*2)
            par_vec = model(input_tensor).reshape(-1,seq_len,2)
        F, C, Ca, Csolid, CO3, pH, simpH, simH, H, U_ph, U_r, U_r_ref, loss = precip_forward_solve(batch,par_vec)
        total_loss += loss.item()
        #display.clear_output(wait=True)
        #print(f"Epoch {epoch + 1}/{num_epochs}")
        #print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        time = batch[:,:,0].cpu().numpy().flatten()
        ph_exp_val = batch[:,:,1].cpu().numpy().flatten()
        ca_ic_exp_val = batch[:,:,3].cpu().numpy().flatten()
        ph_shift_const = batch[:,:,7].cpu().numpy().flatten()
        
        pred_pH = pH.cpu().numpy().flatten()
        pred_ca = Ca.cpu().numpy().flatten()
        pred_u_ph = par_vec[:,:,0].cpu().numpy().flatten()
        pred_u_r = par_vec[:,:,1].cpu().numpy().flatten()
        u_r_np = U_r.cpu().numpy().flatten()
        u_r_ref_np = U_r_ref.cpu().numpy().flatten()

        sup = 'g' if GRU else 'a'
        s = 0 if GRU else 0
        e = time.shape[0]-2 if GRU else time.shape[0]-2
        s1 = s+1; e1 = e+1

        res = (time[s1:e1:h], ph_exp_val[s1:e1:h], ca_ic_exp_val[s1:e1:h], pred_pH[s1:e1:h], pred_ca[s1:e1:h], pred_u_ph[s:e:h], pred_u_r[s:e:h], u_r_ref_np[s:e:h], train_loss, val_loss)
        #print(time.shape, ph_exp_val.shape, C.shape, Ca.shape, Csolid.shape, pH.shape, simpH.shape, simH.shape, H.shape)
        #print(par_vec[::h,:,])
        axs[0].plot(time[s1:e1:2*h]/60, ph_exp_val[s1:e1:2*h],'-o',color='magenta',linewidth=2,mew=2,alpha=.5,mfc='none',label=f'$\\bar H$');
        axs[0].plot(time[s1:e1:h]/60, pred_pH[s1:e1:h],'-',color='tab:cyan',lw=3,label=f'$\hat H^{sup}$'); axs[0].grid(True)
        axs[0].set_xlabel('Time $t~(\\rm min)$'); axs[0].set_ylabel("H"); axs[0].set_title("Acidity index as a function of time");axs[0].legend(loc='lower right')

        axs[2].plot(time[s1:e1:2*h]/60, ca_ic_exp_val[s1:e1:2*h],'-o',color='magenta',linewidth=2,mew=2,alpha=.5,mfc='none',label=f'$\\bar Q$')
        axs[2].plot(time[s1:e1:h]/60, pred_ca[s1:e1:h],'-',color='tab:cyan',lw=3,label=f'$\hat Q^{sup}$'); axs[2].grid(True)
        axs[2].set_xlabel('Time $t~(\\rm min)$'); axs[2].set_ylabel("Q"); axs[2].set_title("[$Ca^{+2}$] as a function of time");axs[2].legend()
        
        #h = 5
        ur = 1 * (pred_u_ph[s:e:h]*pred_u_r[s:e:h])
        axs[1].plot(time[s:e:h]/60, (pred_u_ph[s:e:h]*pred_u_r[s:e:h]),'g-',lw=3,label=f'$U^{sup}_r$')
        axs[1].plot(time[s:e:h]/60, 0*pred_u_r[s:e:h],'k--',alpha=.5,lw=5,);axs[1].legend(); axs[1].grid(True)
        axs[1].set_xlabel('Time $t~(\\rm min)$'); axs[1].set_ylabel("${\\bf U_r}$",labelpad=-15); axs[1].set_title("Rate modulation function")
        axs[1].set_ylim([-.001,min(.001,ur.max())])
        
        axs[1].legend()


        axs[3].plot(train_loss, '-b', lw=3, alpha=.75, label='Train Loss')
        axs[3].plot(val_loss, '-', color='tab:red', lw=3, alpha=.75, label='Validation Loss')
        axs[3].set_xlabel('Epochs'); axs[3].set_ylabel("Loss value");
        axs[3].grid(True)
        ylim = 2000 if GRU else 300
        axs[3].set_ylim([0,ylim])
        axs[3].set_title("Training and Validation Loss")
        axs[3].legend()


    plt.savefig(f'../results/precip_design/figures/{model_typ}_{model_file}_bat_pred_{exp_typ}.png', bbox_inches='tight', pad_inches=0.1)
    
    return total_loss / len(batch), res
    

def plot_states(res, GRU):
    h = 1
    batch, par_vec, F, C, Ca, G_arr, CO3, pH, simpH, simH, H, U_ph, U_r, U_r_ref, loss = res
    time = batch[:,:,0].cpu().numpy()
    ph_exp_val = batch[:,:,1].cpu().numpy()
    ca_ic_exp_val = batch[:,:,3].cpu().numpy()
    print(F.shape)
    
    sup = 'g' if GRU else 'a'
    s = 2 if GRU else 2
    e = 3 if GRU else 3
    s1 = s+1; e1 = e+1
    res = (time[::h,s1:e1].mean(axis=1).flatten()[::h],ph_exp_val[::h,s1:e1].mean(axis=1).flatten()[::h],ca_ic_exp_val[::h,s1:e1].mean(axis=1).flatten()[::h],\
            pH[::h,s1:e1].mean(dim=1).flatten().cpu().numpy(),  Ca[::h,s1:e1].mean(dim=1).flatten().cpu().numpy(),\
                par_vec[::h,s:e,0].mean(dim=1).flatten().cpu().numpy()[::h], par_vec[::h,s:e,1].mean(dim=1).flatten().cpu().numpy()[::h],\
                    U_r_ref[::h,s:e].mean(dim=1).flatten().cpu().numpy()[::h])
    
    #ph_shift_const = batch[:,:,7].cpu().numpy()
    plt.figure(figsize=(18, 8),facecolor='white')
    #plt.rcParams.update({'axes.titlesize': 20, 'axes.labelsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 20})
    plt.subplot(221);
    plt.plot(time[::h,s1:e1].mean(axis=1).flatten()[::5*h]/60, ph_exp_val[::h,s1:e1].mean(axis=1).flatten()[::5*h],'-o',color='magenta',linewidth=2,mew=2,alpha=.5,mfc='none',label=f'$\\bar H$')
    plt.plot(time[::h,s1:e1].mean(axis=1).flatten()/60, pH[::h,s1:e1].mean(dim=1).flatten().cpu().numpy(),'-',color='tab:cyan',lw=3,label=f'$\hat H^{sup}$')	
    plt.xlabel('Time $t~(\\rm min)$'); plt.ylabel("H"); plt.title("Acidity index as a function of time"); 
    plt.grid(True)
    plt.legend(loc='lower right')

    plt.subplot(223);
    plt.plot(time[::h,s1:e1].mean(axis=1).flatten()[::5*h]/60, ca_ic_exp_val[::h,s:e].mean(axis=1).flatten()[::5*h],'-o',color='magenta',linewidth=2,mew=2,alpha=.5,mfc='none',label='$\\bar Q$')
    plt.plot(time[::h,s1:e1].mean(axis=1).flatten()/60, Ca[::h,s:e].mean(dim=1).flatten().cpu().numpy(),'-',color='tab:cyan',lw=3,label=f'$\hat Q^{sup}$')
    plt.plot(time[::h,s1:e1].mean(axis=1).flatten()/60, C[::h,s:e].mean(dim=1).flatten().cpu().numpy(),'-',color='tab:blue',lw=3,label=f'$\hat C^{sup}$')
    plt.xlabel('Time $t~(\\rm min)$'); plt.ylabel(f'$\\bf Q, C$'); plt.title("[$Ca^{+2}$] and [${CaCO_3}$] as functions of time"); 
    plt.grid(True)
    plt.legend()

    plt.subplot(222);
    ur = 10 * par_vec[::h,s:e,0]*par_vec[::h,s:e,1]
    plt.plot(time[::h,s:e].mean(axis=1).flatten()[::h]/60, .1*par_vec[::h,s:e,0].mean(dim=1).flatten().cpu().numpy()[::h],'b-',lw=2, label=f'$0.1 U^{{{sup}}}_{{\\rm H}}$')
    plt.plot(time[::h,s:e].mean(axis=1).flatten()[::h]/60, (ur).mean(dim=1).flatten().cpu().numpy()[::h],'g-',lw=3,label=f'$10 U^{sup}_r$')
    plt.plot(time[::h,s:e].mean(axis=1).flatten()[::h]/60, 0*par_vec[::h,s:e,1].mean(dim=1).flatten().cpu().numpy()[::h],'k--',alpha=.5,lw=5,); 
    plt.ylim([-.005, min(.01,10*ur.flatten().cpu().numpy().max())])
    
    plt.xlabel('Time $t~(\\rm min)$'); plt.ylabel(f"$\\bf U_H, U_r$", labelpad=-5); plt.title("Rate modulation function")
    plt.grid(True)
    plt.legend()

    
    plt.subplot(224); 
    #print(F[0,:,:].cpu().numpy().T)
    #plt.imshow(F[0,:,:].cpu().numpy().T, aspect='auto', cmap='viridis', origin='lower')
    b = 100
    plt.plot(np.arange(64)*.1, F[b,2::8,:].cpu().numpy().T,'c-',lw=4)
    plt.plot(np.arange(64)*.1, F[b,1,:].cpu().numpy(),'c-',lw=4, label=f'$\hat F^{sup}(t,x)$') #, t \in \{8,16,24,32,40,48\}
    plt.plot(np.arange(64)*.1, F[b,0,:].cpu().numpy(),'b--',lw=2, label='$F(0,x)$')
    plt.plot(np.arange(64)*.1, F[b,-1,:].cpu().numpy(),'r--',lw=2, label=f'$\hat F^{sup}(T,x)$') #, t \in \{8,16,24,32,40,48\}
    plt.legend(ncol=2,loc='upper right')
    plt.xlabel('Size $x~(\mu m)$'); plt.ylabel("${\\bf F}$", labelpad=5); plt.title("PSD")
    #print(f">>>>>>>>>>>>>>>>>>>: {F.shape}")
    plt.subplots_adjust(hspace=0.45,wspace=.2)
    #plt.pause(.1)
    
    # Inset plot of G_arr as a function of time for a selected batch and size index
    # Place inset at a specific x, y position (fraction of parent axes: 0.6, 0.1)
    ax_inset = inset_axes(
        plt.gca(),
        width="35%",
        height="35%",
        bbox_to_anchor=(0.58, 0.1, 1, 1),  # (x0, y0, width, height) in axes fraction
        bbox_transform=plt.gca().transAxes,
        loc='lower left',
        borderpad=1
    )
    ax_inset.plot(time[b,:]/60, exp_smth(G_arr[b,:].flatten().cpu().numpy()[::h]), 'y-', lw=2, label=f'$\hat a^{sup}$')
    ax_inset.set_xlabel('Time $t~(\\rm min)$', fontsize=12, labelpad=-40)  # 1cm ≈ 40 points
    ax_inset.set_ylabel('$a$', fontsize=12)
    ax_inset.tick_params(axis='both', which='major', labelsize=12)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.set_title('Growth rate $a$', fontsize=12)
    ax_inset.legend(loc='best', fontsize=12, frameon=True)
    
    return res

def plot_train_results(res, h=5):
    batch, par_vec, F, C, Ca, Csolid, CO3, pH, simpH, simH, H, U_ph, U_r, U_r_ref, loss = res
    time = batch[:,:,0].cpu().numpy()
    ph_exp_val = batch[:,:,1].cpu().numpy()
    ca_ic_exp_val = batch[:,:,3].cpu().numpy()
    #ph_shift_const = batch[:,:,7].cpu().numpy()
    #print(time.shape, ph_exp_val.shape, C.shape, Ca.shape, Csolid.shape, pH.shape, simpH.shape, simH.shape, H.shape)
    #print(par_vec[::h,:,])
    plt.figure(figsize=(18, 8))
    plt.subplot(131);
    plt.plot(time[::h,:].T/60, ph_exp_val[::h,:].T,'-+')
    plt.plot(time[::h,:].T/60, pH[::h,:].T.cpu().numpy())

    plt.subplot(132);
    plt.plot(time[::h,:].T/60, ca_ic_exp_val[::h,:].T,'-+')
    plt.plot(time[::h,:].T/60, Ca[::h].T.cpu().numpy())
    # plt.plot(time[::h,:].T/60, C[::h].T.cpu().numpy(),'c-')
    # plt.plot(time[::h,:].T/60, Csolid[::h].T.cpu().numpy(),'y-')
    
    plt.subplot(133);
    plt.plot(time[::h,:].T/60, par_vec[::h,:,0].T.cpu().numpy(),'b-')
    plt.plot(time[::h,:].T/60, par_vec[::h,:,1].T.cpu().numpy(),'r-')


# Training loop
def run_train(args):
    # Training and validation loop
    seq_len = args.seq_len
    device = args.device
    GRU = args.model_type == "gru"
    LOAD = args.load_model
    num_epochs = args.num_epochs
    model_path = args.model_path
    
    print_model_stats(GRU)
    
    if GRU:
        model = GRUNetwork().to(device)
    else:
        model = ANN(seq_len=seq_len).to(device)
        
    if LOAD:
        model.load_state_dict(torch.load(model_path, map_location=args.device))

    optimizer = optim.Adamax(model.parameters(), lr=0.001)
    #optimizer = optim.AdamW(model.parameters(), lr=0.001)

    dataframes = [ser_ndf1, ser_ndf2, ser_ndf3, ser_ndf4]  # List of dataframes
    # Split the dataframes into training and validation datasets
    train_dataframes = [ser_ndf1, ser_ndf2, ser_ndf4]
    val_dataframes = [ser_ndf3]

    # Create datasets and dataloaders for training and validation
    train_dataset = MultiDataFrameDataset(train_dataframes, sequence_length=seq_len)
    train_dataset2 = MultiDataFrameDataset(train_dataframes, sequence_length=2*seq_len)
    train_dataset3 = MultiDataFrameDataset(train_dataframes, sequence_length=4*seq_len)
    train_dataset4 = MultiDataFrameDataset(train_dataframes, sequence_length=8*seq_len)
    train_dataset5 = MultiDataFrameDataset(train_dataframes, sequence_length=16*seq_len)
    val_dataset = MultiDataFrameDataset(val_dataframes, sequence_length=seq_len)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=250, shuffle=True)
    train_dataloader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=250, shuffle=True)
    train_dataloader3 = torch.utils.data.DataLoader(train_dataset3, batch_size=250, shuffle=True)
    train_dataloader4 = torch.utils.data.DataLoader(train_dataset4, batch_size=250, shuffle=True)
    train_dataloader5 = torch.utils.data.DataLoader(train_dataset5, batch_size=250, shuffle=True)
    train_dataloaders = [train_dataloader, train_dataloader2, train_dataloader3, train_dataloader4, train_dataloader5]
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=250, shuffle=False)

    best_val_loss = float("inf")
    model_typ = "gru" if GRU else "ann"
    h = 5
    train_arr = []
    val_arr   = []
    save_freq = []
    for epoch in range(num_epochs):
        tid = 0 #np.random.randint(0,2) if GRU else 0
        train_dataloader = train_dataloaders[tid]
        train_res = train_model(model, train_dataloader, optimizer, args)
        train_loss = train_res[-1]
        if epoch % 5 == 0:
            val_res = validate_model(model, val_dataloader, args)
            val_loss = val_res[-1]
            display.clear_output(wait=True)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_res[-1]:.4f}")
            #plot_train_results(val_res)
            plt.pause(.1)

        # Save the best model
        if val_loss < best_val_loss:
            save_freq.append(epoch)
            best_val_loss = val_loss
            print("Saved Best Model!")
            torch.save(model.state_dict(), f"../results/precip_design/weights/{model_typ}_ph_v3_model_fit_best.pth")
        else:
            save_freq.append(-1)

        train_arr.append(train_res)
        val_arr.append(val_res)

    # Save training and validation results as pickle files
    res_dict = {
        'train': train_arr,
        'val': val_arr,
        'save_freq': save_freq
    }
    torch.save(res_dict, f"../results/precip_design/data/dnn_{model_typ}_ph_v3_model_fit_e{num_epochs}_results.pt")
        

    model_typ = "gru" if GRU else "ann"
    torch.save(model.state_dict(), f"../results/precip_design/weights/{model_typ}_ph_v3_model_fit_e{num_epochs}.pth")

# Testing loop
def run_test(args):

    seq_len = args.seq_len
    device = args.device
    GRU = args.model_type == "gru"
    model_path = args.model_path
    
    
    print_model_stats(GRU)

    if GRU:
        model = GRUNetwork().to(device)
    else:
        model = ANN(seq_len=seq_len).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=args.device))
            
    h=1
    df1_dataset = MultiDataFrameDataset([ser_ndf1], sequence_length=seq_len); bs1 = len(ser_ndf1) - seq_len
    df2_dataset = MultiDataFrameDataset([ser_ndf2], sequence_length=seq_len); bs2 = len(ser_ndf2) - seq_len
    df3_dataset = MultiDataFrameDataset([ser_ndf3], sequence_length=seq_len); bs3 = len(ser_ndf3) - seq_len
    df4_dataset = MultiDataFrameDataset([ser_ndf4], sequence_length=seq_len); bs4 = len(ser_ndf4) - seq_len
    df1_dataloader = torch.utils.data.DataLoader(df1_dataset, batch_size=bs1, shuffle=False)
    df2_dataloader = torch.utils.data.DataLoader(df2_dataset, batch_size=bs2, shuffle=False)
    df3_dataloader = torch.utils.data.DataLoader(df3_dataset, batch_size=bs3, shuffle=False)
    df4_dataloader = torch.utils.data.DataLoader(df4_dataset, batch_size=bs4, shuffle=False)
    print(len(ser_ndf1), len(ser_ndf2), len(ser_ndf3), len(ser_ndf4)) #1838 791 4422 3723
    print(len(df1_dataloader), len(df2_dataloader), len(df3_dataloader), len(df4_dataloader))
    model_typ = "gru" if GRU else "ann"
    model_file = args.model_file
    num_epochs = args.num_epochs

    # with open(f"../results/precip_design/data/dnn_{model_typ}_ph_v3_model_fit_e{num_epochs}_results.pkl", 'rb') as f:
    #     res_dict = pickle.load(f)

    res_dict = torch.load(f"../results/precip_design/data/dnn_{model_typ}_ph_v3_model_fit_e{num_epochs}_results.pt", map_location=args.device)
    train_arr = res_dict['train']
    val_arr = res_dict['val']
    save_freq = res_dict['save_freq']
    print(len(train_arr), len(val_arr), len(save_freq))
    # convert to tensor from a list of tuples
    train_loss = torch.tensor([train_arr[k][-1] for k in range(len(train_arr))])
    val_loss = torch.tensor([val_arr[k][-1] for k in range(len(val_arr))])  
    losses = (train_loss, val_loss)
    
    # Plot training and validation loss
    res1 = validate_model(model, df1_dataloader, args);
    _ = plot_states(res1, GRU)
    plt.savefig(f'../results/precip_design/figures/{model_typ}_{model_file}_states_exp1.png', bbox_inches='tight', pad_inches=0.1)
    pres1 = plot_test_results(res1, GRU, losses)
    plt.savefig(f'../results/precip_design/figures/{model_typ}_{model_file}_exp1.png', bbox_inches='tight', pad_inches=0.1)
    
    res2 = validate_model(model, df2_dataloader, args);
    _ = plot_states(res2, GRU)
    plt.savefig(f'../results/precip_design/figures/{model_typ}_{model_file}_states_exp2.png', bbox_inches='tight', pad_inches=0.1)
    pres2 = plot_test_results(res2, GRU, losses)
    plt.savefig(f'../results/precip_design/figures/{model_typ}_{model_file}_exp2.png', bbox_inches='tight', pad_inches=0.1)
    
    res3 = validate_model(model, df3_dataloader, args);
    _ = plot_states(res3, GRU)
    plt.savefig(f'../results/precip_design/figures/{model_typ}_{model_file}_states_exp3.png', bbox_inches='tight', pad_inches=0.1)
    pres3 = plot_test_results(res3, GRU, losses)
    plt.savefig(f'../results/precip_design/figures/{model_typ}_{model_file}_exp3.png', bbox_inches='tight', pad_inches=0.1)
    
    res4 = validate_model(model, df4_dataloader, args);
    _ = plot_states(res4, GRU)
    plt.savefig(f'../results/precip_design/figures/{model_typ}_{model_file}_states_exp4.png', bbox_inches='tight', pad_inches=0.1)
    pres4 = plot_test_results(res4, GRU, losses)
    plt.savefig(f'../results/precip_design/figures/{model_typ}_{model_file}_exp4.png', bbox_inches='tight', pad_inches=0.1)

    h=1
    def divide_into_chunks(df, seq_len):
        num_batches = len(df) // seq_len
        return np.stack([df.iloc[i * seq_len:(i + 1) * seq_len].to_numpy() for i in range(num_batches)],axis=0)

    sl = seq_len if GRU else seq_len
    df1_batches = divide_into_chunks(ser_ndf1, seq_len=sl)
    df2_batches = divide_into_chunks(ser_ndf2, seq_len=sl)
    df3_batches = divide_into_chunks(ser_ndf3, seq_len=sl)
    df4_batches = divide_into_chunks(ser_ndf4, seq_len=sl)
    print(df1_batches.shape, df2_batches.shape, df3_batches.shape, df4_batches.shape)

    setup_figure()

    # Run inference test for disjoint chunks of sequences stacked as batches
    args.exp_typ = "exp1"
    _, ipres1 = test_chunks_in_parallel(model, df1_batches, losses, args);
    args.exp_typ = "exp2"
    _, ipres2 = test_chunks_in_parallel(model, df2_batches, losses, args);
    args.exp_typ = "exp3"
    _, ipres3 = test_chunks_in_parallel(model, df3_batches, losses, args);
    args.exp_typ = "exp4"
    _, ipres4 = test_chunks_in_parallel(model, df4_batches, losses, args);

    
    pouts = [pres1, pres2, pres3, pres4]
    ipouts = [ipres1, ipres2, ipres3, ipres4]
    return pouts, ipouts
    

def compare_itr_vs_direct_inf(pouts, ipouts, args):
    setup_figure()
    fig, axs = plt.subplots(2,2,figsize=(18, 8),gridspec_kw=dict(hspace=.45, wspace=.215),facecolor='white'); axs = axs.flatten()
    plt.rcParams.update({'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
    model_typ = args.model_type
    model_file = args.model_file
    lp = 6 if model_typ=='ann' else 4
    for i in range(4):
        time, ph_exp_val, ca_ic_exp_val, pred_pH, pred_ca, pred_u_ph, pred_u_r, pred_u_r_ref, train_loss, val_loss = pouts[i]
        itime, iph_exp_val, ica_ic_exp_val, ipred_pH, ipred_ca, ipred_u_ph, ipred_u_r, ipred_u_r_ref, itrain_loss, ival_loss = ipouts[i]
        #pred_u_r = 10*pred_u_r
        umax = (pred_u_ph*pred_u_r).max(); umin = (pred_u_ph*pred_u_r).min()
        iumax = (ipred_u_ph*ipred_u_r).max(); iumin = (ipred_u_ph*ipred_u_r).min()
        axs[i].plot(itime/60, ipred_u_ph*ipred_u_r,'--',color='tab:red',lw=3,alpha=.75,label='batched')	
        axs[i].plot(time/60, pred_u_ph*pred_u_r,'-',color='tab:green',linewidth=3,mew=2,mfc='none',label='sequential')
        axs[i].set_xlabel('Time $t~(\\rm min)$'); #axs[i].set_ylabel("${\\bf U_r}$"); 
        axs[i].set_title(f"Rate modulation function for Exp-{i+1} "); 
        axs[i].set_ylim([max(min(umin,iumin),-0.001), min(max(umax,iumax),.001)])
        if i == 1 and model_typ == 'gru':
            lp = -4
        if i == 3:
            lp = 10
        axs[i].set_ylabel('${\\bf U_r}$',labelpad=-lp)
        axs[i].grid(True)
        if i == 0:
            axs[i].legend(ncol=2, loc='upper center', bbox_to_anchor=(1.1, -0.1), framealpha=1, edgecolor="gray", fancybox=True, frameon=True, facecolor='lavender')

    plt.suptitle(f"Comparison of sequential and batched inference for {model_typ} model")
    # save figure
    plt.savefig(f'../results/precip_design/figures/{model_typ}_{model_file}_seq_vs_bat_inf.png', bbox_inches='tight', pad_inches=0.1)
    
    
def compare_model_inf(ann_outs, gru_outs, suf = 'bat'):
    setup_figure()
    fig, axs = plt.subplots(2,2,figsize=(18, 8),gridspec_kw=dict(hspace=.45, wspace=.2),facecolor='white'); axs = axs.flatten()
    plt.rcParams.update({'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
    alp = .75 #if SEQ else 0.75
    for i in range(4):
        time, ph_exp_val, ca_ic_exp_val, pred_pH, pred_ca, pred_u_ph, pred_u_r, pred_u_r_ref, train_loss, val_loss = ann_outs[i]
        gtime, gph_exp_val, gca_ic_exp_val, gpred_pH, gpred_ca, gpred_u_ph, gpred_u_r, gpred_u_r_ref, gtrain_loss, gval_loss = gru_outs[i]
        #pred_u_r = 10*pred_u_r
        #gpred_u_r = 10*gpred_u_r
        umax = (pred_u_ph*pred_u_r).max(); umin = (pred_u_ph*pred_u_r).min()
        gumax = (gpred_u_ph*gpred_u_r).max(); gumin = (gpred_u_ph*gpred_u_r).min()
        axs[i].plot(time/60, pred_u_ph*pred_u_r,'--',color='tab:red',lw=3,alpha=alp,label='ANN'); 
        axs[i].plot(gtime/60, gpred_u_ph*gpred_u_r,'-',color='tab:green',linewidth=3,mew=2,mfc='none',label='GRUN'); 
        axs[i].set_ylabel("${\\bf U_r}$", labelpad=-10)
        axs[i].set_xlabel('Time $t~(\\rm min)$'); #axs[i].set_ylabel("${\\bf U_r}$"); 
        axs[i].set_title(f"Rate modulation function for Exp-{i+1} "); 
        axs[i].set_ylim([max(min(umin,gumin),-0.001), min(max(umax,gumax),.001)])
        axs[i].grid(True)
        if i == 0:
            axs[i].legend(ncol=2, loc='upper center', bbox_to_anchor=(1.1, -0.1), framealpha=1, edgecolor="gray", fancybox=True, frameon=True, facecolor='lavender')
    plt.suptitle(f"Comparison of sequential inference for ANN and GRU models")
    # save figure
    #suf = "seq" if SEQ else "bat"
    plt.savefig(f'../results/precip_design/figures/ann_vs_gru_{suf}.png', bbox_inches='tight', pad_inches=0.1)

def smoothen_compare_with_fbsde(ann_outs, gru_outs):
    setup_figure()
    fig, axs = plt.subplots(2,2,figsize=(18, 8),gridspec_kw=dict(hspace=.45, wspace=.2),facecolor='white'); axs = axs.flatten()
    plt.rcParams.update({'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
    win_size = 5
    ker = np.ones(win_size)/win_size
    for i in range(4):
        with open(f'../results/precip_design/data/fbsde_exp{i+1}.pkl', 'rb') as f:
            pdata = pickle.load(f)
        ph_exp_t, u_opt_np, u_r_np  = pdata['ph_exp_t'], pdata['U_opt'][0].detach().cpu().numpy(), pdata['U_r'][0].detach().cpu().numpy()
        time, ph_exp_val, ca_ic_exp_val, pred_pH, pred_ca, pred_u_ph, pred_u_r, pred_u_r_ref, train_loss, val_loss = ann_outs[i]
        gtime, gph_exp_val, gca_ic_exp_val, gpred_pH, gpred_ca, gpred_u_ph, gpred_u_r, gpred_u_r_ref, gtrain_loss, gval_loss = gru_outs[i]
        #pred_u_r = 10*pred_u_r
        #gpred_u_r = 10*gpred_u_r
        #u_opt_np = 10*u_opt_np
        umax = (pred_u_ph*pred_u_r).max(); umin = (pred_u_ph*pred_u_r).min()
        gumax = (gpred_u_ph*gpred_u_r).max(); gumin = (gpred_u_ph*gpred_u_r).min()
        fumax = min((u_opt_np).max(),gumax); fumin = max((u_opt_np).min(),gumin)
        ph_exp_t = ph_exp_t[:len(time)]; u_opt_np = u_opt_np[:len(time)]
        #axs[i].plot(ph_exp_t/60, np.convolve(pred_u_r_ref, ker, mode='same'),'-',color='black',lw=2,alpha=1,label='Manual')	
        axs[i].plot(ph_exp_t/60, exp_smth(pred_u_r_ref),'-',color='black',lw=2,alpha=.75,label='Manual')	
        axs[i].plot(ph_exp_t/60, exp_smth(u_opt_np),'-.',color='tab:blue',lw=4,alpha=.75,label='FBSSM')	
        axs[i].plot(time/60, exp_smth(pred_u_ph*pred_u_r),'--',color='tab:red',lw=3,alpha=.75,label='ANN')	
        axs[i].plot(gtime/60, exp_smth(gpred_u_ph*gpred_u_r),'-',color='tab:green',linewidth=3,mew=2,mfc='none',label='GRUN')
        axs[i].set_xlabel('Time $t~(\\rm min)$'); #axs[i].set_ylabel("${\\bf U_r}$"); 
        suf = "" #if i < 2 else ""
        axs[i].set_title(f"Rate modulation function for Exp-{i+1}{suf}"); 
        #axs[i].set_ylim([max(min(umin,fumin),-0.01), min(max(umax,fumax),.01)])
        axs[i].set_ylim([-0.0025, 0.0075])
        axs[i].grid(True)
        if i == 0:
            axs[i].legend(ncol=2, loc='upper center', bbox_to_anchor=(1.1, -0.08), framealpha=1, edgecolor="gray", fancybox=True, frameon=True, facecolor='lavender')
        print(len(ph_exp_t), len(time), len(gtime),flush=True)
        
    plt.suptitle(f"Comparison of ANN, FBSSM and GRU methods")
    #plt.subplots_adjust(top=0.85)
    # save figure
    plt.savefig(f'../results/precip_design/figures/ann_vs_gru_fbsde_smooth.png', bbox_inches='tight', pad_inches=0.1)


def compare_with_fbsde(ann_outs, gru_outs):
    setup_figure()
    fig, axs = plt.subplots(2,2,figsize=(18, 8),gridspec_kw=dict(hspace=.45, wspace=.215),facecolor='white'); axs = axs.flatten()
    plt.rcParams.update({'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
    for i in range(4):
        with open(f'../results/precip_design/data/fbsde_exp{i+1}.pkl', 'rb') as f:
            pdata = pickle.load(f)
        ph_exp_t, u_opt_np, u_r_np  = pdata['ph_exp_t'], pdata['U_opt'][0].detach().cpu().numpy(), pdata['U_r'][0].detach().cpu().numpy()
        time, ph_exp_val, ca_ic_exp_val, pred_pH, pred_ca, pred_u_ph, pred_u_r, pred_u_r_ref, train_loss, val_loss = ann_outs[i]
        gtime, gph_exp_val, gca_ic_exp_val, gpred_pH, gpred_ca, gpred_u_ph, gpred_u_r, gpred_u_r_ref, gtrain_loss, gval_loss = gru_outs[i]
        #pred_u_r    = 10*pred_u_r
        #gpred_u_r   = 10*gpred_u_r
        #u_opt_np    = 10*u_opt_np
        umax = (pred_u_ph*pred_u_r).max(); umin = (pred_u_ph*pred_u_r).min()
        gumax = (gpred_u_ph*gpred_u_r).max(); gumin = (gpred_u_ph*gpred_u_r).min()
        fumax = min((u_opt_np).max(),gumax); fumin = max((u_opt_np).min(),gumin)
        ph_exp_t = ph_exp_t[:len(time)]; u_opt_np = u_opt_np[:len(time)]
        axs[i].plot(ph_exp_t/60, u_opt_np,'-.',color='tab:blue',lw=4,alpha=.75,label='FBSSM')	
        axs[i].plot(time/60, pred_u_r_ref,'-',color='black',lw=2,alpha=.75,label='Manual')	
        axs[i].plot(time/60, pred_u_ph*pred_u_r,'--',color='tab:red',lw=3,alpha=.75,label='ANN')	
        axs[i].plot(gtime/60, gpred_u_ph*gpred_u_r,'-',color='tab:green',linewidth=3,mew=2,mfc='none',label='GRUN')
        axs[i].set_xlabel('Time $t~(\\rm min)$'); #axs[i].set_ylabel("${\\bf U_r}$"); 
        axs[i].set_title(f"Rate modulation function for Exp-{i+1}"); 
        #axs[i].set_ylim([max(min(umin,fumin),-0.01), min(max(umax,fumax),.01)])
        axs[i].set_ylim([-0.0025, 0.0075])
        axs[i].grid(True)
        if i == 0:
            axs[i].legend(ncol=2, loc='upper center', bbox_to_anchor=(1.1, -0.08), framealpha=1, edgecolor="gray", fancybox=True, frameon=True, facecolor='lavender')

    plt.suptitle(f"Comparison of ANN, FBSSM and GRU methods")
    #plt.subplots_adjust(top=0.85)
    # save figure
    plt.savefig(f'../results/precip_design/figures/ann_vs_gru_fbsde.png', bbox_inches='tight', pad_inches=0.1)

    smoothen_compare_with_fbsde(ann_outs, gru_outs)

def smoothen_cmp_with_fbsde_per_exp(ann_outs, gru_outs, exp):
    setup_figure()
    plt.rcParams.update({'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
    win_size = 5
    ker = np.ones(win_size)/win_size
    #i = exp-1
    for i in range(4):
        fig, axs = plt.subplots(2,2,figsize=(18, 8),gridspec_kw=dict(hspace=.45, wspace=.218),facecolor='white'); axs = axs.flatten()
        with open(f'../results/precip_design/data/fbsde_exp{i+1}.pkl', 'rb') as f:
            pdata = pickle.load(f)
        ph_exp_t, u_opt_np, u_r_np  = pdata['ph_exp_t'], pdata['U_opt'][0].detach().cpu().numpy(), pdata['U_r'][0].detach().cpu().numpy()
        time, ph_exp_val, ca_ic_exp_val, pred_pH, pred_ca, pred_u_ph, pred_u_r, pred_u_r_ref, train_loss, val_loss = ann_outs[i]
        gtime, gph_exp_val, gca_ic_exp_val, gpred_pH, gpred_ca, gpred_u_ph, gpred_u_r, gpred_u_r_ref, gtrain_loss, gval_loss = gru_outs[i]
        #pred_u_r = 10*pred_u_r
        #gpred_u_r = 10*gpred_u_r
        #u_opt_np = 10*u_opt_np
        umax = (pred_u_ph*pred_u_r).max(); umin = (pred_u_ph*pred_u_r).min()
        gumax = (gpred_u_ph*gpred_u_r).max(); gumin = (gpred_u_ph*gpred_u_r).min()
        fumax = min((u_opt_np).max(),gumax); fumin = max((u_opt_np).min(),gumin)
        ph_exp_t = ph_exp_t[:len(time)]; u_opt_np = u_opt_np[:len(time)]
        #axs[i].plot(ph_exp_t/60, np.convolve(pred_u_r_ref, ker, mode='same'),'-',color='black',lw=2,alpha=1,label='Manual')	
        axs[0].plot(ph_exp_t/60, exp_smth(pred_u_r_ref),'-',color='black',lw=2,alpha=.75,label='Manual'); axs[0].set_ylabel("${\\bf \hat U^m_r}$")	
        axs[1].plot(ph_exp_t/60, exp_smth(u_opt_np),'-.',color='tab:blue',lw=4,alpha=.75,label='FBSSM'); axs[1].set_ylabel("${\\bf \hat U^f_r}$", labelpad=-8+min(8,8*i))	
        axs[2].plot(time/60, exp_smth(pred_u_ph*pred_u_r),'--',color='tab:red',lw=3,alpha=.75,label='ANN'); axs[2].set_ylabel("${\\bf \hat U^a_r}$")	
        axs[3].plot(gtime/60, exp_smth(gpred_u_ph*gpred_u_r),'-',color='tab:green',linewidth=3,mew=2,mfc='none',label='GRUN'); axs[3].set_ylabel("${\\bf \hat U^g_r}$", labelpad=-8+min(8,8*i))
        labels_arr = ['Manual', 'FBSSM', 'ANN', 'GRUN']
        for j in range(4):
            axs[j].set_xlabel('Time $t~(\\rm min)$'); 
            axs[j].set_title(f"$\\bf U_r$ obtained via {labels_arr[j]} method"); 
            axs[j].grid(True)
            axs[j].legend(loc='best', framealpha=1, edgecolor="gray", fancybox=True, frameon=True)
        print(len(ph_exp_t), len(time), len(gtime),flush=True)
            
        plt.suptitle(f"Comparison of FBSSM, ANN and GRU methods for Exp-{i+1}")
        # save figure
        plt.savefig(f'../results/precip_design/figures/ann_vs_gru_fbsde_smooth_exp{i+1}.png', bbox_inches='tight', pad_inches=0.1)
    
    
def compare_with_in_ph(ann_outs, gru_outs):
    setup_figure()
    plt.rcParams.update({'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
    for i in range(4):
        with open(f'../results/precip_design/data/fbsde_exp{i+1}.pkl', 'rb') as f:
            pdata = pickle.load(f)

        ph_exp_t, pH, dph, u_opt_np, u_r_np  = pdata['ph_exp_t'], pdata['pH'].detach().cpu().numpy(), pdata['dph'].detach().cpu().numpy(), pdata['U_opt'][0].detach().cpu().numpy(), pdata['U_r'][0].detach().cpu().numpy()
        time, ph_exp_val, ca_ic_exp_val, pred_pH, pred_ca, pred_u_ph, pred_u_r, pred_u_r_ref, train_loss, val_loss = ann_outs[i]
        gtime, gph_exp_val, gca_ic_exp_val, gpred_pH, gpred_ca, gpred_u_ph, gpred_u_r, gpred_u_r_ref, gtrain_loss, gval_loss = gru_outs[i]
        #pred_u_r = 10*pred_u_r;
        #gpred_u_r = 10*gpred_u_r;
        #u_opt_np = 10*u_opt_np;
        umax = (pred_u_ph*pred_u_r).max(); umin = (pred_u_ph*pred_u_r).min()
        gumax = (gpred_u_ph*gpred_u_r).max(); gumin = (gpred_u_ph*gpred_u_r).min()
        fumax = min((u_opt_np).max(),gumax); fumin = max((u_opt_np).min(),gumin)
        ph_exp_t = ph_exp_t[:len(time)]; u_opt_np = u_opt_np[:len(time)]
        plt.figure(figsize=(18, 8), facecolor='white')
        plt.subplot(221);plt.plot(ph_exp_t[::1]/60, exp_smth(.025*dph[:len(ph_exp_t):1]),'-',color='magenta',lw=4,alpha=.75,label='Input pH ($\\bar U_H$)'); plt.grid(True)
        plt.subplot(221);plt.plot(time[::1]/60, exp_smth(.25*pred_u_r_ref[::1]),'-',color='black',lw=4,alpha=.75,label='Manual'); plt.xlabel('Time $t~(\\rm min)$');
        plt.legend(loc='lower center', ncol=2, framealpha=1, edgecolor="gray", fancybox=True, frameon=True)
        plt.subplot(222);plt.plot(ph_exp_t[::1]/60, exp_smth(.025*dph[:len(ph_exp_t):1]),'-',color='magenta',lw=4,alpha=.75,label='Input pH ($\\bar U_H$)'); plt.grid(True)
        plt.subplot(222);plt.plot(ph_exp_t/60, exp_smth(u_opt_np),'-.',color='tab:blue',lw=4,alpha=.75,label='FBSSM'); plt.xlabel('Time $t~(\\rm min)$');
        plt.legend(loc='lower center', ncol=2, framealpha=1, edgecolor="gray", fancybox=True, frameon=True)
        plt.subplot(223);plt.plot(ph_exp_t[::1]/60, exp_smth(.025*dph[:len(ph_exp_t):1]),'-',color='magenta',lw=4,alpha=.75,label='Input pH ($\\bar U_H$)'); plt.grid(True)
        plt.subplot(223);plt.plot(time/60, exp_smth(pred_u_ph*pred_u_r),'--',color='tab:red',lw=4,alpha=.75,label='ANN'); plt.xlabel('Time $t~(\\rm min)$');
        plt.legend(loc='lower center', ncol=2, framealpha=1, edgecolor="gray", fancybox=True, frameon=True)
        plt.subplot(224);plt.plot(ph_exp_t[::1]/60, exp_smth(.025*dph[:len(ph_exp_t):1]),'-',color='magenta',lw=4,alpha=.75,label='Input pH ($\\bar U_H$)'); plt.grid(True)
        plt.subplot(224);plt.plot(gtime/60, exp_smth(gpred_u_ph*gpred_u_r),'-',color='tab:green',linewidth=3,mew=2,mfc='none',label='GRUN'); plt.xlabel('Time $t~(\\rm min)$');
        plt.legend(loc='lower center', ncol=2, framealpha=1, edgecolor="gray", fancybox=True, frameon=True)
        #plt.title(f"Rate modulation function for Exp-{i+1}"); 
        #plt.ylim([-0.005, 0.005])
        ilines = np.where(dph[:len(ph_exp_t)-5] > .24 )
        for l in ilines[0]:
            x_mark = (ph_exp_t/60)[l+5]
            #plt.axvline(x = x_mark, color='k',linestyle='--',alpha=.25)
        #plt.legend(ncol=1, loc='upper center', bbox_to_anchor=(1.11, 0.8), framealpha=1, edgecolor="gray", fancybox=True, frameon=True, facecolor='lavender')
        print(len(ph_exp_t), len(time), len(gtime), flush=True)
        plt.suptitle(f"Comparison of input pH, manual, FBSSM and GRU methods")
        #plt.subplots_adjust(top=0.85)
        # save figure
        plt.savefig(f'../results/precip_design/figures/in_ph_man_gru_fbsde_exp{i+1}.png', bbox_inches='tight', pad_inches=0.1)
    

# main function to run training or testing based on command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the precipitation model.")
    parser.add_argument('-m', "--mode", type=str, choices=["train", "test", "cmp"], required=True, help="Mode to run: 'train' or 'test'")
    parser.add_argument('-f', "--model_file", type=str, default="best", help="Path to the pre-trained model")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length for training/testing")
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs for training")
    parser.add_argument("--model_type", type=str, choices=["gru", "ann"], default="gru", help="Type of model to use: 'gru' or 'ann'")
    parser.add_argument("--load_model", action="store_true", help="Load a pre-trained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    args.model_path = f"../results/precip_design/weights/{args.model_type}_ph_v3_model_fit_{args.model_file}.pth"
    
    torch.manual_seed(args.seed)
    GRU = args.model_type == "gru"
    args.device = "cpu" if (torch.cuda.device_count() == 0) else ("cuda:0" if GRU else "cuda:1")
    print(f"{torch.cuda.device_count()} GPUs available, using {args.device}")

    if not os.path.exists('../results/precip_design/data'):
        print("Creating data directory...")
        os.makedirs('../results/precip_design/data', exist_ok=True)
    if not os.path.exists('../results/precip_design/figures'):
        print("Creating figures directory...")
        os.makedirs('../results/precip_design/figures', exist_ok=True)
    if not os.path.exists('../results/precip_design/weights'):
        print("Creating weights directory...")
        os.makedirs('../results/precip_design/weights', exist_ok=True)
        
    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        pouts, ipouts = run_test(args)
        compare_itr_vs_direct_inf(pouts, ipouts, args)
    elif args.mode == "cmp":
        args.model_type = "ann"; GRU = False
        args.model_path = f"../results/precip_design/weights/{args.model_type}_ph_v3_model_fit_{args.model_file}.pth"
        ann_pouts, ann_ipouts = run_test(args)
        compare_itr_vs_direct_inf(ann_pouts, ann_ipouts, args)
        
        args.model_type = "gru"; GRU = True
        args.seq_len = 64
        args.model_path = f"../results/precip_design/weights/{args.model_type}_ph_v3_model_fit_{args.model_file}.pth"
        gru_pouts, gru_ipouts = run_test(args)
        compare_itr_vs_direct_inf(gru_pouts, gru_ipouts, args)
        
        compare_model_inf(ann_pouts, gru_pouts, 'seq')
        
        compare_model_inf(ann_ipouts, gru_ipouts, 'bat')
        
        compare_with_fbsde(ann_pouts, gru_pouts)
        
        compare_with_in_ph(ann_pouts, gru_pouts)
        
        smoothen_cmp_with_fbsde_per_exp(ann_pouts, gru_pouts, exp=1)