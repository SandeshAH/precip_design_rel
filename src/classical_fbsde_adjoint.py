#########################################################################
# --------------------------------------------------------------------- #
# | Author      : Sandesh Athni Hiremath                                |
# |---------------------------------------------------------------------|
# | Description :                                                       |
# |   This script implements the classical Forward-Backward Stochastic  |
# |   Differential Equation (FBSDE) adjoint algorithm for optimal       |
# |   control of a pH-driven precipitation process. It uses PyTorch for |
# |   numerical computations and supports both optimization and         |
# |   visualization of results. The script loads experimental data,     |
# |   runs the FBSDE-based optimization, and provides plotting and      |
# |   result-saving utilities.                                          |
# |---------------------------------------------------------------------|
# | Usage       : python classical_fbsde_adjoint.py                     |
# |                --mode [opt|plot] --exp_typ N [--max_sim M] [--cont] |
# |   --mode opt   : Run FBSDE optimization for the selected experiment |
# |   --mode plot  : Plot results for the selected experiment           |
# |   --exp_typ N  : Select experiment number (1-4)                     |
# |   --max_sim M  : Set max optimization iterations (default: 100)     |
# |   --cont       : Continue from saved results                        |
# |   --seed S     : Set random seed (default: 42)                      |
# |   See --help for more options.                                      |
# --------------------------------------------------------------------- #
#########################################################################

import torch
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
import pickle
import seaborn as sns
from tqdm import tqdm
import warnings
import os
warnings.simplefilter('ignore')

import sys
sys.path.append('../')

from src.utils import *
from src.data_utils import *

ser_ndf1, ser_ndf2, ser_ndf3, ser_ndf4 = proc_data()
ser_ndf1.columns, ser_ndf1.shape

F, C, Ca, H, V = torch.randn(50,64), torch.randn(50), torch.randn(50), torch.randn(50), torch.randn(50)
U_r = torch.randn(50)
x_ax    = torch.tensor(0.1 * np.arange(64))
J_vec = N_fn(C,H)
G_vec = a_fn(C,H)
St      = St_fn(x_ax,F[-1])
St_star = St_star_fn(x_ax,F[-1])
St_vec  = (St, St_star)
X_vec = F, C, Ca, H, V
rhsF, drhsF_dF, drhsF_dC, drhsF_dCa, drhsF_dH = rhsF_fn(X_vec, J_vec)
rhsC, drhsC_dF, drhsC_dC, drhsC_dCa, drhsC_dH = rhsC_fn(X_vec, J_vec, G_vec, St_vec, U_r )
rhsCa, drhsCa_dF, drhsCa_dC, drhsCa_dCa, drhsCa_dH = rhsCa_fn(X_vec, U_r)
#print(rhsF.shape, drhsF_dF.shape, drhsC_dF.shape, drhsCa_dF.shape)
rhsC.shape, rhsCa.shape


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch1 = torch.tensor(ser_ndf1.to_numpy()).float().to(device)[None,:,:].repeat(5, 1, 1)
par_vec1  = torch.tensor([1, 1, 1e2, .5, 1, .02, .002, 0.0005, .001]).view(1,1,-1).expand(5,batch1.size(1),-1).to(device)
batch2 = torch.tensor(ser_ndf2.to_numpy()).float().to(device)[None,:,:].repeat(5, 1, 1)
par_vec2  = torch.tensor([1, 1, 1e2, 1, 1, .015, .0075, .002, .003]).view(1,1,-1).expand(5,batch2.size(1),-1).to(device)
batch3 = torch.tensor(ser_ndf3.to_numpy()).float().to(device)[None,:,:].repeat(5, 1, 1)
par_vec3  = torch.tensor([1, 1, 1e2, .1, 1, .019, .02, 0.0001, .001]).view(1,1,-1).expand(5,batch3.size(1),-1).to(device)
batch4 = torch.tensor(ser_ndf4.to_numpy()).float().to(device)[None,:,:].repeat(5, 1, 1)
par_vec4  = torch.tensor([ 1, 1, 1e2, 1, 1, .038, .018, 0.0004, .001]).view(1,1,-1).expand(5,batch4.size(1),-1).to(device)
#print(batch1.shape, batch2.shape, batch3.shape, batch4.shape)
#print(par_vec1.shape, par_vec2.shape, par_vec3.shape, par_vec4.shape)

setup_figure()

def plot_iter(pdata, SAVE=False):
    ph_exp_t = pdata['ph_exp_t']
    U_opt = pdata['U_opt']
    U_r = pdata['U_r']
    Ca = pdata['Ca'][0].detach().cpu().numpy()
    Y_Ca = pdata['Y_Ca']
    dph = pdata['dph']
    pH = pdata['pH'][0].detach().cpu().numpy()
    simpH = pdata['simpH'][0].detach().cpu().numpy()
    ca_ic_exp_val = pdata['ca_ic_exp_val'][0].detach().cpu().numpy()
    cost = pdata['cost']
    n = pdata['iter']
    exp_typ = None if 'exp_typ' not in pdata.keys() else pdata['exp_typ']

    u_r_np      = U_r[0].detach().cpu().numpy()
    u_opt_np    = U_opt[0].detach().cpu().numpy()

    #display.clear_output(True)
    fig,axs=plt.subplots(2,2,figsize=(18,8),gridspec_kw=dict(hspace=.45, wspace=.2),facecolor='white');axs = axs.flatten()

    axs[0].plot(ph_exp_t/60,pH,'-o',color='magenta',linewidth=2,mew=2,alpha=.5,mfc='none',label='pH (measured)')
    axs[0].plot(ph_exp_t/60,simpH,'-',color='tab:cyan',lw=3,label='pH (predicted)')
    axs[0].set_xlabel('Time $t~(\\rm min)$'); axs[0].set_ylabel("H"); axs[0].set_title("Acidity index as a function of time");
    axs[0].grid(True)
    axs[0].legend(loc='lower right');

    axs[2].plot(ph_exp_t/60,1*ca_ic_exp_val,'-o',color='magenta',linewidth=2,mew=2,alpha=.5,mfc='none',label='Ca (measured)')
    axs[2].plot(ph_exp_t/60,Ca,'-',color='tab:cyan',lw=3,label='Ca (predicted)')
    axs[2].set_xlabel('Time $t~(\\rm min)$'); axs[2].set_ylabel("[$Ca^{+2}$]"); axs[2].set_title("[$Ca^{+2}$] as a function of time");
    axs[2].legend()
    axs[2].grid(True)


    axs[3].plot(cost[:500],'b-o', lw=2, mew=4, ms=4, label='J(u)');
    if n > 0:
        axs[3].plot(n,cost[-1],'r>', markersize=10, label='J(u)$_n$');
        axs[3].text(0.5, 0.8, f'Iter: {n}, J(u)$_n$: {round(1*cost[-1],4)}', fontsize=20, ha='center', va='center', transform=axs[3].transAxes)
    axs[3].set_xlabel('Iterations'); axs[3].set_ylabel('J');
    axs[3].set_title(f'Objective function');
    axs[3].grid(True)
    axs[3].legend()
    
    #axs[3].plot(ph_exp_t/60,Y_Ca[0].detach().cpu().numpy(),'y',lw=2,label='$\\lambda_{\\rm Ca}$');
    axs[1].plot(ph_exp_t/60,u_opt_np,'g-',lw=3,label='$U_r$');
    axs[1].plot(ph_exp_t/60, 0*u_opt_np,'k--',alpha=.5,lw=5,);
    axs[1].set_title(f'Generated signal'); 
    axs[1].set_ylabel("${\\bf U_r}$");
    axs[1].set_xlabel('Time $t~(\\rm min)$');
    axs[1].grid(True)
    axs[1].legend()

    if exp_typ is not None:
        plt.suptitle(f'Experiment {int(exp_typ)}',fontsize=20,fontweight='bold')

    if SAVE:
        print(f"Saving plot in results/precip_design/figures/fbsde_exp{int(exp_typ)}.png")
        plt.savefig(f'../results/precip_design/figures/fbsde_exp{int(exp_typ)}.png', bbox_inches='tight', pad_inches=0.1)

def precip_fbsde_solve(batch, par_vec, max_sim=500, exp_typ = 1, U=None, cost_=None, SAVE=False, DEBUG=False):
    ##################
    ph_exp_val = batch[:,:,1]
    dph_exp_val = torch.zeros_like(batch[:,:,1],device=batch.device);
    dph_exp_val[:,1:] = ph_exp_val[:,1:] - ph_exp_val[:,:-1]
    dph = dph_exp_val.mean(axis=(0)).to(batch.device)[:5440]
    mg_ic_exp_val = batch[:,:,2]
    ca_ic_exp_val = batch[:,:,3]
    ca_exp_val = batch[:,:,4]
    exp_typ = batch[:,:,7].mean(axis=(0,1)).to(batch.device)
    exp_ca0 = batch[:,:,8].to(batch.device)
    exp_c0 = exp_ca0 - ca_ic_exp_val
    ph_shift_const = batch[:,0,9].to(batch.device)

    ############    
    dt      = torch.tensor(.01)
    dx      = torch.tensor(.1)
    Nx      = 64
    x_ax    = torch.tensor(dx * torch.arange(Nx)).to(dph.device)
    f0      = (6 / (2 * np.pi)) * torch.exp(-6e2 * (1.5 - x_ax)**2)
    alp     = .1
    
    ############
    Dxf, Dxb, Dxfab, Dxbab = getFBDiffOps1D(Nx, True)
    D_xx = npSps2Torch(get1DLapOp(Nx))  # lhs operator
    # Advection operator
    h = (dt / dx)  # Courant number (dt/self.dx)
    adOp = (0.5 * ((Dxbab - Dxb * h) - (Dxfab - Dxf * h)) * h).to(f0.device)
    
    #############
    p1,p2,p3,p4,p5,p6,p7,p8,p9 = par_vec[:,:,0],par_vec[:,:,1],par_vec[:,:,2],par_vec[:,:,3],par_vec[:,:,4],par_vec[:,:,5],par_vec[:,:,6],par_vec[:,:,7],par_vec[:,:,8]
    
    U_ph    = dph_exp_val
    U_r     =  p5*torch.clamp(p6*U_ph,min=-p8,max=p7)
    U_opt   = torch.zeros_like(dph_exp_val[:,:len(dph)]) if U is None else U
    #print(U_ph.shape, U_r.shape)
    
    #p1, p2, p3, p4 = 1, 1, 1e2, 10
    ###################
    sig_C      = .025;  dW1 = torch.randn_like(ph_exp_val)
    sig_Ca     = .0005; dW2 = torch.randn_like(ph_exp_val)
    sig_H      = .001;  dW3 = torch.randn_like(ph_exp_val) 
    
    ########
    cost = [] if cost_ is None else cost_
    
    for n in range(max_sim):

        ###############################################
        # Forward sweep
        ###############################################
        
        #############
        C       = [exp_c0[:,0].to(dph.device)]
        Csolid  = [exp_c0[:,0].to(dph.device)]
        Ca      = [ca_ic_exp_val[:,0]]
        V       = [v0*torch.ones_like(ph_exp_val[:,0]).to(dph.device)]
        simpH   = [ph_exp_val[:,0]]
        simH    = [torch.float_power(10,-simpH[0])]
        pH      = [ph_exp_val[:,0]]
        H       = [torch.float_power(10,-(pH[0]))]
        F       = [f0.expand(batch.size(0), -1)]
        G_arr   = [torch.zeros_like(ph_exp_val[:,0]).to(dph.device)]
        M_arr   = [torch.zeros_like(ph_exp_val[:,0]).to(dph.device)]
        drhsF_dF_arr    =   [torch.zeros_like(F[-1]).float().to(dph.device)]
        drhsF_dC_arr    =   [torch.zeros_like(F[-1]).float().to(dph.device)]
        drhsF_dCa_arr   =   [torch.zeros_like(F[-1]).float().to(dph.device)]
        drhsF_dH_arr    =   [torch.zeros_like(F[-1]).float().to(dph.device)]
        drhsC_dF_arr    =   [torch.zeros_like(ph_exp_val[:,0]).float().to(dph.device)]
        drhsC_dC_arr    =   [torch.zeros_like(ph_exp_val[:,0]).float().to(dph.device)]
        drhsC_dCa_arr   =   [torch.zeros_like(ph_exp_val[:,0]).float().to(dph.device)]
        drhsC_dH_arr    =   [torch.zeros_like(ph_exp_val[:,0]).float().to(dph.device)]
        drhsCa_dF_arr   =   [torch.zeros_like(ph_exp_val[:,0]).float().to(dph.device)]
        drhsCa_dC_arr   =   [torch.zeros_like(ph_exp_val[:,0]).float().to(dph.device)]
        drhsCa_dCa_arr  =   [torch.zeros_like(ph_exp_val[:,0]).float().to(dph.device)]
        drhsCa_dH_arr   =   [torch.zeros_like(ph_exp_val[:,0]).float().to(dph.device)]
        
        #print(f0.shape,F[-1].shape,pH[-1].shape)
        for k in range(0,len(dph)-1):
            dpH_k     = U_ph[:,k].to(dph.device)
            dCa_k     = U_r[:,k].to(dph.device)
            u_opt_k   = U_opt[:,k] #+ U_r[:,k].to(dph.device)
            #print(pH[-1].shape,dpH_k.shape, F[-1].shape) 
            
            # Update volume and pH states
            preV    = V[-1] + dt*vdot_k #1e2*torch.pow(10,dpH_k-7) * dt
            ph      = torch.clamp((pH[-1])*(1 + torch.sqrt(dt) * sig_H * dW3[:,k]) + 1e2 * dpH_k * dt,min=0,max=14)
            preH    = torch.float_power(10,-ph)
            ph_adj  = ph + ph_shift_const
            
            #print(F[0].shape, Ca[0].shape,H[0].shape,V[0].shape)
            X_vec = F[-1], C[-1], Ca[-1], pH[-1]+ ph_shift_const, V[-1]
            
            # Nucleation rate
            J_vec = N_fn(C[-1],H[-1])
            J, dJ_dC, dJ_dH = J_vec
            #print("----------------->",k,torch.isnan(J.mean()),torch.isnan(dJ_dC.mean()),torch.isnan(dJ_dH.mean()))
            if torch.isnan(J.mean()):
                print(f"NaN detected in J_vec at iteration {J}", k)
                return
            # Growth rate
            G_vec = a_fn(C[-1],H[-1])
            Gt, dG_dC, dG_dH = G_vec

            #print(cts.shape, csk.shape, F[-1].shape)
            
            
            # Birth rate
            rhsF, drhsF_dF, drhsF_dC, drhsF_dCa, drhsF_dH = rhsF_fn(X_vec, J_vec) #(F[-1] * J.view((-1,1))).to(torch.float)
            St      = St_fn(F[-1],x_ax)
            St_star = St_star_fn(F[-1],x_ax)
            St_vec  = (St, St_star)
            # Update particle size distribution
            preF = torch.abs((F[-1].squeeze() + (adOp.matmul(F[-1].T).T * Gt.view((-1,1))) + dt * rhsF.squeeze())).to(torch.float)
            nomralized_F = preF/preF.max(dim=1).values.view(-1,1)
            #print('F:', torch.isnan(rhsF.mean()),torch.isnan(preF.mean()),torch.isnan(nomralized_F.mean()))
            if torch.isnan(drhsF_dF).any() or torch.isnan(drhsF_dC).any() or torch.isnan(drhsF_dCa).any() or torch.isnan(drhsF_dH).any():
                print(f"NaN detected in drhsF at iteration {k}")
                print("drhsF_dF:", drhsF_dF, "drhsF_dC:", drhsF_dC, "drhsF_dCa:", drhsF_dCa, "drhsF_dH:", drhsF_dH)
                return
            
            # Update Ca concentration
            rhsCa, drhsCa_dF, drhsCa_dC, drhsCa_dCa, drhsCa_dH = rhsCa_fn(X_vec, u_opt_k)
            preCa       = torch.clamp(Ca[-1]*(1 + torch.sqrt(dt)*sig_Ca*dW2[:,k]) + dt * rhsCa, min=0, max=1)
            if torch.isnan(drhsCa_dF).any() or torch.isnan(drhsCa_dC).any() or torch.isnan(drhsCa_dCa).any() or torch.isnan(drhsCa_dH).any():
                print(f"NaN detected in drhsCa at iteration {k}")
                print("drhsCa_dF:", drhsCa_dF, "drhsCa_dC:", drhsCa_dC, "drhsCa_dCa:", drhsCa_dCa, "drhsCa_dH:", drhsCa_dH)
                return
            
            # Update CaCO3 concentration
            rhsC, drhsC_dF, drhsC_dC, drhsC_dCa, drhsC_dH    =  rhsC_fn(X_vec, J_vec, G_vec, St_vec, u_opt_k)
            noise_term = C[-1] * sig_C * dW1[:,k]
            preC = torch.clamp(((C[-1] + dt *  rhsC + torch.sqrt(dt) * noise_term)), min=0, max=1) 
            #print(rhsC.mean())
            #print("preC:", preC.shape, preF.shape)
            if torch.isnan(drhsC_dC).any() or torch.isnan(drhsC_dF).any() or torch.isnan(drhsC_dCa).any() or torch.isnan(drhsC_dH).any():
                print(f"NaN detected in drhsC at iteration {k}")
                print("drhsC_dF:", drhsC_dF, "drhsC_dC:", drhsC_dC, "drhsC_dCa:", drhsC_dCa, "drhsC_dH:", drhsC_dH)
                return
            
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
            drhsF_dF_arr.append(drhsF_dF); drhsF_dC_arr.append(drhsF_dC); drhsF_dCa_arr.append(drhsF_dCa); drhsF_dH_arr.append(drhsF_dH)
            drhsCa_dF_arr.append(drhsCa_dF); drhsCa_dC_arr.append(drhsCa_dC); drhsCa_dCa_arr.append(drhsCa_dCa); drhsCa_dH_arr.append(drhsCa_dH)
            drhsC_dF_arr.append(drhsC_dF); drhsC_dC_arr.append(drhsC_dC); drhsC_dCa_arr.append(drhsC_dCa); drhsC_dH_arr.append(drhsC_dH)
        
        
        
        F = torch.stack(F).permute(1,0,2)#.detach().cpu().numpy()
        C = torch.stack(C).permute(1,0)#.detach().cpu().numpy()
        Ca = torch.stack(Ca).permute(1,0)#.detach().cpu().numpy()
        pH = torch.stack(pH).permute(1,0)#.detach().cpu().numpy()
        H = torch.stack(H).permute(1,0)#.detach().cpu().numpy()
        simpH = torch.stack(simpH).permute(1,0)#.detach().cpu().numpy()
        simH = torch.stack(simH).permute(1,0)#.detach().cpu().numpy()
        V = torch.stack(V).permute(1,0)#.detach().cpu().numpy()
        G_arr = torch.stack(G_arr).permute(1,0)#.detach().cpu().numpy()
        M_arr = torch.stack(M_arr).permute(1,0)#.detach().cpu().numpy()
        CO3 = CO3_fn(H)#.detach().cpu().numpy()
        drhsF_dF_arr = torch.stack(drhsF_dF_arr).permute(1,0,2)#.detach().cpu().numpy()
        drhsF_dC_arr = torch.stack(drhsF_dC_arr).permute(1,0,2)#.detach().cpu().numpy()
        drhsF_dCa_arr = torch.stack(drhsF_dCa_arr).permute(1,0,2)#.detach().cpu().numpy()
        drhsF_dH_arr = torch.stack(drhsF_dH_arr).permute(1,0,2)#.detach().cpu().numpy()
        
        drhsCa_dF_arr = torch.stack(drhsCa_dF_arr).permute(1,0)#.detach().cpu().numpy()
        drhsCa_dC_arr = torch.stack(drhsCa_dC_arr).permute(1,0)#.detach().cpu().numpy()
        drhsCa_dCa_arr = torch.stack(drhsCa_dCa_arr).permute(1,0)#.detach().cpu().numpy()
        drhsCa_dH_arr = torch.stack(drhsCa_dH_arr).permute(1,0)#.detach().cpu().numpy()
        
        drhsC_dF_arr = torch.stack(drhsC_dF_arr).permute(1,0)#.detach().cpu().numpy()
        drhsC_dC_arr = torch.stack(drhsC_dC_arr).permute(1,0)#.detach().cpu().numpy()
        drhsC_dCa_arr = torch.stack(drhsC_dCa_arr).permute(1,0)#.detach().cpu().numpy()
        drhsC_dH_arr = torch.stack(drhsC_dH_arr).permute(1,0)#.detach().cpu().numpy()

        ###############################################
        # Backward sweep
        ###############################################
        
        # compute terminal cost
        ca_residual     = 5*(Ca - ca_ic_exp_val[:,:len(dph)]).float().to(dph.device)
        ca_loss         = 0.5*(ca_residual**2).mean(dim=0).to(dph.device)
        
        # Adjoint variables
        Y_F   = [torch.zeros_like(F[:,-1]).float().to(dph.device)]
        Y_C   = [torch.zeros_like(ph_exp_val[:,0]).float().to(dph.device)]
        Y_Ca  = [-0*ca_residual[:,-1].float()]
        Y_H   = [torch.zeros_like(ph_exp_val[:,0]).float().to(dph.device)]
        Z_C   = [torch.zeros_like(ph_exp_val[:1,0]).float().to(dph.device)]
        Z_Ca  = [torch.zeros_like(ph_exp_val[:1,0]).float().to(dph.device)]
        Z_H   = [torch.zeros_like(ph_exp_val[:1,0]).float().to(dph.device)]
        dt_inv = 1/dt
        for i in range(len(dph)-2,-1,-1):
            Z_Ca_i      = dt_inv*torch.mean(dW2[:,i]*Y_Ca[-1],dim=0,keepdim=True).float()
            rhs_Y_Ca_i  =   (drhsCa_dCa_arr[:,i] * Y_Ca[-1]) #+ (drhsF_dCa_arr[:,i]  * Y_F[-1]).mean(dim=1) + (drhsC_dCa_arr[:,i]  * Y_C[-1])
                            
            Y_Ca_i   = torch.clamp(Y_Ca[-1]   + dt*1*rhs_Y_Ca_i -  dt*1*ca_residual[:,i]  + dt*1*Z_Ca_i*sig_Ca   + .02*torch.sqrt(dt)*Z_Ca_i*dW2[:,i],min=-.01,max=.01)
            Y_Ca.append(Y_Ca_i);
        
        Y_Ca = torch.stack(Y_Ca).permute(1,0).flip((1,));
        Z_Ca = torch.stack(Z_Ca).flip((0,)); 

        # Compute new control
        eta = .00005
        U_opt = torch.clamp(U_opt - .1*eta*(U_opt - U_r[:,:len(dph)]) - 5*eta*(Y_Ca*1), min=-1,max=1)
        #print(U_opt,rhs_Y_Ca_i)
        loss = ca_loss.sum().item()
        cost.append(loss)
        print(f"Exp: {exp_typ}, iter: {n}, J(u): {loss:.4f} \n", flush=True)

        ph_exp_t = batch[0,:len(dph),0].detach().cpu().numpy()

        # collect plotting data and return it
        pdata = {'ph_exp_t': ph_exp_t, 'pH': pH, 'simpH': simpH, 'U_opt': U_opt, 'U_r': U_r, 'ca_ic_exp_val': ca_ic_exp_val, 'Ca': Ca, 'Y_Ca': Y_Ca,\
                  'cost': cost, 'iter': n, 'dph': dph, 'exp_typ': exp_typ}
        #print('U_opt:', U_opt.shape, U_opt[0,:len(dph)].shape, U_opt[0,:len(dph)].detach().cpu().numpy()

        if DEBUG:
            # plotting
            plot_iter(pdata)

            plt.pause(.1)
    
    if SAVE:
        with open(f'../results/precip_design/data/fbsde_exp{int(exp_typ)}.pkl', 'wb') as f:
            pickle.dump(pdata, f)
        print(f"Results saved to ../results/precip_design/data/fbsde_exp{int(exp_typ)}.pkl")
    return U_opt,cost,pdata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run or plot the classical FBSDE algorithm.")
    parser.add_argument('-m', "--mode", type=str, choices=["opt", "plot"], required=True, help="Mode to run: 'opt' or 'plot'")
    parser.add_argument('-e', "--exp_typ", type=int, default="1", help="Which experiment to run")
    parser.add_argument('-N', "--max_sim", type=int, default="100", help="Nuber of iterations")
    parser.add_argument("--cont", action="store_true", help="Load a pre-trained model")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    exp_typ = int(args.exp_typ)
    mode = args.mode
    cont = args.cont
    max_sim = args.max_sim
    
    print(f"Running experiment {int(exp_typ)} in {mode} mode with max_sim={max_sim} and cont={cont} \n", flush=True)

    if not os.path.exists('../results/precip_design/data'):
        print("Creating data directory...")
        os.makedirs('../results/precip_design/data', exist_ok=True)
    if not os.path.exists('../results/precip_design/figures'):
        print("Creating figures directory...")
        os.makedirs('../results/precip_design/figures', exist_ok=True)
    
    # Load the data
    batches     = [batch1, batch2, batch3, batch4]
    par_vecs    = [par_vec1, par_vec2, par_vec3, par_vec4]

    batch = batches[exp_typ-1]
    par_vec = par_vecs[exp_typ-1]   

    if mode == "plot" or cont:
        with open(f'../results/precip_design/data/fbsde_exp{int(exp_typ)}.pkl', 'rb') as f:
            pdata = pickle.load(f)
    else:
        pdata = None

    if mode == "opt":
        if pdata is not None:
            uopt = pdata['U_opt']
            cost = pdata['cost']
            uopt, cost, pdata = precip_fbsde_solve(batch, par_vec, max_sim=max_sim, exp_typ=exp_typ, U=uopt, cost_=cost, SAVE=True)
        else:
            uopt, cost, pdata = precip_fbsde_solve(batch, par_vec, max_sim=max_sim, exp_typ=exp_typ, SAVE=True)
    elif mode == "plot":
        if pdata is not None:
            pdata['iter'] = 0
            plot_iter(pdata, True)