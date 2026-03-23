#########################################################################
# --------------------------------------------------------------------- #
# | Author      : Sandesh Athni Hiremath                                |
# |---------------------------------------------------------------------|
# | Description :                                                       |
# |   This script provides utility functions and model coefficient      |
# |   definitions for precipitation process simulations. It includes    |
# |   mathematical models for nucleation, growth, and dissolution       |
# |   rates, their derivatives, and right-hand side functions for       |
# |   system ODEs. It also provides finite difference operators for     |
# |   numerical PDE solutions and helper functions for sparse matrix    |
# |   conversions.                                                      |
# |---------------------------------------------------------------------|
# | Usage       : Import this module to access:                         |
# |   - Model coefficient functions (e.g., a_fn, N_fn, D_fn)            |
# |   - Derivative functions for model coefficients                     |
# |   - Right-hand side functions for ODE/PDE systems                   |
# |   - Finite difference operator generators (e.g., get1DGradOp)       |
# |   - Utility for converting scipy sparse matrices to torch tensors   |
# --------------------------------------------------------------------- #
#########################################################################

import numpy as np
import scipy.sparse as sps
import torch
from matplotlib import pyplot as plt

# Figure setup
def setup_figure():
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = 10

# Model constants 
K_sp    = 8.5e-7
K_a1    = 10**(-2.35)
K_a2    = 10**(-6.3)
d       = 1e-6
v_nuc   = 3.142/6*d**3
K_co2   = 1e-3
k_D     = 1

c0k     = 0.02660198137164116
v0      = 5.18315315246582

vdot_k   = 1e-2
kg_k     = 0.45933839678764343 
rho_k    = 0.3154085576534271
nu_k     = 1.0

# Model functions as lambdas
csat_fn              = lambda co3: 1.2e-10*torch.clamp(torch.sqrt(K_sp/co3),min=0,max=1e3) 
csat_fn_prime        = lambda co3: -0.5 * torch.sqrt(K_sp) / (co3**1.5)
csat_simp_fn         = lambda co3: 1.09e1/(1+9e10*torch.sqrt(co3/K_sp)) #  simplified csat function
csat_simp_fn_prime   = lambda co3: -0.5 * 9e10 * 1.09e1 / (torch.sqrt(K_sp*co3) * (1 + 9e10 * torch.sqrt(co3 / K_sp))**2)
St_fn                  = lambda F_t,x: torch.abs(((torch.float_power(x,2)).view((1,-1)) * F_t.squeeze()).mean(axis=1))
St_star_fn             = lambda F_t,x: torch.abs(((torch.float_power(x,2)).view((1,-1)) * torch.ones_like(F_t.squeeze())).mean(axis=1))


# Model coefficients as functions of state variables
def CO3_concentration(H, C_T):
    """Bicarbonate ion concentration"""
    H_plus = 10**(-H)
    denominator = H_plus**2 + K_a1*H_plus + K_a1*K_a2
    return C_T * K_a1 * K_a2 / denominator

def C_total(H):
    """Total carbonate concentration"""
    H_plus = 10**(-H)
    return K_co2 * (1 + K_a1/H_plus + K_a1*K_a2/H_plus**2)

def CO3_fn(H):
    """Bicarbonate ion concentration"""
    k_p1 = 1e3; k_p2 = 0.55; h1 = 7.1; h2 = 13.5
    ret = k_p1/(1+torch.exp(-k_p2*((H - h1)*(H + h2))))
    return ret

def CO3_fn_prime(H):
    """Bicarbonate ion concentration"""
    k_p1 = 1e3; k_p2 = 0.55; h1 = 7.1; h2 = 13.5
    exp_term = torch.exp(-k_p2*((H - h1)*(H + h2)))
    term1 = k_p1/(1+exp_term)**2
    term2 = k_p2 * (2*H + h2 - h1) * exp_term
    ret = term1*term2  
    # print("CO3_fn_prime:", term1, term2)
    return ret

def a_fn(C,H):
    """Growth rate function"""
    fac = 2e-3
    co3 = CO3_fn(H)
    csat = csat_simp_fn(co3)
    cts = C/csat - 1
    #print(C.mean().item(), csat_simp_fn(co3).mean().item(), cts.mean().item())
    pos_cts = torch.clamp(cts,min=0).to(torch.float)
    pos_mask = (pos_cts > 0).to(torch.float)
    gt =  1 * 1e1 * kg_k * torch.tanh(pos_cts**2)
    Gt =  fac * 1e3 *  gt  # 1e3 is to convert gt to mm/s
    #print(co3, csat, Gt)
    # da_dc
    dgt_dC = (1 * 1e1 * kg_k * pos_mask / csat) * (1 - torch.tanh(pos_cts**2)) * 2 * pos_cts
    dGt_dC =fac * 1e3 * dgt_dC
    
    # da_dh
    dco3_dH = CO3_fn_prime(H)
    dcts_dco3 = csat_simp_fn_prime(co3)
    dcts_dH = -C / (csat_simp_fn(co3)**2) * dcts_dco3 * dco3_dH
    dgt_dH = 1 * 1e1 * kg_k * pos_mask * dcts_dH * (1 - torch.tanh(pos_cts**2)) * 2 * pos_cts
    dGt_dH = fac * 1e3 * dgt_dH
    return Gt, dGt_dC, dGt_dH

def N_fn(C,H):
    """Nucleation rate"""
    # Kb -> Boltzman constant in [J/K]
    # gamma_in -> Interfacial tension in [N/m]
    # T -> Experimental temp in [-1]
    # v_nuc -> Molecular volume in [m3]

    k_A = 1e0 # pre-exponential factor for Na2CO3 in [1/s]
    k_B = 0.2843
    
    co3 = CO3_fn(H)
    csat = csat_simp_fn(co3)
    cts = C / csat - 1
    pos_cts = torch.clamp(cts,min=0).to(torch.float)
    pos_mask = (pos_cts > 0).to(torch.float)
    eps = 1e-6
    Nt = k_A * torch.exp(-(k_B/(eps + pos_cts**2)))
    # dN_dC
    dcts_dC = 1 / csat
    dN_dC = Nt * pos_mask * (k_B / (eps + pos_cts**2)**2) * dcts_dC * 2 * pos_cts
    
    # dN_dH
    dco3_dH = CO3_fn_prime(H)
    dcts_dco3 = csat_simp_fn_prime(co3)
    dcts_dH = -C / (csat**2) * dcts_dco3 * dco3_dH
    dN_dH = Nt * pos_mask * (k_B /(eps + pos_cts**2)**2) * dcts_dH * 2 * pos_cts

    #print('Nuc rate:', C, co3, csat, pos_cts, dN_dC, dN_dH)
    
    return Nt, dN_dC, dN_dH

def D_fn(C,H):
    """Dissolution rate"""
    co3 = CO3_fn(H)
    cts = C / csat_simp_fn(co3) - 1
    Dt  = k_D * (cts**2 / (1 + (cts+1)**2))
    
    # dD_dC
    dcts_dC = 1 / csat_simp_fn(co3)
    dD_dC = k_D * (2 * cts * dcts_dC / (1 + (cts + 1)**2) - 
                    (cts**2 * 2 * (cts + 1) * dcts_dC) / (1 + (cts + 1)**2)**2)
    
    # dD_dH
    dco3_dH = CO3_fn_prime(H)
    dcts_dco3 = csat_simp_fn_prime(co3)
    dcts_dH = -C / (csat_simp_fn(co3)**2) * dcts_dco3 * dco3_dH
    dD_dH = k_D * (2 * cts * dcts_dH / (1 + (cts + 1)**2) - 
                    (cts**2 * 2 * (cts + 1) * dcts_dH) / (1 + (cts + 1)**2)**2)
    
    return Dt, dD_dC, dD_dH

# right-hand side functions of the precipitation model
def rhsF_fn(X_vec, J_vec):
    F, C, Ca, H, V      = X_vec
    
    # Nucleation rate
    J, dJ_dC, dJ_dH     = J_vec
    
    
    rhsF = (F * J.view((-1,1))).to(torch.float)
    
    drhsF_dF, drhsF_dC, drhsF_dCa, drhsF_dH = J.view(-1,1)*torch.ones_like(F), F * dJ_dC.view(-1,1), 0 * F*dJ_dC.view(-1,1), F * dJ_dH.view(-1,1)
    
    # print("F,J,rhsF,dF,dC,dCa,dH:", torch.isnan(F.mean()), torch.isnan(J.mean()), torch.isnan(rhsF.mean()),\
    #     torch.isnan(drhsF_dF.mean()), torch.isnan(drhsF_dC.mean()),\
    #         torch.isnan(drhsF_dCa.mean()), torch.isnan(drhsF_dH.mean()))
    
    return rhsF.float(), drhsF_dF.float(), drhsF_dC.float(), drhsF_dCa.float(), drhsF_dH.float()

def rhsCa_fn(X_vec, U_r):
    # Nucleation rate
    F, C, Ca, H, V  = X_vec
    
    co3             = CO3_fn(H)
    rca_rhs         = (Ca * co3 *  U_r) 
    dilution_ca     = Ca*vdot_k/V
    rhsCa           = -(rca_rhs + dilution_ca)
    
    dco3_dH         = CO3_fn_prime(H)
    drhsCa_dH       = Ca*dco3_dH*U_r
    drhsCa_dCa      = -co3 * U_r -vdot_k/V 
        
    drhsCa_dF       = torch.zeros_like(drhsCa_dH) # Ensure correct shape and device
    drhsCa_dC       = torch.zeros_like(drhsCa_dH) # Ensure correct shape and device
    
    #print("rhsCa: ", rhsCa.shape, drhsCa_dF.shape, drhsCa_dC.shape, drhsCa_dCa.shape, drhsCa_dH.shape)
    
    return rhsCa.float(), drhsCa_dF.float(), drhsCa_dC.float(), drhsCa_dCa.float(), drhsCa_dH.float()


def rhsC_fn(X_vec, J_vec, G_vec, St_vec, U_r):
    # Nucleation rate
    F, C, Ca, H, V      = X_vec
    J, dJ_dC, dJ_dH     = J_vec
    Gt, dG_dC, dG_dH    = G_vec
    St, St_star         = St_vec
    
    co3         = CO3_fn(H)
    rca_rhs     = (Ca * co3 *  U_r)
    
    dis         = rca_rhs #
    dil         = -C[-1] * vdot_k/V
    dec         = -rho_k * nu_k * Gt * St / V - rho_k * v_nuc * J / V
    rhsC        = dis + dil + dec
    
    #print("rhsC: ", rhsC.shape, drhsC_dF.shape, drhsC_dC.shape, drhsC_dCa.shape, drhsC_dH.shape)
    dco3_dH     = CO3_fn_prime(H)
    d_dis_dH    = -Ca * dco3_dH * U_r
    d_dis_dCa   = -dco3_dH * U_r
    
    d_dil_dC    = -vdot_k/V
    
    d_dec_dF    = -rho_k * nu_k * Gt * St_star / V
    d_dec_dC    = -rho_k * nu_k * dG_dC * St / V - rho_k * v_nuc * dJ_dC / V
    d_dec_dH    = -rho_k * nu_k * dG_dH * St / V - rho_k * v_nuc * dJ_dH / V
    
    drhsC_dF    = d_dec_dF
    drhsC_dCa   = d_dis_dCa
    drhsC_dC    = d_dil_dC + d_dec_dC
    drhsC_dH    = d_dis_dH + d_dec_dH
    
    #print("rhsC: ", rhsC.shape, drhsC_dF.shape, drhsC_dC.shape, drhsC_dCa.shape, drhsC_dH.shape)
    return rhsC.float(), drhsC_dF.float(), drhsC_dC.float(), drhsC_dCa.float(), drhsC_dH.float()

# Finite difference operators
def get1DGradOp(N, coef=None):
    if coef is None:
        coef = np.ones(N)

    did = np.arange(N)

    udid = did + 1
    D = sps.lil_matrix((N, N))
    D[did, did] = -1 * coef[did]
    D[did[:-1], udid[:-1]] = 1 * coef[udid[:-1]]
    D[-1, -1] = 0  # D[0,:] = 0;
    return D


def get2DGradOp(M, N, coef=None):
    if coef is None:
        coef = np.ones(N)

    O = max(M, N)
    P = min(M, N)
    h = O - P

    # create 1D discrete grad operator
    D = get1DGradOp(O, coef)

    # distribute the 1D op into 2D
    if M > N:
        return sps.kron(sps.eye(O), D[h:, h:]), sps.kron(D, sps.eye(P))
    else:
        return sps.kron(sps.eye(P), D), sps.kron(D[h:, h:], sps.eye(O))


def get1DLapOp(N, coef=None):
    if coef is None:
        coef = np.ones(N)

    did = np.arange(N)
    udid = did + 1
    ldid = did - 1
    D = sps.lil_matrix((N, N))
    D[did, did] = 2 * coef[did]
    D[did[1:-1], udid[1:-1]] = -1 * coef[udid[1:-1]]
    D[did[1:-1], ldid[1:-1]] = -1 * coef[ldid[1:-1]]
    D[0, 1] = -1 * coef[1]
    D[-1, -2] = -1 * coef[-2]
    D[0, 0] = 1 * coef[0]
    D[-1, -1] = 1 * coef[-1]
    return D.tocoo()


def get2DLapOp(M, N, coef=[]):
    O = max(M, N);
    P = min(M, N);
    h = O - P;

    # create 1D discrete Laplace operator
    D = get1DLapOp(O, coef);
    subD = D[h:, h:].copy();

    # distribute the 1D op into 2D
    if M > N:
        subD[0, 1] = -2;
        return sps.kron(sps.eye(O), subD), sps.kron(D, sps.eye(P));
    else:
        return sps.kron(sps.eye(P), D), sps.kron(subD, sps.eye(O));


def getFBDiffOps1D(M, ifTorch=False):
    """
    Set of finite difference operators for solving PDEs in one dimension using a forward-backward (FB) method
    """
    D_x = get1DGradOp(M)
    # Create copies as lil_matrix to avoid CSR matrix modification warnings
    Dxf = D_x.copy().tolil()
    Dxb = (-D_x.T).tolil()
    Dxf[0, :] = 0
    Dxf[0, 0] = 1
    Dxf[-1, -1] = 1
    Dxb[-1, :] = 0
    Dxb[0, 0] = 1
    Dxb[-1, -1] = 1
    Dxfab = np.abs(Dxf)
    Dxbab = np.abs(Dxb)

    if not ifTorch:
        return Dxf.tocoo(), Dxb.tocoo(), Dxfab.tocoo(), Dxbab.tocoo()
    else:
        return npSps2Torch(Dxf.tocoo()), npSps2Torch(Dxb.tocoo()), npSps2Torch(Dxfab.tocoo()), npSps2Torch(
            Dxbab.tocoo())


def npSps2Torch(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))