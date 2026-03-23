#########################################################################
# --------------------------------------------------------------------- #
# | Author      : Sandesh Athni Hiremath                                |
# |---------------------------------------------------------------------|
# | Description :                                                       |
# |   This script provides utility functions for loading, processing,   |
# |   and serializing experimental precipitation process data. It       |
# |   includes routines for reading Excel files, cleaning and merging   |
# |   datasets, interpolating missing values, and normalizing features. |
# |   The processed data is intended for use in downstream modeling     |
# |   and analysis tasks.                                               |
# |---------------------------------------------------------------------|
# | Usage       : Import this module and call the provided functions:   |
# |   - load_data(): Load and preprocess raw experimental data.         |
# |   - serialize_df(): Merge and interpolate experiment dataframes.    |
# |   - proc_data(): Return processed and normalized datasets.          |
# --------------------------------------------------------------------- #
#########################################################################



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=(12, 8)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

pd.set_option('future.no_silent_downcasting', True)

def extract_psd(df, n_bins=64, psd_start_col=-101, skip_rows=100):
    """
    Interpolate PSD curves in a dataframe to n_bins and fill missing rows using 2D interpolation.

    Args:
        df (pd.DataFrame): Input dataframe containing PSD columns.
        n_bins (int): Number of bins to interpolate to.
        psd_start_col (int): Starting column index for PSD data (default: last 101 columns).

    Returns:
        F_interp (np.ndarray): 2D array of interpolated PSDs (shape: [rows, n_bins]).
    """
    m, n = df.shape
    if n < 100:
        print(f"Warning: Dataframe does not have PSD data")
        return None, None
    
    F = np.zeros((m, n_bins))
    valid = ~df['ca_ic'].isna()
    psd = df[valid].iloc[:, psd_start_col:].fillna(0).to_numpy().astype(float)

    # Interpolate each PSD curve to n_bins along axis 1
    x_old = np.linspace(0, psd.shape[1] - 1, psd.shape[1])
    x_new = np.linspace(0, psd.shape[1] - 1, n_bins)
    psd_interp = np.array([interp1d(x_old, row, kind='linear')(x_new) for row in psd])
    F[valid] = psd_interp
    
    # Shift columns to the right, increasing shift with row index
    psd_shifted = np.zeros_like(F)
    for i in range(F.shape[0]):
        shift = i // (F.shape[0] // 10)  # Example: max shift ~10 bins
        psd_shifted[i] = np.roll(F[i], shift)
    F = psd_shifted
    # Find indices of non-zero rows in F
    nonzero_rows = np.where(F.sum(axis=1) > 0)[0]
    x = nonzero_rows
    y = np.arange(F.shape[1])

    # Prepare grid for interpolation
    X, Y = np.meshgrid(np.arange(F.shape[0]), np.arange(F.shape[1]), indexing='ij')

    # Collect known points and values
    points = np.array([(i, j) for i in x for j in y])
    values = np.array([F[i, j] for i in x for j in y])

    # Interpolate over the full grid
    F_interp = griddata(points, values, (X, Y), method='linear', fill_value=0)
    # Apply Gaussian smoothing along both axes for better continuity
    F_interp = gaussian_filter(F_interp, sigma=(15, 1), mode='nearest')
    F_interp = F_interp / np.nanmax(F_interp)
    df['psd'] = [row for row in F_interp]
    return df

def load_data(data_path='..'):
    exp1 = f'{data_path}/data/lab_exp_data/PrecipitationData/17092024/20240814_combined.xlsx'
    exp2 = f'{data_path}/data/lab_exp_data/PrecipitationData/17092024/20240822_combined.xlsx'
    exp3 = f'{data_path}/data/lab_exp_data/PrecipitationData/16032025/Experiment.xlsx'

    cnames = ['date','atime', 't', 'pH', 'H+', 'ca_ise', 'ca_ic', 'mg_ic', 'na_ic', 'cl_ic','psd_mean'] + [f'psd_bin{k}' for k in range(101)]

    df1 = pd.read_excel(exp1,'Combined_Hist_pH_Conc')
    df2 = pd.read_excel(exp2,'Combined_Hist_pH_Conc')

    df1.rename(columns=dict(zip(df1.columns,cnames)),inplace=True)
    df2.rename(columns=dict(zip(df2.columns,cnames)),inplace=True)
    df1 = df1.iloc[2:]
    df2 = df2.iloc[2:]; df2['ca_ise'] = 1e-3*df2['ca_ise']
    df1['t1'] = df1['t']; df2['t1'] = df2['t'];
    df1['t2'] = df1['t']; df2['t2'] = df2['t'];
    df3 = pd.read_excel(exp3, sheet_name='Swing profile')
    df4 = pd.read_excel(exp3, sheet_name='Constant profile')
    df3.drop(columns=['Unnamed: 2', 'Unnamed: 6'], inplace=True)
    df3.drop(index=0, inplace=True)
    df3.rename(columns=dict(zip(df3.columns,['t', 'pH','t1','mg_ic','ca_ic','t2','ca_ise'])), inplace=True)
    df4.drop(columns=['Unnamed: 2', 'Unnamed: 6'], inplace=True)
    df4.drop(index=0, inplace=True)
    df4.rename(columns=dict(zip(df4.columns,['t', 'pH','t1','mg_ic','ca_ic','t2','ca_ise'])), inplace=True)
    selected_cols = ['t', 'pH','t1','mg_ic','ca_ic','t2','ca_ise']
    ndf1 = df1[selected_cols].copy()
    ndf2 = df2[selected_cols].copy()
    ndf3 = df3[selected_cols].copy()
    ndf4 = df4[selected_cols].copy()

    dfs = [df1, df2, df3, df4]
    ndfs = [ndf1, ndf2, ndf3, ndf4]

    return dfs, ndfs

def serialize_df(df,exp):
    df_ph = df.loc[:,['t','pH']].copy() 
    df_ic = df.loc[~df['t1'].isna(),['t1','mg_ic','ca_ic']].copy()
    df_ise = df.loc[~df['t2'].isna(),['t2','ca_ise']].copy()
    df_ph['time'] = df_ph['t'].copy()
    df_ic['time'] = df_ic['t1'].copy()
    df_ise['time'] = df_ise['t2'].copy()
    len(df_ph), len(df_ic), len(df_ise)
    common_t = np.intersect1d(df_ph['t'], df_ic['t1'])
    common_df = pd.DataFrame(df_ph[df_ph['t'].isin(common_t)],columns=['t','pH'])
    common_df['mg_ic'] = df_ic[df_ic['t1'].isin(common_t)]['mg_ic'].values
    common_df['ca_ic'] = df_ic[df_ic['t1'].isin(common_t)]['ca_ic'].values
    common_df['ca_ise'] = df_ise[df_ise['t2'].isin(common_t)]['ca_ise'].values
    common_df.rename(columns=dict(zip(common_df.columns,['time', 'pH','mg_ic','ca_ic','ca_ise'])), inplace=True)
    rem_df_ph = df_ph[~df_ph['t'].isin(common_t)][['time', 'pH']]	
    rem_df_ic = df_ic[~df_ic['t1'].isin(common_t)][['time', 'mg_ic','ca_ic']]
    rem_df_ise = df_ise[~df_ise['t2'].isin(common_t)][['time', 'ca_ise']]
    merged_df = pd.concat([common_df,rem_df_ph,rem_df_ic,rem_df_ise],axis=0)
    merged_df.sort_values(by='time',inplace=True)
    merged_df.reset_index(drop=True,inplace=True)

    #now merge ise column

    #merged_df
    merged_df['valid_ic'] = [1]*len(merged_df)
    merged_df['valid_ise'] = [1]*len(merged_df)
    merged_df.loc[merged_df['ca_ic'].isna(),'valid_ic'] = 0
    merged_df.loc[merged_df['ca_ise'].isna(),'valid_ise'] = 0
    merged_df['exp'] = exp
    merged_df[['pH', 'mg_ic', 'ca_ic', 'ca_ise']] = merged_df[['pH', 'mg_ic', 'ca_ic', 'ca_ise']].apply(pd.to_numeric, errors='coerce')
    merged_df[['pH', 'mg_ic', 'ca_ic', 'ca_ise']] = merged_df[['pH', 'mg_ic', 'ca_ic', 'ca_ise']].interpolate(method='linear', limit_direction='both')
    merged_df['ca0'] = merged_df.iloc[0,3] 
    merged_df['ph_shift'] = (merged_df.iloc[0,1] - merged_df.iloc[100,1])
    merged_df[['nmg_ic', 'npH', 'nca_ic']] = scaler.fit_transform(merged_df[['mg_ic', 'pH', 'ca_ic']])
    
    return merged_df.astype(float)

def proc_data(data_path='..'):
    dfs, ndfs = load_data(data_path=data_path)
   
    df1, df2, df3, df4 = dfs
    ndf1, ndf2, ndf3, ndf4 = ndfs

    ser_ndf1 = df1[['t','pH','mg_ic','ca_ic','ca_ise']].copy()
    ser_ndf1.rename(columns=dict(zip(ser_ndf1.columns,['time', 'pH','mg_ic','ca_ic','ca_ise'])), inplace=True)
    ser_ndf1['valid_ic'] = [1]*len(ser_ndf1)
    ser_ndf1.loc[ser_ndf1['ca_ic'].isna(),'valid_ic'] = 0
    ser_ndf1['valid_ise'] = [1]*len(ser_ndf1)
    ser_ndf1.loc[ser_ndf1['ca_ise'].isna(),'valid_ise'] = 0
    ser_ndf1['exp'] = 1
    ser_ndf1[['pH', 'mg_ic', 'ca_ic','ca_ise']] = ser_ndf1[['pH', 'mg_ic', 'ca_ic','ca_ise']].apply(pd.to_numeric, errors='coerce')
    ser_ndf1[['pH', 'mg_ic', 'ca_ic','ca_ise']] = ser_ndf1[['pH', 'mg_ic', 'ca_ic','ca_ise']].interpolate(method='linear', limit_direction='both')
    #ser_ndf1['caco3'] = ser_ndf1.iloc[0,3] - ser_ndf1['ca_ic'] 
    ser_ndf1['ca0'] = ser_ndf1.iloc[0,3]
    ser_ndf1['ph_shift'] = (ser_ndf1.iloc[0,1] - ser_ndf1.iloc[100,1])
    ser_ndf1[['nmg_ic', 'npH', 'nca_ic']] = scaler.fit_transform(ser_ndf1[['mg_ic', 'pH', 'ca_ic']])

    ser_ndf2 = df2[['t','pH','mg_ic','ca_ic','ca_ise']].copy()
    ser_ndf2.rename(columns=dict(zip(ser_ndf2.columns,['time', 'pH','mg_ic','ca_ic'])), inplace=True)
    ser_ndf2['valid_ic'] = [1]*len(ser_ndf2)
    ser_ndf2.loc[ser_ndf2['ca_ic'].isna(),'valid_ic'] = 0
    ser_ndf2['valid_ise'] = [1]*len(ser_ndf2)
    ser_ndf2.loc[ser_ndf2['ca_ise'].isna(),'valid_ise'] = 0
    ser_ndf2['exp'] = 2
    ser_ndf2[['pH', 'mg_ic', 'ca_ic','ca_ise']] = ser_ndf2[['pH', 'mg_ic', 'ca_ic','ca_ise']].apply(pd.to_numeric, errors='coerce')
    ser_ndf2[['pH', 'mg_ic', 'ca_ic','ca_ise']] = ser_ndf2[['pH', 'mg_ic', 'ca_ic','ca_ise']].interpolate(method='linear', limit_direction='both')
    #ser_ndf2['caco3'] = ser_ndf2.iloc[0,3] - ser_ndf2['ca_ic'] 
    ser_ndf2['ca0'] = ser_ndf2.iloc[0,3] 
    ser_ndf2['ph_shift'] = (ser_ndf2.iloc[0,1] - ser_ndf2.iloc[100,1])
    ser_ndf2[['nmg_ic', 'npH', 'nca_ic']] = scaler.fit_transform(ser_ndf2[['mg_ic', 'pH', 'ca_ic']])


    ser_ndf3 = serialize_df(ndf3,3)
    ser_ndf4 = serialize_df(ndf4,4)

    return ser_ndf1, ser_ndf2, ser_ndf3, ser_ndf4