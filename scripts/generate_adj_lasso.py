import numpy as np
import pandas as pd
import argparse
import os

from sklearn.covariance import GraphicalLasso
from regain.covariance import TimeGraphicalLasso

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.helper.data_generation import time_varying_power_network, sample_covariance_matrix, generate_precision_matrix
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, error
from gglasso.helper.utils import get_K_identity
from gglasso.helper.experiment_helper import plot_evolution, plot_deviation, surface_plot, single_heatmap_animation
from gglasso.helper.model_selection import aic, ebic
from gglasso.problem import glasso_problem
import matplotlib.pyplot as plt
import networkx as nx


# L = int(p/M)

# reg = 'FGL'

# Sigma, Theta = time_varying_power_network(p, K, M=5, scale = False, seed = 2340)
# S, sample = sample_covariance_matrix(Sigma, N)

# results = {}
# results['truth'] = {'Theta' : Theta}

# # anim = single_heatmap_animation(Theta)

# l1 = 0.1
# l2 = 0.05
# tmp = sample.transpose(1,0,2).reshape(p,-1).T


# Omega_0 = get_K_identity(K,p)

# sol, info = ADMM_MGL(S, l1, l2, reg, Omega_0, rho = 1, max_iter = 1000, \
#                                                         tol = 1e-10, rtol = 1e-10, verbose = False, measure = True)
# #ltgl = ltgl.fit(X = tmp, y = np.repeat(np.arange(K),N))

# results['ADMM'] = {'Theta' : sol['Theta']}


# diff= results['ADMM']['Theta']

# create time stamp in first dimension

# Sigma, Theta = generate_precision_matrix(p=20, M=1, style='erdos', prob=0.1, seed=1234)

# S, sample = sample_covariance_matrix(Sigma, N)

# P = glasso_problem(S, N, reg_params={'lambda1': 0.05}  ,latent = False,  do_scaling = False)
# print(P)

# lambda1_range = np.logspace(0, -3, 30)
# modelselect_params = {'lambda1_range': lambda1_range}

# # P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = 0.1)

# # regularization parameters are set to the best ones found during model selection

# P.solve()
# print(P.reg_params)

# sol = P.solution.precision_
#res[]

def generate_lasso(args):
    if args.df_filename[-3:] =='.h5':
        df = pd.read_hdf(args.df_filename)
    elif args.df_filename[-4:] =='.csv':
        df = pd.read_csv(args.df_filename)
    len_data = df.shape[0]

    K= args.len_interval
    mode = args.mode
    modify_cov= args.cov_modify

    eps = 0.001
    
    if mode =='static':
    
        t_test = round(len_data * 0.9)
        t_train = round(len_data * 0.8)
        
    elif mode=='dynamic':

        t_train = int(round(len_data * 0.8)/K) *K
        t_test = round(len_data * 0.9)
    
    print("len train:", t_train)
    print("len test:", len_data -t_train)
    train =df.iloc[:t_train,:]
    val =df.iloc[t_train: t_test,:]
    test=df.iloc[t_test: ,:]
    
    #S= np.array(train.cov(numeric_only=True))
    p= train.shape[1]
    train_scaled = StandardScaler().fit_transform(train)

    if mode == "static":
        S= np.array(np.cov(train_scaled.T))
        if modify_cov == True:
            S+= eps*np.eye(p,p)    

        l1 = args.Lambda
        P = glasso_problem(S, t_train, reg_params = {'lambda1': l1}, latent = False, do_scaling = args.scaling)
        P.solve(tol = 1e-10, rtol = 1e-10,)
        

        sol = P.solution.precision_
        #P.solution.calc_adjacency(t = 1e-4)

        result = sol


    elif mode =="dynamic":
        number_intervals = int(t_train/K)
        Omega_0 = get_K_identity(number_intervals,p)
        S=Omega_0.copy()

        for i in range(number_intervals):
            
            S[i,:,:] = np.array(train.iloc[K*i:K*(i+1),:].cov())
            if modify_cov == True:
                S[i,:,:]+= eps*np.eye(p,p) 

        l1 = args.Lambda
        l2 = args.Beta

        sol, info = ADMM_MGL(S, l1, l2,  reg = 'FGL', Omega_0=Omega_0, rho = 1, max_iter = 1000, \
                                                        tol = 1e-5, rtol = 1e-5, verbose = False, measure = True)
        
        
        result=  sol['Theta']

    return result


def main(args):
    print("Generating training data")
    T = generate_lasso(args)
    save= {'len_interval': args.len_interval, 'Theta': T}
    np.save(os.path.join(args.output_dir, "%s.npy" % args.mode), save, allow_pickle=True)
    print("Tau")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/CA1_Food1", help="Output directory."
    )
    parser.add_argument(
        "--df_filename",
        type=str,
        default="data/CA1_Food1.csv",
        help="Raw data readings.",
    )
    parser.add_argument(
        "--mode", type=str, default= 'static', help="unique or time-varying graphical lasso."
    )
    parser.add_argument(
        "--cov_modify", type=bool, default= True, help="add epsilon to diagonal of covariance matrix."
    )
    
    parser.add_argument(
        "--Lambda", type=float, default= 0.02, help="Lambda coefficent Lasso norm."
    )
    parser.add_argument(
        "--Beta", type=float, default= 0.05, help="Beta coefficent time-varying constraint."
    )
    parser.add_argument(
        "--len_interval", type=int, default= 180, help="Length intervals in time-varying cases."
    )
    parser.add_argument(
        "--scaling", type=bool, default= True, help="Scaling for equaly treating."
    )
    
    args = parser.parse_args()
    main(args)


