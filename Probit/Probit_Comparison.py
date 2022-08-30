# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 17:02:49 2022

@author: lidon
"""
# compare our sampler against traditional Gibbs Sampler

import numpy as np
import scipy.stats as stats
from probit import*
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime


beta=np.array([-2,1,3])
#size=10000
batch_size=125
n=60000
# generate data with each size
size_set=[1000,2000,3000,4000,5000,8000]
rho_set=[1,0.5,0.2,0.1]
# number of test wrt each size
num=16

if __name__ == '__main__':
    
    # y,z=GMM_generator(K,u,theta,n)
    # theta0,z0,u0=GMM_initializer(y,K,u,theta)
    # #post_u,post_theta,post_z=GMM_Gibbs(y,z0,K,u0,theta0,2000)
    # post_u,post_theta,post_z=batched_GMM_Gibbs(y,z0,K,u0,theta0,20000,100)
    # post_u=np.array(post_u)
    # post_theta=np.array(post_theta)
    # post_u,post_theta=permute(post_u,post_theta,u,theta)
    
    
    t=datetime.now()
    time_list=[t.year,t.month,t.day,t.hour,t.minute,t.second]
    time_list=[str(x) for x in time_list]
    time_list='_'.join(time_list)
    folder_name=f'BatchProbit{time_list}'
    os.mkdir(folder_name)
    
    # generate initial data points
    dim=len(beta)
    beta_initial=[np.random.normal(0,1,dim) for i in range(0,num)]
    beta_initial=np.array(beta_initial)
    full_x,full_y,full_z=data_generator(beta,size_set[len(size_set)-1])
    start_z=[sample_z(full_x,full_y,full_z,beta_initial[i]) for i in range(0,len(beta_initial))]
    
    # compute mean of error under different dataset
    total_gibbs_error=np.zeros(len(size_set))
    total_gibbs_variance=np.zeros(len(size_set))
    total_gibbs_time=np.zeros(len(size_set))
    total_MAP_error=np.zeros(len(size_set))
    total_theo_variance=np.zeros(len(size_set))
    
    # compute mean of error under different dataset
    total_our_error=np.zeros((len(rho_set),len(size_set)))
    total_our_variance=np.zeros((len(rho_set),len(size_set)))
    total_our_time=np.zeros((len(rho_set),len(size_set)))
    
    for i in range(0,len(size_set)):
        size=size_set[i]
        our_error=np.zeros((len(rho_set),num))
        gibbs_error=np.zeros(num)
        
        our_variance=np.zeros((len(rho_set),num))
        gibbs_variance=np.zeros(num)
        #Theoretical Variance
        theo_variance=np.zeros(num)
        
        our_time=np.zeros((len(rho_set),num))
        gibbs_time=np.zeros(num)
        MAP=np.zeros(num)
        #covar=[]
        
        for j in range(0,num):
            print(f'size: {size}, experiment: {j}')
            
            # generate data
            size=size_set[i]
            
            
            x=full_x[0:size]
            y=full_y[0:size]
            z=full_z[0:size]
            beta0=beta_initial[j]
            z0=start_z[j][0:size]
            
            dim=beta.shape[0]
            cov=np.linalg.inv(np.dot(x.T,x)+np.eye(dim))
            est=np.dot(cov,np.dot(x.T,z))
            
            # error of MAP
            MAP[j]=sum(abs(est-beta))
            # first,gibbs
            start=time.time()
            post_beta,post_z=probit_Gibbs(x,y,z0,beta0,2500)
            end=time.time()
            gibbs_time[j]=end-start
            
            pu=post_beta[-int(0.8*len(post_beta)):]
            p_mean=sum(pu)/len(pu)
            p_var=sum(np.var(pu,axis=0))
            
            gibbs_error[j]=np.sum(abs(p_mean-beta))
            gibbs_variance[j]=p_var
            theo_variance[j]=np.trace(cov)
            for k in range(0,len(rho_set)):
                rho=rho_set[k]
                print(f'size: {size}, experiment: {j}, rho: {rho_set[k]}')
                start=time.time()
                our_beta,our_copy,our_z,tau=randomized_minibatch_probit_Gibbs(x, y, z0, beta0, batch_size, rho, n)
                end=time.time()
                
                # record time
                our_time[k,j]=end-start
                
                # calculate error and variance
                ou=our_beta[-len(pu):]
                o_mean=sum(ou)/len(ou)
                o_var=sum(np.var(ou,axis=0))
                
                # record error and variance
                our_error[k,j]=np.sum(abs(o_mean-beta))
                our_variance[k,j]=o_var
        
        # plot error
        plt.plot(gibbs_error,'*-',label='Standard Gibbs')
        plt.plot(MAP,'*-',label='MAP on full data')
        for s in range(0,len(rho_set)):
            plt.plot(our_error[s,:],'*-',label=f'Our Sampler(rho={rho_set[s]})')
            total_our_error[s,i]=sum(our_error[s,:])/num
            
        # add error into total_ array
        total_gibbs_error[i]=sum(gibbs_error)/len(gibbs_error)
        total_MAP_error[i]=sum(MAP)/len(MAP)
        
        plt.xticks(np.arange(num),np.arange(num))
        plt.xlabel('Experiment')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig(f'{folder_name}/Error Comparison_DataSize{size_set[i]}')
        plt.close('all')
        
        
        
        # plot variance
        plt.plot(gibbs_variance,'*-',label='Standard Gibbs')
        plt.plot(theo_variance,'*-',label='Theoretical Variance')
        for s in range(0,len(rho_set)):
            plt.plot(our_variance[s,:],'*-',label=f'Our Sampler(rho={rho_set[s]})')
            total_our_variance[s,i]=sum(our_variance[s,:])/num
        
        total_gibbs_variance[i]=sum(gibbs_variance)/len(gibbs_variance)
        total_theo_variance[i]=sum(theo_variance)/len(theo_variance)
        
        plt.xticks(np.arange(num),np.arange(num))
        plt.xlabel('Experiment')
        plt.ylabel('Variance')
        plt.legend()
        plt.savefig(f'{folder_name}/Variance Comparison_DataSize{size_set[i]}')
        plt.close('all')
        
        # plot time
        plt.plot(gibbs_time,'*-',label='Standard Gibbs')
        #plt.plot(theo_variance,'*-',label='Theoretical Variance')
        for s in range(0,len(rho_set)):
            plt.plot(our_time[s,:],'*-',label=f'Our Sampler(rho={rho_set[s]})')
            total_our_time[s,i]=sum(our_time[s,:])/num
        total_gibbs_time[i]=sum(gibbs_time)/len(gibbs_time)
        
        
        plt.xticks(np.arange(num),np.arange(num))
        plt.xlabel('Experiment')
        plt.ylabel('Used time (second)')
        plt.legend()
        plt.savefig(f'{folder_name}/Time Comparison_DataSize{size_set[i]}')
        plt.close('all')

        # plot average error rate under different data size
        # compare with standard gibbs & estimation of MAP under full data
        plt.plot(total_gibbs_error,'*-',label='Standard Gibbs')
        plt.plot(total_MAP_error,'*-',label='MAP with full data')
        for s in range(0,len(rho_set)):
            plt.plot(total_our_error[s,:],'*-',label=f'Our Sampler(rho={rho_set[s]})')
        plt.xticks(np.arange(len(size_set)),size_set)
        plt.xlabel('Experiment')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig(f'{folder_name}/Average Error Comparison{size_set[i]}')
        plt.close('all')
        
        # plot average variance under different data size
        # compare with standard gibbs & estimation of MAP under full data
        plt.plot(total_gibbs_variance,'*-',label='Standard Gibbs')
        plt.plot(total_theo_variance,'*-',label='Theoretical Var (with full data)')
        for s in range(0,len(rho_set)):
            plt.plot(total_our_variance[s,:],'*-',label=f'Our Sampler(rho={rho_set[s]})')
        plt.xticks(np.arange(len(size_set)),size_set)
        plt.xlabel('Experiment')
        plt.ylabel('Variance')
        plt.legend()
        plt.savefig(f'{folder_name}/Average Variance Comparison{size_set[i]}')
        plt.close('all')
        
        # plot average consumed time under different data size
        # compare with standard gibbs
        plt.plot(total_gibbs_time,'*-',label='Standard Gibbs')
        for s in range(0,len(rho_set)):
            plt.plot(total_our_time[s,:],'*-',label=f'Our Sampler(rho={rho_set[s]})')
        plt.xticks(np.arange(len(size_set)),size_set)
        plt.xlabel('Experiment')
        plt.ylabel('Consumed Time(Seconds)')
        plt.legend()
        plt.savefig(f'{folder_name}/Average Time Comparison{size_set[i]}')
        plt.close('all')
                    
            


    