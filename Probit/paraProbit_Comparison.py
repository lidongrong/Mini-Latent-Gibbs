# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:49:55 2022

@author: s1155151972
"""

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
import tensorflow_probability as tfp
import tensorflow as tf
from multiprocessing import Pool


beta=np.array([-2,1,3])
#size=10000
batch_size=125
n=250000
# generate data with each size
size_set=[1000,2000,3000,4000,5000,8000]
rho_set=[0.3,0.25,0.2,0.15,0.1]
# number of test wrt each size
num=16
ess=100

if __name__ == '__main__':
    # define multiprocessing pool
    pool=Pool(16)
    np.random.seed(1919810)
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
    start_z=np.array(start_z)
    
    # compute mean of error under different dataset
    total_gibbs_error=np.zeros(len(size_set))
    total_gibbs_variance=np.zeros(len(size_set))
    total_gibbs_time=np.zeros(len(size_set))
    total_MAP_error=np.zeros(len(size_set))
    total_theo_variance=np.zeros(len(size_set))
    # amount of time to produce an effective sample size equal to ess
    total_gibbs_ess_time=np.zeros(len(size_set))
    
    # compute mean of error under different dataset
    total_our_error=np.zeros((len(rho_set),len(size_set)))
    total_our_variance=np.zeros((len(rho_set),len(size_set)))
    total_our_time=np.zeros((len(rho_set),len(size_set)))
    # amount of time to produce an effective sample size equal to ess
    total_our_ess_time=np.zeros((len(rho_set),len(size_set)))
    
    
    
    for i in range(0,len(size_set)):
        size=size_set[i]
        our_error=np.zeros((len(rho_set),num))
        gibbs_error=np.zeros(num)
        
        # amount of time to produce an effective sample size equal to ess
        our_ess_time=np.zeros((len(rho_set),num))
        gibbs_ess_time=np.zeros(num)
        
        our_variance=np.zeros((len(rho_set),num))
        gibbs_variance=np.zeros(num)
        #Theoretical Variance
        theo_variance=np.zeros(num)
        
        our_time=np.zeros((len(rho_set),num))
        gibbs_time=np.zeros(num)
        MAP=np.zeros(num)
        #covar=[]
        os.mkdir(f'{folder_name}/Data{size}')
        data_folder_name=f'{folder_name}/Data{size}'
        
        
        # run num of experiments at the same time
        
        # define starting points
        x=full_x[0:size]
        y=full_y[0:size]
        z=full_z[0:size]
        z0=np.array([start_z[j][0:size] for j in range(0,num)])
        beta0=np.array([beta_initial[j] for j in range(0,num)])
        
        print(datetime.now())
        print(f'size: {size}')
        # gibbs_result is a list of sets, each set consists of (post_beta,post_z,gibbs_slot)
        gibbs_result=pool.starmap(random_probit_Gibbs,[(x,y,z0[k],beta0[k],5000) for k in range(0,num)])
        gibbs_result=np.array(gibbs_result)
        
        # process gibbs output
        for j in range(0,num):
            post_beta=gibbs_result[j][0]
            post_z=gibbs_result[j][1]
            gibbs_slot=gibbs_result[j][2]
            
            # convert them to np array
            post_beta=np.array(post_beta)
            post_z=np.array(post_z)
            gibbs_slot=np.array(gibbs_slot)
            
            gibbs_time[j]=gibbs_slot[len(gibbs_slot)-1]-gibbs_slot[0]
            
            # evaluate error vs time
            acc=np.sum(abs(post_beta-beta),axis=1)
            plt.plot(gibbs_slot,acc,label='Standard Gibbs')
            plt.xlabel('Time (Seconds)')
            plt.ylabel('Mean Absolute Error')
            plt.legend()
            plt.savefig(f'{data_folder_name}/Size{size}Experiment{j}')
            plt.close('all')
            
            # evaluate ess
            after_burn=tf.convert_to_tensor(post_beta[500:])
            ess_low_bound=tfp.mcmc.effective_sample_size(after_burn,filter_threshold=0.1)
            gibbs_ess_time[j]=(gibbs_time[j])*(ess/max(ess_low_bound))
            
            pu=post_beta[500:]
            pu=pu[-min(len(pu),len(pu)*(ess/max(ess_low_bound))):]
            p_mean=sum(pu)/len(pu)
            p_var=sum(np.var(pu,axis=0))
            
            gibbs_error[j]=np.sum(abs(p_mean-beta))
            gibbs_variance[j]=p_var
            #theo_variance[j]=np.trace(cov)
        
        # run num of our sample at the same time under same rho
        for k in range(0,len(rho_set)):
            rho=rho_set[k]
            print(datetime.now())
            print(f'size: {size},  rho: {rho_set[k]}')
            rho_result=pool.starmap(randomized_minibatch_probit_Gibbs,[(x,y,z0[r],beta0[r],batch_size,rho,n) for r in range(0,num)])
            rho_result=np.array(rho_result)
            
            # process results one by one
            for j in range(0,num):
                our_beta=rho_result[j][0]
                our_z=rho_result[j][2]
                our_slot=rho_result[j][4]
                
                # convert them to np array
                our_beta=np.array(our_beta)
                our_z=np.array(our_z)
                our_slot=np.array(our_slot)
                
                # record time
                our_time[k,j]=our_slot[len(our_slot)-1]
                
                # evaluate error vs time
                acc=np.sum(abs(our_beta-beta),axis=1)
                plt.plot(our_slot,acc,label=f'Our Gibbs(rho={rho_set[k]})')
                plt.xlabel('Time (Seconds)')
                plt.ylabel('Mean Absolute Error')
                plt.legend()
                plt.savefig(f'{data_folder_name}/Size{size}Experiment{j}(rho={rho_set[k]}).jpg')
                plt.close('all')
                
                # evaluate ess
                after_burn=tf.convert_to_tensor(our_beta[10000:])
                ess_low_bound=tfp.mcmc.effective_sample_size(after_burn,filter_threshold=0.1)
                our_ess_time[k,j]=(our_time[k,j])*(ess/max(ess_low_bound))
                
               
                
                # calculate error and variance
                ou=our_beta[10000:]
                ou=ou[-min(len(ou),len(ou)*(ess/max(ess_low_bound))):]
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
        
        
        #plot ESS time
        plt.plot(gibbs_ess_time,'*-',label='Standard Gibbs')
        for s in range(0,len(rho_set)):
            plt.plot(our_ess_time[s,:],'*-',label=f'Our Sampler(rho={rho_set[s]})')
            total_our_ess_time[s,i]=sum(our_ess_time[s,:])/num
        # and ess of standard gibbs
        total_gibbs_ess_time[i]=sum(gibbs_ess_time)/len(gibbs_ess_time)
        # finish drawing
        plt.xticks(np.arange(num),np.arange(num))
        plt.xlabel('Experiment')
        plt.ylabel('ESS time')
        plt.legend()
        plt.savefig(f'{folder_name}/ESS Time Comparison_DataSize{size_set[i]}')
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
        
        # plot average ESS TIME rate under different data size
        # compare with standard gibbs & estimation of MAP under full data
        plt.plot(total_gibbs_ess_time,'*-',label='Standard Gibbs')
        #plt.plot(total_MAP_error,'*-',label='MAP with full data')
        for s in range(0,len(rho_set)):
            plt.plot(total_our_ess_time[s,:],'*-',label=f'Our Sampler(rho={rho_set[s]})')
        plt.xticks(np.arange(len(size_set)),size_set)
        plt.xlabel('Experiment')
        plt.ylabel('ESS Time')
        plt.legend()
        plt.savefig(f'{folder_name}/Average ESS time Comparison{size_set[i]}')
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
                    