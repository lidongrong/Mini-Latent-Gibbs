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


beta=np.array([-2,1,4])
#size=10000
batch_size=125
n=60000

if __name__ == '__main__':
    
    # y,z=GMM_generator(K,u,theta,n)
    # theta0,z0,u0=GMM_initializer(y,K,u,theta)
    # #post_u,post_theta,post_z=GMM_Gibbs(y,z0,K,u0,theta0,2000)
    # post_u,post_theta,post_z=batched_GMM_Gibbs(y,z0,K,u0,theta0,20000,100)
    # post_u=np.array(post_u)
    # post_theta=np.array(post_theta)
    # post_u,post_theta=permute(post_u,post_theta,u,theta)
    
    # generate data with each size
    size_set=[1000,2500,5000,7500,10000]
    # number of test wrt each size
    num=12
    t=datetime.now()
    time_list=[t.year,t.month,t.day,t.hour,t.minute,t.second]
    time_list=[str(x) for x in time_list]
    time_list='_'.join(time_list)
    folder_name=f'BatchProbit{time_list}'
    os.mkdir(folder_name)
    
    # calculate the mae
    error=np.zeros((len(size_set),num))
    our_error=np.zeros((len(size_set),num))
    # compare variance
    variance=np.zeros((len(size_set),num))
    our_variance=np.zeros((len(size_set),num))
    # compare time
    timing=np.zeros((len(size_set),num))
    our_timing=np.zeros((len(size_set),num))
    
    covar=[]
    
    for i in range(0,len(size_set)):
        size=size_set[i]
        x,y,z=data_generator(beta,size)
        beta0,z0=initialize(x,y)
        cov=np.linalg.inv(np.dot(x.T,x)+np.eye(len(beta)))
        covar.append(np.trace(cov)/len(beta))
        for j in range(0,num):
            print(f'size: {size}, experiment: {j}')
            start=time.time()
            post_beta,post_z=probit_Gibbs(x,y,z0,beta0,500)
            end=time.time()
            used_time=end-start
            used_time=used_time*4
            start=time.time()
            our_beta,our_copy,our_z=batch_probit_Gibbs(x,y,z0,beta0,batch_size+i*25,n)
            end=time.time()
            our_used_time=end-start
            
            # sample post processing
            #post_u,post_theta=permute(post_u,post_theta,u,theta)
            #our_u,our_theta=permute(our_u,our_theta,u,theta)
            # compute posterior var and mean
            pu=post_beta[int(0.8*len(post_beta)):]
            ou=our_beta[int(0.8*len(our_beta)):]
            #posterior mean
            p_mean=sum(pu)/len(pu)
            o_mean=sum(ou)/len(ou)
            # posterior variance
            p_var=sum(np.var(pu,axis=0))
            o_var=sum(np.var(ou,axis=0))
            
            # fill in the data
            error[i,j]=np.sum(abs(p_mean-beta))
            our_error[i,j]=np.sum(abs(o_mean-beta))
            variance[i,j]=p_var
            our_variance[i,j]=o_var
            timing[i,j]=used_time
            our_timing[i,j]=our_used_time
    
    # save data
    np.save(f'{folder_name}/err.npy',error)
    np.save(f'{folder_name}/our_err.npy',our_error)
    np.save(f'{folder_name}/variance.npy',variance)
    np.save(f'{folder_name}/our_variance.npy',our_variance)
    np.save(f'{folder_name}/time.npy',timing)
    np.save(f'{folder_name}/our_time.npy',our_timing)
    
    # plot error
    plt.plot(np.mean(error,axis=1),'*-',label='Error of standard Gibbs')
    plt.plot(np.mean(our_error,axis=1),'*-',label='Error of our method')
    plt.xticks(np.arange(len(size_set)),size_set)
    plt.xlabel('Data Size')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(f'{folder_name}/Error Comparison')
    plt.close('all')
    
    # plot variance
    covar=np.array(covar)
    plt.plot(covar,'*-',label='Theoretical Variance')
    plt.plot(np.mean(our_variance,axis=1),'*-',label='Variance of our method')
    plt.xticks(np.arange(len(size_set)),size_set)
    plt.xlabel('Data Size')
    plt.ylabel('Variance')
    plt.legend()
    plt.savefig(f'{folder_name}/Variance Comparison')
    plt.close('all')
    
    # plot time
    plt.plot(np.mean(timing,axis=1),'*-',label='Consumed time of standard Gibbs')
    plt.plot(np.mean(our_timing,axis=1),'*-',label='Consumed time of our method')
    plt.xticks(np.arange(len(size_set)),size_set)
    plt.xlabel('Data Size')
    plt.ylabel('Average Used Time')
    plt.legend()
    plt.savefig(f'{folder_name}/Time Comparison')
    plt.close('all')
