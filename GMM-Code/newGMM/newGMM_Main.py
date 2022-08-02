# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:58:18 2022

@author: lidon
"""

import numpy as np
import scipy.stats as stats
from newGMM import*
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime


u=np.array([4,0,-6])
theta=np.array([0.5,0.25,0.25])
K=len(u)

if __name__ == '__main__':
    
    # y,z=GMM_generator(K,u,theta,n)
    # theta0,z0,u0=GMM_initializer(y,K,u,theta)
    # #post_u,post_theta,post_z=GMM_Gibbs(y,z0,K,u0,theta0,2000)
    # post_u,post_theta,post_z=batched_GMM_Gibbs(y,z0,K,u0,theta0,20000,100)
    # post_u=np.array(post_u)
    # post_theta=np.array(post_theta)
    # post_u,post_theta=permute(post_u,post_theta,u,theta)
    
    num=8
    t=datetime.now()
    time_list=[t.year,t.month,t.day,t.hour,t.minute,t.second]
    time_list=[str(x) for x in time_list]
    time_list='_'.join(time_list)
    folder_name=f'BatchGMM{time_list}'
    os.mkdir(folder_name)
    for i in range(0,num):
        size=1000
        y,z=GMM_generator(K,u,theta,size)
        theta0,z0,u0=GMM_initializer(y,K,u,theta)
        theta0=theta
        #post_u,post_theta,post_z=batched_GMM_Gibbs(y,z0,K,u0,theta0,25000,50)
        #post_u,post_theta,post_z=GMM_Gibbs(y,z0,K,u0,theta0,1000)
        post_u,post_theta,post_z=novel_batched_GMM_Gibbs(y,z0,K,u0,theta0,50000,50)
        post_u=np.array(post_u)
        post_theta=np.array(post_theta)
        post_u,post_theta=permute(post_u,post_theta,u,theta)
        
        for k in range(0,u.shape[0]):
            plt.subplot(2,2,k+1)
            plt.plot(post_u[:,k])
            plt.axhline(y=u[k],c='red')
            plt.axhline(y=sum(post_u[10000:,k])/len(post_u[10000:]),c='blue')
        plt.savefig(f'{folder_name}/u{i}')
        plt.close('all')
            
        for k in range(0,theta.shape[0]):
            plt.subplot(2,2,k+1)
            plt.plot(post_theta[:,k])
            plt.axhline(y=theta[k],c='red')
            plt.axhline(y=sum(post_theta[10000:,k])/len(post_theta[10000:]),c='blue')
        plt.savefig(f'{folder_name}/theta{i}')
        plt.close('all')
        