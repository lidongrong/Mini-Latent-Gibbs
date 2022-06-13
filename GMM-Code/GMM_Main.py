# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:58:18 2022

@author: lidon
"""

import numpy as np
import scipy.stats as stats
from GMM import*
import matplotlib.pyplot as plt


u=np.array([8,4,0,-4])
theta=np.array([0.1,0.2,0.3,0.4])
K=len(u)

if __name__ == '__main__':
    
    # y,z=GMM_generator(K,u,theta,n)
    # theta0,z0,u0=GMM_initializer(y,K,u,theta)
    # #post_u,post_theta,post_z=GMM_Gibbs(y,z0,K,u0,theta0,2000)
    # post_u,post_theta,post_z=batched_GMM_Gibbs(y,z0,K,u0,theta0,20000,100)
    # post_u=np.array(post_u)
    # post_theta=np.array(post_theta)
    # post_u,post_theta=permute(post_u,post_theta,u,theta)
    
    num=4
    for i in range(0,num):
        n=10000
        y,z=GMM_generator(K,u,theta,n)
        theta0,z0,u0=GMM_initializer(y,K,u,theta)
        post_u,post_theta,post_z=batched_GMM_Gibbs(y,z0,K,u0,theta0,25000,100)
        post_u=np.array(post_u)
        post_theta=np.array(post_theta)
        post_u,post_theta=permute(post_u,post_theta,u,theta)
        
        for k in range(0,u.shape[0]):
            plt.subplot(2,2,k+1)
            plt.plot(post_u[:,k])
            plt.axhline(y=u[k],c='red')
        plt.savefig(f'u{i}')
        plt.close('all')
            
        for k in range(0,theta.shape[0]):
            plt.subplot(2,2,k+1)
            plt.plot(post_theta[:,k])
            plt.axhline(y=theta[k],c='red')
        plt.savefig(f'theta{i}')
        plt.close('all')
        