# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:55:09 2022

@author: lidon
"""

# perform multiple tests on Stochastic Gibbs sampling

import numpy as np
import HMM
import SeqSampling as Sampling
import time
from multiprocessing import Pool
import scipy.stats as stats
import math
import os
import multiprocessing as mp
from EMHMM import*
#from ZMARGibbs import*
from SGibbs import*
import matplotlib.pyplot as plt

 
transition=np.array([[0.7,0.2,0.1],[0.1,0.8,0.1],[0.1,0.3,0.6]])
#transition=np.array([[0.9,0.05,0.05],[0.02,0.95,0.03],[0.01,0.04,0.95]])
state=np.array(['0','1','2'])
hidden_state=state
obs_state=np.array(['Blue','Green','Yellow'])
obs_prob=np.array([[0.7,0.2,0.1],
                   [0.1,0.7,0.2],
                   [0.2,0.1,0.7]
    ])
pi=np.array([0.7,0.2,0.1])



if __name__=='__main__':
    # Use multicore CPU
    p=Pool(16)
    rate=0
    size=10000
    long=20
    A,B,pi0,data,I,hidden_data=initialize(hidden_state,obs_state,transition,obs_prob,pi,rate,size,long)
    sub_data=data[0:3000]
    
    sub_I=I[0:3000] 
    
    
    # Test Gibbs sampler using different batch size/different sampler
    ###################################################################################################
    post_A,post_B,post_pi,I=random_batched_Gibbs(data,I,A,B,pi0,25000,250,hidden_state,obs_state,p)
    # post_A,post_B,post_pi,I=batched_Gibbs(data,I,A,B,pi0,10000,250,hidden_state,obs_state,p)
    # post_A,post_B,post_pi,I=batched_Gibbs(data,I,A,B,pi0,8000,200,hidden_state,obs_state,p)
    # post_A,post_B,post_pi,I=batched_Gibbs(data,I,A,B,pi0,8000,80,hidden_state,obs_state,p)
    #post_A,post_B,post_pi,I=batched_Gibbs(data,I,A,B,pi0,12000,16,hidden_state,obs_state,p)
    ##################################################################################################
    
    
    # Test our model against standard Gibbs w.r.t. the running time
    ###################################################################################################
    # start2=time.time()
    # post_A1,post_B1,post_pi1,I1=parallel_Gibbs(data, I, A, B, pi0,2500, hidden_state, obs_state, p)
    # end2=time.time()
    # period2=end2-start2
    # start1=time.time()
    # post_A,post_B,post_pi,I=batched_Gibbs(data,I,A,B,pi0,10000,250,hidden_state,obs_state,p)
    # end1=time.time()
    # period1=end1-start1
    ##################################################################################################
    
    
    
    # test novel batched gibbs
    ####################################################################################################
    #post_A,post_B,post_pi,I=novel_batched_Gibbs(data,I,A,B,pi0,10000,250,hidden_state,obs_state,p)
    #post_A,post_B,post_pi,I=Naive_Minibatch_Gibbs(data,I,A,B,pi0,10000,80,hidden_state,obs_state,p)
    #post_A,post_B,post_pi,I=batched_Gibbs(data,I,A,B,pi0,10000,250,hidden_state,obs_state,p)
    ###################################################################################################
    
    
    
    # paint the graphs
    ####################################################################################################
    post_pi=np.array(post_pi)
    post_A=np.array(post_A)
    post_B=np.array(post_B)
    post_pi,post_A,post_B=permute(post_pi,post_A,post_B,pi,transition,obs_prob)
    
    
    k=1
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            plt.subplot(A.shape[0],A.shape[1],k)
            plt.plot(post_A[:,i,j])
            plt.xlabel(f'A{i+1}{j+1}')
            plt.axhline(y=transition[i,j],c='red')
            
            k+=1
    #plt.savefig('Bootstrapp_A')
    plt.legend(loc='best')
    #plt.close('all')


    k=1
    for i in range(0,B.shape[0]):
        for j in range(0,B.shape[1]):
            plt.subplot(B.shape[0],B.shape[1],k)
            plt.plot(post_B[:,i,j])
            plt.xlabel(f'B{i+1}{j+1}')
            plt.axhline(y=obs_prob[i,j],c='red')
            k+=1
    plt.savefig('Bootstrap_B')
    plt.legend(loc='best')
    #plt.close('all')


    k=1
    for i in range(0,pi.shape[0]):
        plt.subplot(1,pi.shape[0],k)
        plt.plot(post_pi[:,i])
        plt.xlabel(f'pi{i+1}')
        plt.axhline(y=pi[i],c='red')
        k+=1
    plt.savefig('Bootstrap_pi')
    plt.legend(loc='best')
    #plt.close('all')
    ##################################################################################################
    
    
    