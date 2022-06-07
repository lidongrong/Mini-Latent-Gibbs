# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:26:59 2022

@author: lidon
"""

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
    
    A=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
    B=np.array([[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4]])
    pi0=np.array([0.3,0.4,0.3])
    #post_A,post_B,post_pi,I=parallel_Gibbs(sub_data,sub_I,A,B,pi,2000,hidden_state,obs_state,p)
    #post_A1,post_B1,post_pi1=Minibatch_Gibbs(data,I,A,B,pi,20000,250,hidden_state,obs_state,p)
    #post_A,post_B,post_pi=Ordered_Minibatch_Gibbs(data, I, A, B, pi, 20000, 250, hidden_state, obs_state, p)
    #post_A,post_B,post_pi,I=batched_Gibbs(data,I,A,B,pi0,25000,250,hidden_state,obs_state,p)
    #post_A,post_B,post_pi,I=Naive_Minibatch_Gibbs(data,I,A,B,pi0,15000,250,hidden_state,obs_state,p)
    
    post_A1,post_B1,post_pi1,I1=batched_Gibbs(data,I,A,B,pi0,12500,250,hidden_state,obs_state,p)
    post_A2,post_B2,post_pi2,I2=Ordered_Minibatch_Gibbs(data,I,A,B,pi0,12500,250,hidden_state,obs_state,p)
    post_A3,post_B3,post_pi3,I3=Naive_Minibatch_Gibbs(data,I,A,B,pi0,12500,250,hidden_state,obs_state,p)
    