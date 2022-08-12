# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:48:03 2022

@author: lidon
"""

import numpy as np
import scipy.stats as stats
from probit import*
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

beta=np.array([-2,1,3])
size=2000
batch_size=125
n=60000
rho=0.08

if __name__ == '__main__':
    x,y,z=data_generator(beta,size)
    beta0,z0=initialize(x,y)
    #post_beta,post_z=probit_Gibbs(x,y,z0,beta0,2000)
    #post_beta,post_copy,post_z=batch_probit_Gibbs(x,y,z0,beta0,batch_size,n)
    our_beta,our_copy,our_z,tau=minibatch_probit_Gibbs(x, y, z0, beta0, batch_size, rho, n)
    
    # true
    dim=beta.shape[0]
    cov=np.linalg.inv(np.dot(x.T,x)+np.eye(dim))
    est=np.dot(cov,np.dot(x.T,z))
