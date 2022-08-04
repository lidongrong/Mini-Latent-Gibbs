# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:11:56 2022

@author: lidon
"""

import numpy as np
import scipy.stats as stats
import time

# generate data for probit regression
# assume beta ~ N(0,I)
# X sampled from uniform distribution on (-1,1) for element
# size: sample size
def data_generator(beta,size):
    # dimension
    dim=beta.shape[0]
    x=[np.random.uniform(-1,1,dim) for i in range(0,size)]
    x=np.array(x)
    # generate latent variable
    z=stats.multivariate_normal.rvs(np.dot(x,beta),np.eye(x.shape[0]),1)
    for i in range(0,len(z)):
        z[i]=np.random.normal(0,1,1)+np.dot(x[i],beta)
    
    y=z>0
    
    return x,y,z

# Gibbs sampler part
# initialize parameters
# construct initial guess of beta and z
def initialize(x,y):
    dim=x.shape[1]
    # initial guess of beta
    beta0=np.random.normal(0,1,dim)
    # initialize z
    z0=np.zeros(x.shape[0])
    for i in range(0,x.shape[0]):
        if y[i]==True:
            z0[i]=stats.truncnorm.rvs(a=0-np.dot(x[i],beta0),b=np.inf-np.dot(x[i],beta0),loc=np.dot(x[i],beta0),scale=1,size=1)[0]
        elif y[i]==False:
            z0[i]=stats.truncnorm.rvs(a=-np.inf-np.dot(x[i],beta0),b=0-np.dot(x[i],beta0),loc=np.dot(x[i],beta0),scale=1,size=1)[0]
    
    return beta0,z0

# sample beta
def sample_beta(x,y,z,beta):
    dim=beta.shape[0]
    #new_beta=np.zeros(dim)
    # specify normal mean and covariance
    cov=np.linalg.inv(np.dot(x.T,x)+np.eye(dim))
    mean=np.dot(cov,np.dot(x.T,z))
    
    new_beta=stats.multivariate_normal.rvs(mean,cov,1)
    
    #new_beta=beta
    return new_beta

# used in scalable Gibbs computing
# sample beta with an mh step
def mh_sample_beta(x,y,z,beta,global_beta):
    dim=beta.shape[0]
    new_beta=np.random.normal(0,0.4,len(beta))+beta
    cov=np.linalg.inv(np.dot(x.T,x)+np.eye(dim))
    mean=np.dot(cov,np.dot(x.T,z))
    #new_beta=stats.multivariate_normal.rvs(mean,cov,1)
    ratio=0
    #ratio=stats.multivariate_normal.logpdf(new_beta,mean,cov)-stats.multivariate_normal.logpdf(beta,mean,cov)
    #ratio=ratio+stats.multivariate_normal.logpdf(new_beta,global_beta,4*np.eye(len(y)))-stats.multivariate_normal.logpdf(beta,global_beta,4*np.eye(len(y)))
    #ratio=sum(stats.norm.logpdf(new_beta,global_beta,4)-stats.norm.logpdf(beta,global_beta,4))
    ratio=ratio+stats.multivariate_normal.logpdf(z,np.dot(x,new_beta),np.eye(len(z)))-stats.multivariate_normal.logpdf(z,np.dot(x,beta),np.eye(len(z)))
    ratio=ratio+sum(stats.norm.logpdf(new_beta,global_beta,4)-stats.norm.logpdf(beta,global_beta,4))
    # ratio=ratio+stats.multivariate_normal.logpdf(beta,mean,cov)-stats.multivariate_normal.logpdf(new_beta,mean,cov)
    ratio=min(ratio,0)
    u=np.random.uniform(0,1,1)[0]
    u=np.log(u)
    if u<=ratio:
        return new_beta
    else:
        return beta



# sample latent variables
def sample_z(x,y,z,beta):
    new_z=z.copy()
    for i in range(0,len(new_z)):
        mean=np.dot(x[i],beta)
        if y[i]==True:
            ratio=0
            prop_z=stats.gamma.rvs(a=z[i],size=1)[0]
            #prop_z=np.random.exponential(4,1)[0]
            #ratio=stats.expon.logpdf(z[i],0,1)-stats.expon.logpdf(prop_z,0,1)
            ratio=stats.gamma.logpdf(z[i],a=prop_z)-stats.gamma.logpdf(prop_z,a=z[i])
            ratio=ratio+stats.norm.logpdf(prop_z,mean,1)-stats.norm.logpdf(z[i],mean,1)
            u=np.random.uniform(0,1,1)[0]
            u=np.log(u)
            #ratio=ratio[0]
            ratio=min(ratio,0)
            if u<=ratio:
                new_z[i]=prop_z
            
        elif y[i]==False:
            ratio=0
            #prop_z=np.random.exponential(4,1)[0]
            prop_z=stats.gamma.rvs(a=-z[i],size=1)[0]
            stats.gamma.rvs(a=-z[i],size=1)[0]
            ratio=stats.gamma.logpdf(-z[i],a=prop_z)-stats.gamma.logpdf(prop_z,a=-z[i])
            ratio=ratio+stats.norm.logpdf(-prop_z,mean,1)-stats.norm.logpdf(z[i],mean,1)
            u=np.random.uniform(0,1,1)[0]
            u=np.log(u)
            #ratio=ratio[0]
            ratio=min(ratio,0)
            if u<=ratio:
                new_z[i]=-1*prop_z
            
    return new_z

def sample_global_param(x,y,z,global_beta,beta_copy):
    global_beta=sum(beta_copy)/len(beta_copy)
    return global_beta

# generate copies of theta
def copy_generator(x,y,z,beta,batch_size):
    batch_num=len(y)/batch_size
    batch_num=int(batch_num)
    beta_copy=[]
    for i in range(0,batch_num):
        beta_copy.append(np.random.normal(0,4,len(beta))+beta)
    beta_copy=np.array(beta_copy)
    return beta_copy

# Gibbs sampler
# n is the sample size
def probit_Gibbs(x,y,z,beta,n):
    post_beta=[]
    for i in range(0,n):
        print('iteration: ',i)
        beta=sample_beta(x,y,z,beta)
        z=sample_z(x,y,z,beta)
        
        print('beta: ',beta)
        post_beta.append(beta)
    return post_beta,z


# n: number of iteration
def batch_probit_Gibbs(x,y,z,beta,batch_size,n):
    batch_num=int(len(y)/batch_size)
    post_beta=[]
    beta_copy=copy_generator(x,y,z,beta,batch_size)
    prob=np.array([1/3,1/3,1/3])
    for i in range(0,n):
        start=time.time()
        print(f'iteration: {i}')
        group=np.random.choice([0,1,2],1,True,prob)[0]
        # update theta
        if group==0:
            beta=sample_global_param(x,y,z,beta,beta_copy)
            print(f'beta: {beta}')
            
            post_beta.append(beta)
        # update thetas
        if group==1:
            tau=np.random.choice(np.arange(batch_num),1,True)[0]
            index=np.arange(tau*batch_size,(tau+1)*batch_size)
            #index=np.random.choice(np.arange(len(y)),batch_size,False)
            #y_batch=y[tau*batch_size:(tau+1)*batch_size].copy()
            #z_batch=z[tau*batch_size:(tau+1)*batch_size].copy()
            x_batch=x[index].copy()
            y_batch=y[index].copy()
            z_batch=z[index].copy()
            beta_copy[tau]=mh_sample_beta(x_batch,y_batch,z_batch,beta_copy[tau],beta)
            #theta_copy[tau]=mh_sample_theta(y_batch,z_batch,K,u_copy[tau],theta_copy[tau],batch_num,theta)
            
        # update z
        if group==2:
            tau=np.random.choice(np.arange(batch_num),1,True)[0]
            index=np.arange(tau*batch_size,(tau+1)*batch_size)
            #index=np.random.choice(np.arange(len(y)),batch_size,False)
            #y_batch=y[tau*batch_size:(tau+1)*batch_size].copy()
            #z_batch=z[tau*batch_size:(tau+1)*batch_size].copy()
            x_batch=x[index].copy()
            y_batch=y[index].copy()
            z_batch=z[index].copy()
            z_batch=sample_z(x_batch,y_batch,z_batch,beta_copy[tau])
            #z[tau*batch_size:(tau+1)*batch_size]=z_batch
            z[index]=z_batch
        end=time.time()
        #print('time used: ',end-start)
    return beta,beta_copy,z