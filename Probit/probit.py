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
    
    # perform likelhood evaluation
    # conjugate is not always the case
    evaluator(x,y,z,beta,beta,1)
    cov=np.linalg.inv(np.dot(x.T,x)+np.eye(dim))
    mean=np.dot(cov,np.dot(x.T,z))
    
    new_beta=stats.multivariate_normal.rvs(mean,cov,1)
    
    #new_beta=beta
    return new_beta

# evaluate joint distribution on a batch
# up to some normalizing constant
def evaluator(x,y,z,beta,global_beta,rho):
    dim=len(global_beta)
    output=stats.multivariate_normal.logpdf(z,np.dot(x,beta),np.eye(len(z)))
    output=output+stats.multivariate_normal.logpdf(beta,global_beta,(rho**2)*np.eye(dim))
    return output
    
    
    
    
# used in scalable Gibbs computing
# sample beta with an mh step
# rho: the divergence parameter
def mh_sample_beta(x,y,z,beta,global_beta,rho):
    dim=beta.shape[0]
    # mh step
    # new_beta=np.random.normal(0,0.2,len(beta))+beta
    # ratio=0
    # ratio=evaluator(x,y,z,new_beta,global_beta,rho)-evaluator(x,y,z,beta,global_beta,rho)
    # ratio=min(ratio,0)
    # u=np.random.uniform(0,1,1)[0]
    # u=np.log(u)
    # if u<=ratio:
    #     return new_beta
    # else:
    #     return beta
    
    # Gibbs step
    
    # perform likelihood evaluation
    evaluator(x,y,z,beta,global_beta,rho)
    
    cov=np.linalg.inv(np.dot(x.T,x)+np.eye(dim)/(rho**2))
    mean=np.dot(cov,global_beta/(rho**2)+np.dot(x.T,z))
    
    new_beta=stats.multivariate_normal.rvs(mean,cov,1)
    
    #new_beta=beta
    return new_beta



# sample latent variables
def sample_z(x,y,z,beta):
    new_z=z.copy()
    # the case where z is a single obs
    
    try:
        # handle normal case
        for i in range(0,len(new_z)):
            mean=np.dot(x[i],beta)
            if y[i]==True:
                lower=0
                upper=np.inf
                mu=mean
                sigma=1
                new_z[i]=stats.truncnorm.rvs(lower-mu,upper-mu,mu,sigma,size=1)[0]
                '''
                ratio=0
                prop_z=stats.gamma.rvs(a=z[i],size=1)[0]
                #prop_z=np.random.exponential(4,1)[0]
                #ratio=stats.expon.logpdf(z[i],0,1)-stats.expon.logpdf(prop_z,0,1)
                ratio=stats.gamma.logpdf(z[i],a=prop_z)-stats.gamma.logpdf(prop_z,a=z[i])
                ratio=ratio+stats.norm.logpdf(prop_z,mean,1)-stats.norm.logpdf(z[i],mean,1)
                u=np.random.uniform(0,1,1)[0]
                u=np.log(u)
                ratio=min(ratio,0)
                if u<=ratio:
                    new_z[i]=prop_z
                    '''
                
            elif y[i]==False:
                lower=-np.inf
                upper=0
                mu=mean
                sigma=1
                new_z[i]=stats.truncnorm.rvs(lower-mu,upper-mu,mu,sigma,size=1)[0]
                '''
                ratio=0
                prop_z=stats.gamma.rvs(a=-z[i],size=1)[0]
                stats.gamma.rvs(a=-z[i],size=1)[0]
                ratio=stats.gamma.logpdf(-z[i],a=prop_z)-stats.gamma.logpdf(prop_z,a=-z[i])
                ratio=ratio+stats.norm.logpdf(-prop_z,mean,1)-stats.norm.logpdf(z[i],mean,1)
                u=np.random.uniform(0,1,1)[0]
                u=np.log(u)
                ratio=min(ratio,0)
                if u<=ratio:
                    new_z[i]=-1*prop_z
                '''
                
        return new_z
    # incase z is a single observation
    except TypeError:
        mean=np.dot(x,beta)
        if y==True:
            lower=0
            upper=np.inf
            mu=mean
            sigma=1
            new_z=stats.truncnorm.rvs(lower-mu,upper-mu,mu,sigma,size=1)[0]
            '''
            ratio=0
            prop_z=stats.gamma.rvs(a=z[i],size=1)[0]
            #prop_z=np.random.exponential(4,1)[0]
            #ratio=stats.expon.logpdf(z[i],0,1)-stats.expon.logpdf(prop_z,0,1)
            ratio=stats.gamma.logpdf(z[i],a=prop_z)-stats.gamma.logpdf(prop_z,a=z[i])
            ratio=ratio+stats.norm.logpdf(prop_z,mean,1)-stats.norm.logpdf(z[i],mean,1)
            u=np.random.uniform(0,1,1)[0]
            u=np.log(u)
            ratio=min(ratio,0)
            if u<=ratio:
                new_z[i]=prop_z
                '''
            
        elif y==False:
            lower=-np.inf
            upper=0
            mu=mean
            sigma=1
            new_z=stats.truncnorm.rvs(lower-mu,upper-mu,mu,sigma,size=1)[0]
            '''
            ratio=0
            prop_z=stats.gamma.rvs(a=-z[i],size=1)[0]
            stats.gamma.rvs(a=-z[i],size=1)[0]
            ratio=stats.gamma.logpdf(-z[i],a=prop_z)-stats.gamma.logpdf(prop_z,a=-z[i])
            ratio=ratio+stats.norm.logpdf(-prop_z,mean,1)-stats.norm.logpdf(z[i],mean,1)
            u=np.random.uniform(0,1,1)[0]
            u=np.log(u)
            ratio=min(ratio,0)
            if u<=ratio:
                new_z[i]=-1*prop_z
            '''
        return new_z
        
        
    
    

# sample theta based on theta_1,..., theta_s
# rho is the divergence parameter
def sample_global_param(x,y,z,global_beta,beta_copy,rho):
    #global_beta=sum(beta_copy)/len(beta_copy)
    copy_num=len(beta_copy)
    dim=len(beta_copy[0])
    
    cov=np.linalg.inv(np.eye(dim)+(rho**2/copy_num)*np.eye(dim))
    mean=np.dot(cov,sum(beta_copy)/copy_num)
    scale=np.dot(cov,((rho**2)/copy_num)*np.eye(dim))
    
    #mean=sum(beta_copy)/(rho**2+copy_num)
    #scale=1/(copy_num/rho**2+1)*np.eye(len(global_beta))
    global_beta=stats.multivariate_normal.rvs(mean,scale,size=1)
    return global_beta

# generate copies of theta
# rho: divergence parameter
def copy_generator(x,y,z,beta,rho,batch_size):
    batch_num=len(y)/batch_size
    batch_num=int(batch_num)
    beta_copy=[]
    for i in range(0,batch_num):
        beta_copy.append(np.random.normal(0,rho,len(beta))+beta)
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

# Random Scan Gibbs Sampler
# every time sample theta|z or a mini-batch of z
def random_probit_Gibbs(x,y,z,beta,n):
    # batch_num=int(len(y)/batch_size)
    post_beta=[]
    for i in range(0,n):
        print('iteration: ',i)
        # decide sample theta or z
        toss=np.random.uniform(0,1,1)[0]
        # sample theta
        if toss>=0.5:
            beta=sample_beta(x,y,z,beta)
            print('beta: ',beta)
            post_beta.append(beta)
        if toss<0.5:
            z=sample_z(x,y,z,beta)
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

# n: number of iteration
# Use mini batch gibbs sampler proposed in 2022.8.12
# rho is the divergence parameter
def minibatch_probit_Gibbs(x,y,z,beta,batch_size,rho,n):
    batch_num=int(len(y)/batch_size)
    post_beta=[]
    beta_copy=copy_generator(x,y,z,beta,rho,batch_size)
    # weight for sampling theta,thetas,z and tau
    prob=np.array([1/4,1/4,1/4,1/4])
    # tau[s] decides the batch corresponding to theta_s
    tau=np.arange(batch_num)
    for i in range(0,n):
        start=time.time()
        print(f'iteration: {i}')
        # choose parameter to update
        group=np.random.choice([0,1,2,3],1,True,prob)[0]
        # update theta
        if group==0:
            beta=sample_global_param(x,y,z,beta,beta_copy,rho)
            print(f'beta: {beta}')
            
            post_beta.append(beta)
        # update thetas
        if group==1:
            #tau=np.random.choice(np.arange(batch_num),1,True)[0]
            # select the corresponding batch, s be the index
            # s is the index of parameter to update
            s=np.random.choice(np.arange(batch_num),1,True)[0]
            batch=tau[s]
            index=np.arange(batch*batch_size,(batch+1)*batch_size)
            #index=np.random.choice(np.arange(len(y)),batch_size,False)
            #y_batch=y[tau*batch_size:(tau+1)*batch_size].copy()
            #z_batch=z[tau*batch_size:(tau+1)*batch_size].copy()
            x_batch=x[index].copy()
            y_batch=y[index].copy()
            z_batch=z[index].copy()
            beta_copy[s]=mh_sample_beta(x_batch,y_batch,z_batch,beta_copy[s],beta,rho)
            #theta_copy[tau]=mh_sample_theta(y_batch,z_batch,K,u_copy[tau],theta_copy[tau],batch_num,theta)
            
        # update z
        if group==2:
            s=np.random.choice(np.arange(batch_num),1,True)[0]
            batch=tau[s]
            index=np.arange(batch*batch_size,(batch+1)*batch_size)
            #index=np.random.choice(np.arange(len(y)),batch_size,False)
            #y_batch=y[tau*batch_size:(tau+1)*batch_size].copy()
            #z_batch=z[tau*batch_size:(tau+1)*batch_size].copy()
            x_batch=x[index].copy()
            y_batch=y[index].copy()
            z_batch=z[index].copy()
            z_batch=sample_z(x_batch,y_batch,z_batch,beta_copy[s])
            #z[tau*batch_size:(tau+1)*batch_size]=z_batch
            z[index]=z_batch
        if group==3:
            # randomly switch two tau and decide whether to reject
            index=np.random.choice(np.arange(batch_num),2,False)
            new_tau=tau.copy()
            # s: index of tau
            s0=index[0]
            s1=index[1]
            batch0=tau[s0]
            batch1=tau[s1]
            index0=np.arange(batch0*batch_size,(batch0+1)*batch_size)
            index1=np.arange(batch1*batch_size,(batch1+1)*batch_size)
            ratio=evaluator(x[index0],y[index0],z[index0],beta_copy[s1],beta,rho)
            ratio=ratio+evaluator(x[index1],y[index1],z[index1],beta_copy[s0],beta,rho)
            ratio=ratio-evaluator(x[index0],y[index0],z[index0],beta_copy[s0],beta,rho)
            ratio=ratio-evaluator(x[index1],y[index1],z[index1],beta_copy[s1],beta,rho)
            
            new_tau[s0]=batch1
            new_tau[s1]=batch0
            ratio=min(ratio,0)
            u=np.random.uniform(0,1,1)[0]
            u=np.log(u)
            if u<=ratio:
                #print('tau switched!!!!!')
                tau=new_tau.copy()
            
        end=time.time()
        #print('time used: ',end-start)
    return post_beta,beta_copy,z,tau

# n: number of iteration
# Use mini batch gibbs sampler proposed in 2022.8.12
# rho is the divergence parameter
# permit enough randomization among batches
def randomized_minibatch_probit_Gibbs(x,y,z,beta,batch_size,rho,n):
    batch_num=int(len(y)/batch_size)
    post_beta=[]
    beta_copy=copy_generator(x,y,z,beta,rho,batch_size)
    # weight for sampling theta,thetas,z and tau
    prob=np.array([0.2,0.5,0.2,0.1])
    # tau[s] decides the parameter a sample belongs to
    # tau[0]=2, means sample 0 belongs to parameter 2
    tau=np.arange(len(y))
    for i in range(0,batch_num):
        tau[i*batch_size:(i+1)*batch_size]=i
    # randomly permute tau
    tau=np.random.permutation(tau)
    for i in range(0,n):
        start=time.time()
        print(f'iteration: {i}')
        # choose parameter to update
        group=np.random.choice([0,1,2,3],1,True,prob)[0]
        # update theta
        if group==0:
            beta=sample_global_param(x,y,z,beta,beta_copy,rho)
            print(f'beta: {beta}')
            
            post_beta.append(beta)
        # update thetas
        if group==1:
            #tau=np.random.choice(np.arange(batch_num),1,True)[0]
            # select the corresponding batch, s be the index
            # s is the index of parameter to update
            s=np.random.choice(np.arange(batch_num),1,True)[0]
            #batch=tau[s]
            #index=np.arange(batch*batch_size,(batch+1)*batch_size)
            # obtain the index of data that corresponds to parameter s
            index=np.where(tau==s)[0]
            #index=np.random.choice(np.arange(len(y)),batch_size,False)
            #y_batch=y[tau*batch_size:(tau+1)*batch_size].copy()
            #z_batch=z[tau*batch_size:(tau+1)*batch_size].copy()
            x_batch=x[index].copy()
            y_batch=y[index].copy()
            z_batch=z[index].copy()
            beta_copy[s]=mh_sample_beta(x_batch,y_batch,z_batch,beta_copy[s],beta,rho)
            #theta_copy[tau]=mh_sample_theta(y_batch,z_batch,K,u_copy[tau],theta_copy[tau],batch_num,theta)
            
        # update z
        if group==2:
            # decide which batch to obtain
            batch=np.random.choice(np.arange(batch_num),1,True)[0]
            # obtain the index of z
            index=np.arange(batch*batch_size,(batch+1)*batch_size)
            # obtain the parameters that corresponding to each z
            para_index=tau[index]
            #index=np.random.choice(np.arange(len(y)),batch_size,False)
            #y_batch=y[tau*batch_size:(tau+1)*batch_size].copy()
            #z_batch=z[tau*batch_size:(tau+1)*batch_size].copy()
            x_batch=x[index].copy()
            y_batch=y[index].copy()
            z_batch=z[index].copy()
            for r in range(0,batch_size):
               
                z_batch[r]=sample_z(x_batch[r],y_batch[r],z_batch[r],beta_copy[para_index[r]])
            
            #z_batch=sample_z(x_batch,y_batch,z_batch,beta_copy[s])
            #z[tau*batch_size:(tau+1)*batch_size]=z_batch
            z[index]=z_batch
        # update tau
        if group==3:
            # randomly select two parameters
            s0,s1=np.random.choice(np.arange(batch_num),2,False)
            new_tau=tau.copy()
            # obtain index of samples corresponding to para s0 and s1
            index0=np.where(tau==s0)[0]
            index1=np.where(tau==s1)[0]
            # switch the indexes
            switcher=np.random.choice([0,1],len(index0),True)
            new_index0=np.where(switcher==0,index0,index1)
            new_index1=np.where(switcher==1,index0,index1)
            
            
            ratio=evaluator(x[new_index0],y[new_index0],z[new_index0],beta_copy[s0],beta,rho)
            ratio=ratio+evaluator(x[new_index1],y[new_index1],z[new_index1],beta_copy[s1],beta,rho)
            ratio=ratio-evaluator(x[index0],y[index0],z[index0],beta_copy[s0],beta,rho) 
            ratio=ratio-evaluator(x[index1],y[index1],z[index1],beta_copy[s1],beta,rho)
            
            # construct the switched tau
            new_tau[new_index0]=s0
            new_tau[new_index1]=s1
            ratio=min(ratio,0)
            u=np.random.uniform(0,1,1)[0]
            u=np.log(u)
            if u<=ratio:
                #print('tau switched!!!!!')
                tau=new_tau.copy()
            
        end=time.time()
        #print('time used: ',end-start)
    return post_beta,beta_copy,z,tau