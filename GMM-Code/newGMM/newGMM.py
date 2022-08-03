# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:40:21 2022

@author: lidon
"""


import numpy as np
import scipy.stats as stats
import time

# generate data from a Bayesian Gaussian mixture model
# the variance are all set to 1
# Use results established in 2022.8.2

# K: the total number of clusters
# u: the vector consisting of each mean of each cluster
# theta: the vector for allocating classes
def GMM_generator(K,u,theta,n):
    z=[]
    y=[]
    for i in range(0,n):
        new_z=np.random.choice(K,1,True,p=theta)[0]
        new_y=np.random.normal(u[new_z],1,1)[0]
        z.append(new_z)
        y.append(new_y)
    y=np.array(y)
    z=np.array(z)
    
    return y,z

# initialize parameters & latent variables
def GMM_initializer(y,K,u,theta):
    theta0=np.random.dirichlet([1 for i in range(0,K)],1)[0]
    u0=np.random.normal(0,1,K)
    z0=[]
    for i in range(0,len(y)):
        norm_pdf=[stats.norm.pdf(y[i],u0[j],1) for j in range(0,K)]
        norm_pdf=np.array(norm_pdf)
        category=norm_pdf*theta0/sum(norm_pdf*theta0)
        new_z0=np.random.choice(K,1,p=category)[0]
        z0.append(new_z0)
    z0=np.array(z0)
    return theta0,z0,u0

# evaluate log-likelihood on a batch (p(y,z|u,theta))
# y,z are batches
def batch_log_likelihood(y,z,K,u,theta):
    log_likelihood=0
    #u_prior=np.array([stats.norm.pdf(u[i],0,1) for i in range(0,K)])
    #log_likelihood=sum(np.log(u_prior))
    for i in range(0,len(z)):
        log_likelihood=log_likelihood+np.log(theta[z[i]])+np.log(stats.norm.pdf(y[i],u[z[i]],1))
    return log_likelihood

# Gibbs sampler for GMM

# sample the weight
def sample_theta(y,z,K,u):
    obs_num=np.array([np.sum(z==k) for k in range(0,K)])
    new_theta=np.random.dirichlet(1+obs_num,1)[0]
    return new_theta

# sample the weigth, but accept using a MH step
# sample the copy of theta
# global_theta is the true param, theta is the copy
def mh_sample_theta(y,z,K,u,theta,batch_num,global_theta):
    new_theta=theta.copy()
    
    obs_num=np.array([np.sum(z==k) for k in range(0,K)])
    new_theta=np.random.dirichlet(1+batch_num*obs_num,1)[0]
    
    
    new_theta=np.random.dirichlet(1+theta,1)[0]
    # MH step
    # compute the MH ratio
    # likelihood part
    ratio=batch_log_likelihood(y,z,K,u,new_theta)-batch_log_likelihood(y,z,K,u,theta)
    #proposal distribution correction
    ratio=ratio+stats.dirichlet.logpdf(theta,1+new_theta)-stats.dirichlet.logpdf(new_theta,1+theta)
    #ratio=ratio+np.log(stats.dirichlet.pdf(theta,1+obs_num))-np.log(stats.dirichlet.pdf(new_theta,1+obs_num))
    # joint distribution part
    ratio=ratio+stats.dirichlet.logpdf(new_theta,1+global_theta)-stats.dirichlet.logpdf(theta,1+global_theta)
    alpha= ratio and 0
    v=np.random.uniform(0,1,1)[0]
    v=np.log(v)
    
    if v>alpha:
        new_theta=theta
    
    return new_theta

# sample latent variables
def sample_z(y,z,K,u,theta):
    new_z=z.copy()
    for i in range(0,len(z)):
        norm_pdf=[stats.norm.pdf(y[i],u[j],1) for j in range(0,K)]
        norm_pdf=np.array(norm_pdf)
        category=norm_pdf*theta
        category=category/sum(category)
        new_zi=np.random.choice(K,1,p=category)[0]
        new_z[i]=new_zi
    return new_z

# sample normal means
def sample_u(y,z,K):
    new_u=np.array([1. for i in range(0,K)])
    for i in range(0,K):
        indexer=(z==i)
        nk=sum(indexer)
        #print(nk)
        new_u[i]=np.random.normal(1/(nk+1)*sum(y[indexer]),1/(nk+1))
    return new_u

# sample normal means, but accept using a MH step
# what we sample is the copy, not the true one
# global_u is the true u, u is the copy generated
def mh_sample_u(y,z,K,u,theta,global_u):
    new_u=u.copy()
    '''
    for i in range(0,len(u)):
        indexer=(z==i)
        nk=sum(indexer)
        #print(nk)
        new_u[i]=np.random.normal(1/(nk+1)*sum(y[indexer]),1/(nk+1))
        '''
    # sample u by random walk mh
    new_u=np.random.normal(0,0.1,K)+u
    # MH step
    # compute the MH ratio
    ratio=batch_log_likelihood(y,z,K,new_u,theta)-batch_log_likelihood(y,z,K,u,theta)
    ratio=ratio+sum(stats.norm.logpdf(new_u,global_u,1)-stats.norm.logpdf(u,global_u,1))
    #print(ratio)
    '''
    for i in range(0,K):
        indexer=(z==i)
        nk=sum(indexer)
        print(stats.norm.pdf(u[i],1/(nk+1)*sum(y[indexer]),1/(nk+1)))
        ratio=ratio+np.log(stats.norm.pdf(u[i],1/(nk+1)*sum(y[indexer]),1/(nk+1)))
        #print(ratio)
        print(stats.norm.pdf(new_u[i],1/(nk+1)*sum(y[indexer]),1/(nk+1)))
        ratio=ratio-np.log(stats.norm.pdf(new_u[i],1/(nk+1)*sum(y[indexer]),1/(nk+1)))
        #print(ratio)
    '''
    #print(ratio)
    alpha= min(ratio,0)
    v=np.random.uniform(0,1,1)[0]
    v=np.log(v)
    if v> alpha:
        new_u=u
    return new_u

# follow results in 2022.8.2
# generate batch_num copies of theta and u from a pre-specified distribution
# use N(u,1) to generate us and dirichlet(1+theta) to generate thetas
def copy_generator(y,z,K,u,theta,batch_size):
    theta_copy=[]
    u_copy=[]
    batch_num=int(y.shape[0]/batch_size)
    for i in range(0,batch_num):
        u_copy.append(np.random.normal(0,1,len(u))+u)
        theta_copy.append(np.random.dirichlet(1+theta,1)[0])
    theta_copy=np.array(theta_copy)
    u_copy=np.array(u_copy)
    return u_copy,theta_copy

# u and theta are global u and global theta
# sample them according to their copies
# use mh step
def sample_global_param(y,z,K,u,theta,u_copy,theta_copy):
    '''
    # first, sample u
    # use random walk metropolis
    new_u=np.random.normal(0,0.1,len(u))+u
    #prior_mean=np.zeros(len(new_u))
    ratio=sum(stats.norm.logpdf(new_u,0,1)-stats.norm.logpdf(u,0,1))
    upper=np.array([sum(stats.norm.logpdf(u_copy[k],new_u,1)) for k in range(0,u_copy.shape[0])])
    lower=np.array([sum(stats.norm.logpdf(u_copy[k],u,1)) for k in range(0,u_copy.shape[0])])
    ratio=ratio+sum(upper)-sum(lower)
    #print(ratio)
    alpha= min(ratio,0)
    v=np.random.uniform(0,1,1)[0]
    v=np.log(v)
    if v> alpha:
        new_u=u
    '''
    # sample directly from a Gibbs step
    #new_u=np.random.normal(0,1,len(u))
    #new_u=new_u*np.sqrt(1/(len(u_copy)+1))+(sum(u_copy))/(len(u_copy)+1)
    new_u=sum(u_copy)/len(u_copy)
    #new_u=[stats.norm.rvs(sum(u_copy)[i]/(len(u_copy)+1),1/(len(u_copy)+1),1)[0] for i in range(0,len(u))]
    
    # then sample theta
    # again, use metropolis
    # proprosal: dir(1+theta)
    '''
    # proposal distribution is a projected random walk Metropolis
    ratio=0
    theta_trans=np.log(theta/theta[len(theta)-1])[0:len(theta)-1]
    new_theta_trans=np.random.normal(0,0.1,len(theta_trans))+theta_trans
    new_theta=np.exp(new_theta_trans)/(1+sum(np.exp(new_theta_trans)))
    new_theta=np.append(new_theta,1-sum(new_theta))
    #print('new theta:',new_theta)
    #new_theta=np.random.dirichlet(1+theta,1)[0]
    upper=np.array([stats.dirichlet.logpdf(theta_copy[k],1+new_theta) for k in range(0,theta_copy.shape[0])])
    lower=np.array([stats.dirichlet.logpdf(theta_copy[k],1+theta) for k in range(0,theta_copy.shape[0])])
    #ratio=stats.dirichlet.logpdf(theta,1+new_theta)-stats.dirichlet.logpdf(new_theta,1+theta)
    ratio=ratio+sum(np.log(new_theta))-sum(np.log(theta))
    ratio=ratio+sum(upper)-sum(lower)
    print(ratio)
    alpha= min(ratio,0)
    v=np.random.uniform(0,1,1)[0]
    v=np.log(v)
    if v> alpha:
        new_theta=theta
        '''
    new_theta=sum(theta_copy)/len(theta_copy)
    return new_u, new_theta
    
    
def GMM_Gibbs(y,z,K,u,theta,n):
    post_u=[]
    post_theta=[]
    for i in range(0,n):
        print('iteration: ',i)
        z=sample_z(y,z,K,u,theta)
        u=sample_u(y,z,K)
        theta=sample_theta(y,z,K,u)
        print('u: ',u)
        print('theta: ',theta)
        post_u.append(u)
        post_theta.append(theta)
    return post_u,post_theta,z

def batched_GMM_Gibbs(y,z,K,u,theta,n,batch_size):
    batch_num=int(y.shape[0]/batch_size)
    post_u=[]
    post_theta=[]
    
    
    for i in range(0,n):
        tau=np.random.choice(np.arange(batch_num),1,True)[0]
        print('tau: ',tau)
        y_batch=y[tau*batch_size:(tau+1)*batch_size].copy()
        z_batch=z[tau*batch_size:(tau+1)*batch_size].copy()
        print('iteration: ',i)
        toss=np.random.uniform(0,1,1)[0]
        if toss>0.5:
            tester=np.random.choice(batch_num+1,1,True)[0]
            if tester !=1:
                u=sample_u(y_batch,z_batch,K)
                post_u.append(u)
                theta=sample_theta(y_batch,z_batch,K,u)
                post_theta.append(theta)
                print('u: ',u)
                print('theta: ',theta)
            else:
                print('pass')
        else:
            z_batch=sample_z(y_batch,z_batch,K,u,theta)
            z[tau*batch_size:(tau+1)*batch_size]=z_batch
    return post_u,post_theta,z

def mh_batched_GMM_Gibbs(y,z,K,u,theta,n,batch_size):
    batch_num=int(y.shape[0]/batch_size)
    post_u=[]
    post_theta=[]
    
    
    for i in range(0,n):
        tau=np.random.choice(np.arange(batch_num),1,True)[0]
        print('tau: ',tau)
        y_batch=y[tau*batch_size:(tau+1)*batch_size].copy()
        z_batch=z[tau*batch_size:(tau+1)*batch_size].copy()
        print('iteration: ',i)
        toss=np.random.uniform(0,1,1)[0]
        if toss>0.5:
            tester=np.random.choice(batch_num+1,1,True)[0]
            if tester !=1:
                u=mh_sample_u(y_batch,z_batch,K,u,theta)
                post_u.append(u)
                theta=mh_sample_theta(y_batch,z_batch,K,u,theta,batch_num)
                post_theta.append(theta)
                print('u: ',u)
                print('theta: ',theta)
            else:
                print('pass')
        else:
            z_batch=sample_z(y_batch,z_batch,K,u,theta)
            z[tau*batch_size:(tau+1)*batch_size]=z_batch
    return post_u,post_theta,z


def novel_batched_GMM_Gibbs(y,z,K,u,theta,n,batch_size):
    batch_num=int(y.shape[0]/batch_size)
    post_u=[]
    post_theta=[]
    u_copy,theta_copy=copy_generator(y,z,K,u,theta,batch_size)
    theta_copy=np.array([[0.5,0.25,0.25] for i in range(0,batch_num)])
    # specify the weight for RSGS
    prob=np.array([1/3,1/3,1/3])
    # 1/3 for updating theta, 1/3 for updating thetas, 1/3 for updating zs
    for i in range(0,n):
        start=time.time()
        print(f'iteration: {i}')
        group=np.random.choice([0,1,2],1,True,prob)[0]
        # update theta
        if group==0:
            u,theta=sample_global_param(y, z, K, u, theta, u_copy, theta_copy)
            print(f'u: {u}')
            
            post_u.append(u)
            #theta=np.array([0.5,0.25,0.25])
            print(f'theta: {theta}')
            post_theta.append(theta)
        # update thetas
        if group==1:
            tau=np.random.choice(np.arange(batch_num),1,True)[0]
            index=np.arange(tau*batch_size,(tau+1)*batch_size)
            #index=np.random.choice(np.arange(len(y)),batch_size,False)
            #y_batch=y[tau*batch_size:(tau+1)*batch_size].copy()
            #z_batch=z[tau*batch_size:(tau+1)*batch_size].copy()
            y_batch=y[index].copy()
            z_batch=z[index].copy()
            u_copy[tau]=mh_sample_u(y_batch,z_batch,K,u_copy[tau],theta_copy[tau],u)
            #theta_copy[tau]=mh_sample_theta(y_batch,z_batch,K,u_copy[tau],theta_copy[tau],batch_num,theta)
            
        # update z
        if group==2:
            tau=np.random.choice(np.arange(batch_num),1,True)[0]
            index=np.arange(tau*batch_size,(tau+1)*batch_size)
            #index=np.random.choice(np.arange(len(y)),batch_size,False)
            #y_batch=y[tau*batch_size:(tau+1)*batch_size].copy()
            #z_batch=z[tau*batch_size:(tau+1)*batch_size].copy()
            y_batch=y[index].copy()
            z_batch=z[index].copy()
            z_batch=sample_z(y_batch,z_batch,K,u,theta)
            #z[tau*batch_size:(tau+1)*batch_size]=z_batch
            z[index]=z_batch
        end=time.time()
        #print('time used: ',end-start)
    return post_u,post_theta,z,u_copy,theta_copy

def permute(post_u,post_theta,u,theta):
    if 1:
        #est_u=sum(post_u[5000:])/len(post_u[5000:])
        est_u=post_u[-1]
        # if the estimated pi is not at the right order
        if np.any(-np.sort(-est_u)!=est_u):
            right_u=-np.sort(-est_u)
            permutation=[np.where(right_u[i]==est_u)[0][0] for i in range(0,len(est_u))]
            for i in range(0,len(post_u)):
                post_u[i]=post_u[i][permutation]
                
                post_theta[i]=post_theta[i][permutation]
    return post_u,post_theta
