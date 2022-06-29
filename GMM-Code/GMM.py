# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:24:43 2022

@author: lidon
"""

import numpy as np
import scipy.stats as stats


# generate data from a Bayesian Gaussian mixture model
# the variance are all set to 1

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
    theta0=np.random.dirichlet([8 for i in range(0,K)],1)[0]
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




# Gibbs sampler for GMM

# sample the weight
def sample_theta(y,z,K,u):
    obs_num=np.array([np.sum(z==k) for k in range(0,K)])
    new_theta=np.random.dirichlet(1+obs_num,1)[0]
    return new_theta

# sample latent variables
def sample_z(y,z,K,u,theta):
    new_z=z.copy()
    for i in range(0,len(z)):
        norm_pdf=[stats.norm.pdf(y[i],u[j],1) for j in range(0,K)]
        norm_pdf=np.array(norm_pdf)
        category=norm_pdf*theta/sum(norm_pdf*theta)
        new_zi=np.random.choice(K,1,p=category)[0]
        new_z[i]=new_zi
    return new_z

# sample normal menas
def sample_u(y,z,K,u):
    new_u=u.copy()
    for i in range(0,len(u)):
        indexer=(z==i)
        nk=sum(indexer)
        print(nk)
        new_u[i]=np.random.normal(1/(nk+1)*sum(y[indexer]),1/(nk+1))
    return new_u
    

def GMM_Gibbs(y,z,K,u,theta,n):
    post_u=[]
    post_theta=[]
    for i in range(0,n):
        print('iteration: ',i)
        z=sample_z(y,z,K,u,theta)
        u=sample_u(y,z,K,u)
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
                u=sample_u(y_batch,z_batch,K,u)
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

def novel_batched_GMM_Gibbs(y,z,K,u,theta,n,batch_size):
    batch_num=int(y.shape[0]/batch_size)
    post_u=[]
    post_theta=[]
    
    
    for i in range(0,n):
        #tau=np.random.choice(np.arange(batch_num),1,True)[0]
        batch_index=np.random.choice(y.shape[0],batch_size,False)
        #print('tau: ',tau)
        y_batch=y[batch_index].copy()
        z_batch=z[batch_index].copy()
        print('iteration: ',i)
        toss=np.random.uniform(0,1,1)[0]
        if toss>0.5:
            u=sample_u(y_batch,z_batch,K,u)
            post_u.append(u)
            theta=sample_theta(y_batch,z_batch,K,u)
            post_theta.append(theta)
            print('u: ',u)
            print('theta: ',theta)
        else:
            z_batch=sample_z(y_batch,z_batch,K,u,theta)
            z[batch_index]=z_batch
    return post_u,post_theta,z

def permute(post_u,post_theta,u,theta):
    if 1:
        est_u=sum(post_u[5000:])/len(post_u[5000:])
        # if the estimated pi is not at the right order
        if np.any(-np.sort(-est_u)!=est_u):
            right_u=-np.sort(-est_u)
            permutation=[np.where(right_u[i]==est_u)[0][0] for i in range(0,len(est_u))]
            for i in range(0,len(post_u)):
                post_u[i]=post_u[i][permutation]
                
                post_theta[i]=post_theta[i][permutation]
                
    
    return post_u,post_theta