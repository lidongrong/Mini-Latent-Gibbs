# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:23:29 2022

@author: lidon
"""

######
# Perform MH within Gibbs Sampler
######


import numpy as np
import HMM
import SeqSampling as Sampling
import time
from multiprocessing import Pool
import scipy.stats as stats
import math
import os
import multiprocessing as mp




# initialize data, transition matrix and obs_matrix
# size: number of sample points
# long: length of each seq
def data_initializer(transition,obs_prob,pi,hidden_state,obs_state,rate,size,long):
    #print('Data Preprocessing and Initializing Parameters...')
    #data=Sampling.data
    
    # Generate the HMM object
    MC=HMM.HMM(hidden_state,obs_state,transition,obs_prob,pi)


    # Construct Data
    hidden_data=Sampling.Hidden_Generator(MC,long,size)
    data=Sampling.Obs_Generator(MC,hidden_data)

    # Missing at random
    #data=Sampling.Missing(data,p=rate)
    
    # Initialize transition matrix A
    
    A=np.array([[1/transition.shape[1] for i in range(0,transition.shape[1])] for j in range(0,transition.shape[0])])
    for i in range(0,A.shape[0]):
        A[i,:]=A[i,:]/sum(A[i,:])
    
    # Initialize observation matrix B and pi
    # Dirichlet parameter of B
    
    #A=np.random.dirichlet(np.ones(transition.shape[1]),transition.shape[0])
    
    #alpha_B=np.array([1 for i in range(0,obs_prob.shape[1])])
    #B=np.random.dirichlet(alpha_B,obs_prob.shape[0])
        
    B=np.array([[1/obs_prob.shape[1] for i in range(0,obs_prob.shape[1])] for j in range(0,obs_prob.shape[0])])
    for i in range(0,B.shape[0]):
        B[i,:]=B[i,:]/sum(B[i,:])
    
    
    pi=np.array([1/len(pi) for i in range(0,len(pi))])
    pi=pi/sum(pi)
    
    
    
    A=np.random.dirichlet([8 for i in range(0,A.shape[1])],A.shape[0])
    B=np.random.dirichlet([8 for i in range(0,B.shape[1])],B.shape[0])
    pi=np.random.dirichlet([8 for i in range(0,pi.shape[0])],pi.shape[0])[0]
    
    
    return A,B,pi,data,hidden_data

def param_initialize():
    A=np.random.dirichlet((8,8,8),3)
    B=np.random.dirichlet((8,8,8),3)
    pi0=np.random.dirichlet((8,8,8),1)[0]
    return A,B,pi0

# Sample the latent state using forward backward sampling
# Based on research note in 2021.10.21
# A,B: transition matrix and observation matrix
# pi: initial probability
# state: observed sequence with missing observation
def f_b_sampling(A,B,pi,obs,hidden_state,obs_state):
    
    # Check if the whole sequence is missing
    if np.all(obs=='None'):
        return obs
    
    # acquire the index that correspond to observations that are not missing
    indexer=np.where(obs!='None')[0]
    
    # length of the index
    T=len(obs)
    #state=HMM.obs_state
    #state=obs_state
    #hidden_state=HMM.hidden_state
    #hidden_state=hidden_state
    
    # start to compute alpha recursively
    alpha=np.zeros((T,len(hidden_state)))

    #y=np.where(obs_state==obs[indexer[0]])[0][0]
    initial=pi
    
    # Handle the boundary case: the first line of alpha
    if obs[0]!='None':
        y=np.where(obs_state==obs[0])[0][0]
        
        alpha[0,:]=initial*B[:,y]
    else:
        alpha[0,:]=initial
        
        
    #alpha[0,:]=np.dot(initial,np.linalg.matrix_power(A, indexer[0]))*B[:,y]
    
    
    for i in range(1,T):
        # The case when i is an observable data
        if i in indexer:
            y=np.where(obs_state==obs[i])[0][0]
            alpha[i,:]=np.dot(alpha[i-1,:],A)*B[:,y]
        else:
            alpha[i,:]=np.dot(alpha[i-1,:],A)
    
     
    # initialize the output
    output=[]
    
    #print(alpha)
    
    # First sample the last latent state
    w=alpha[T-1,:]/sum(alpha[T-1,:])
    output.append(np.random.choice(hidden_state,1,p=w)[0])
    
    # Then sample each latent state in sequence
    for t in range(1,T):
        # compute the index of hidden state z_{t_{i+1}}
        hidden_index=np.where(hidden_state==output[t-1])[0][0]
        # compute the transition matrix between the two observed states
        trans=A
        # generate the probability distribution
        w=trans[:,hidden_index]*alpha[T-1-t,:]/np.dot(trans[:,hidden_index],alpha[T-1-t,:])
        #output.append(np.random.choice(HMM.hidden_state,1,p=w)[0])
        output.append(np.random.choice(hidden_state,1,p=w)[0])
    output.reverse()
    output=np.array(output)
    return output


# sample the whole latent sequence out
# A,B:transition matrix and obs matrix
# data: partially observed data
# I: latent sequence from the last iteration
# pi: initial probability
def sample_latent_seq(data,I,A,B,pi,hidden_state,obs_state):
    for i in range(0,data.shape[0]):
        I[i,:]=f_b_sampling(A,B,pi,data[i],hidden_state,obs_state)
    return I

# sample y out given I
def sample_y(obs,z,A,B,pi,hidden_state,obs_state):
    y=obs.copy()
    indexer=np.where(y=='None')[0]
    for i in indexer:
        z_pos=np.where(hidden_state==z[i])[0][0]
        y[i]=np.random.choice(obs_state,1,p=B[z_pos])[0]
    return y



# initialize the latent sequence as initial guess
def latent_seq_initializer(data,A,B,pi,hidden_state,obs_state):
    # initialize latent sequence I
    I=[]
    for i in range(0,data.shape[0]):
        I.append(np.repeat('None',data.shape[1]))
    I=np.array(I)

    I=sample_latent_seq(data,I,A,B,pi,hidden_state,obs_state)
    return I

# Make an initial guess of the parameters and latent variables
def initialize(hidden_state,obs_state,transition,obs_prob,pi,rate,size,long):
    A,B,pi,data,hidden_data=data_initializer(transition,obs_prob,pi,hidden_state,obs_state,rate,size,long)
    I=latent_seq_initializer(data,A,B,pi,hidden_state,obs_state)
    return A,B,pi,data,I,hidden_data



# Sample B out in Gibbs Sampling
# B: B sampled in last iteration
def sample_B(data,I,B,hidden_state,obs_state):
    new_B=B.copy()
    
    
    for j in range(0,B.shape[0]):
        # for j th row of B, calculate the number of each 
        # observed states respectively
        
        obs_num=np.array([np.sum(np.logical_and(I==hidden_state[j],data==obs_state[k])) for k in range(0,B.shape[1])])
        new_B[j,:]=np.random.dirichlet(1+obs_num,1)[0]
    
    #print(B)
    return new_B


# Sample B out in Gibbs Sampling
# B: B sampled in last iteration
def bsample_B(data,I,B,hidden_state,obs_state,data_size):
    new_B=B.copy()
    
    
    for j in range(0,B.shape[0]):
        # for j th row of B, calculate the number of each 
        # observed states respectively
        
        obs_num=np.array([np.sum(np.logical_and(I==hidden_state[j],data==obs_state[k])) for k in range(0,B.shape[1])])
        
        obs_num=obs_num*(int(data_size/data.shape[0]))
        
        new_B[j,:]=np.random.dirichlet(1+obs_num,1)[0]
    
    #print(B)
    return new_B


# evaluate the likelihood of a sequqnce given A
# seq: hidden sequence
# A,pi: transition matrix and initial distribution
def p_seq(seq,A,pi,hidden_state,obs_state):
    indexer=np.where(seq!='None')[0]
    log_p=0
    
    # in case the whole sequence is missing
    if indexer.size==0:
        return 0
    
    
        
    if indexer.size>=1:
        pos=np.where(hidden_state==seq[0])[0][0]
        log_p=np.log(np.dot(pi,A[:,pos]))
                
        for i in range(0,len(seq)-1):
            current_pos=np.where(hidden_state==seq[i])[0][0]
            future_pos=np.where(hidden_state==seq[i+1])[0][0]
            log_p=log_p+np.log(A[current_pos][future_pos])
    return log_p


# Sample A out using Metropolis within Gibbs with a log_normal propose
# log-likelihood is evaluated on the full data set
# p is the multiprocessing core
def sample_A(data,I,A,hidden_state,obs_state,p):
    
    new_A=A.copy()
    
    for i in range(0,A.shape[0]):
        
        transform=np.array([np.sum((I[:,:-1]==hidden_state[i])&(I[:,1:]==hidden_state[j])) for j in range(A.shape[1])])
    
        new_A[i,:]=np.random.dirichlet(1+transform,1)[0]
   
    
    return new_A

# sample A using a minibatch and estimate of the joint prob
# data: batch of data
# data_size: size of the full data
def bsample_A(data,I,A,hidden_state,obs_state,p,data_size):
    new_A=A.copy()
    
    for i in range(0,A.shape[0]):
        
        transform=np.array([np.sum((I[:,:-1]==hidden_state[i])&(I[:,1:]==hidden_state[j])) for j in range(A.shape[1])])
        transform=transform*(int(data_size/data.shape[0]))
        new_A[i,:]=np.random.dirichlet(1+transform,1)[0]
   
    
    return new_A


# Compute the probability of the first observable state in a hidden sequence give A & pi
def p_first_state(seq,A,pi,hidden_state,obs_state):
    indexer=np.where(seq!='None')[0]
    log_p=0
    
    # In case the whole seq is missing
    if indexer.size==0:
        return 0
    
    pos=np.where(hidden_state==seq[0])[0][0]
    log_p=np.log(np.dot(pi,A[:,pos]))
    return log_p


# just testing
def sample_pi(I,A,pi,hidden_state,obs_state,p):
    
    new_pi=pi.copy()
    
    transform=[np.count_nonzero(I[:,0]==hidden_state[i]) for i in range(0,len(hidden_state))]
    
    transform=np.array(transform)
    
    new_pi=np.random.dirichlet(1+transform,1)[0]
    
    return new_pi


# just testing
def bsample_pi(I,A,pi,hidden_state,obs_state,p,data_size):
    
    new_pi=pi.copy()
    
    transform=[np.count_nonzero(I[:,0]==hidden_state[i]) for i in range(0,len(hidden_state))]
    
    transform=np.array(transform)
    transform=transform*(int(data_size/I.shape[0]))
    
    new_pi=np.random.dirichlet(1+transform,1)[0]
    
    return new_pi

'''
# sample the initial distribution pi out
# pi: pi from last iteration
# Use Metropolis within Gibbs algorithm
# p is the multiprocessing core
def sample_pi(I,A,pi,hidden_state,obs_state,p):
    
    
    
    new_pi=np.random.dirichlet(pi+1,1)[0]
    log_p=sum(p.starmap(p_first_state,[(seq,A,pi,hidden_state,obs_state) for seq in I]))
    log_new_p=sum(p.starmap(p_first_state,[(seq,A,new_pi,hidden_state,obs_state) for seq in I]))
    
    # Metropolis Step
    r=log_new_p+np.log(stats.dirichlet.pdf(pi,new_pi+1))-log_p-np.log(stats.dirichlet.pdf(new_pi,pi+1))
    r=min(0,r)
    
    # Metropolis step
    u=np.random.uniform(0,1,1)[0]
    if np.log(u)<r:
        pi=new_pi
    
    
    return pi
'''


   
# Locate the indexer of a specific element in hidden_state
def hidden_loc(element,hidden_state):
    return np.where(hidden_state==element)[0][0]

# Locate the indexer of a specific element in obs_seq
def obs_loc(element,obs_state):
    return np.where(obs_state==element)[0][0]

vhidden_loc=np.vectorize(hidden_loc,excluded=['hidden_state'])
vobs_loc=np.vectorize(obs_loc,excluded=['obs_state'])

# evaluate the likelihood of an observed sequence
# seq_hidden, seq_obs: latent sequqnce and observed sequqnce respectively
def p_observe(hidden_seq,obs_seq,B,hidden_state,obs_state):
    #hidden_state=HMM.hidden_state
    #obs_state=HMM.obs_state
    
    # indices of the observed data
    indexer=np.array(np.where(obs_seq!='None')[0])
    
    # initialize the log likelihood
    log_y=0
    
    # in case the whole sequqnce is missing
    if indexer.size>=1:
        '''
        for k in range(0,len(indexer)):
            z_pos=np.where(hidden_state==hidden_seq[indexer[k]])[0][0]
            y_pos=np.where(obs_state==obs_seq[indexer[k]])[0][0]
            log_y=log_y+np.log(B[z_pos,y_pos])
            '''
        
        #z_pos=vhidden_loc(hidden_seq[indexer],hidden_state)
        z_pos=np.array([hidden_loc(hidden_seq[k],hidden_state) for k in indexer])
        #y_pos=vobs_loc(obs_seq[indexer],obs_state)
        y_pos=np.array([obs_loc(obs_seq[k],obs_state) for k in indexer])
        log_y=np.sum(np.log(B[z_pos,y_pos]))
    
    return log_y
            
   


# evaluate the log-likelihood of the estimation that helps selecting the prediction
# p is the multiprocessing core
def p_evaluator(A,B,pi,I,data,hidden_state,obs_state,p):    
    #obs_state=HMM.obs_state
    #hidden_state=HMM.hidden_state
    # initialize the log likelihood
    log_p=0
    
    # first compute log likelihood of B
    alpha=np.array([1 for i in range(0,B.shape[1])])
    
    log_p=np.sum([stats.dirichlet.logpdf(B[i,:],alpha) for i in range(0,B.shape[0])])
    
    '''
    # Then compute the loglikelihood of Y|Z,B and Z
    for i in range(0,I.shape[0]):
    '''   
    # Note here p is the multiprocessing pool defined in global main() function
    
    # log_z is the log likelihood of the latent sequqnce
    log_z=sum(p.starmap(p_seq,[(seq,A,pi,hidden_state,obs_state) for seq in I]))
    # log_y is the log likelihood of the observed sequence
    log_y=sum(p.starmap(p_observe,[(I[i],data[i],B,hidden_state,obs_state) for i in range(0,I.shape[0])]))
    
    log_p=log_p+log_z+log_y
        
    
        
    return log_p

# evaluate the log-likelihood defined on a small batch
def batch_likelihood(A,B,pi,I,data,hidden_state,obs_state):
    log_p=0
    for i in range(0,data.shape[0]):
        log_p=log_p+p_seq(I[i],A,pi,hidden_state,obs_state)+p_observe(I[i],data[i],B,hidden_state,obs_state)
    
    return log_p

# evaluate the log-likelihood of each observation that helps selecting the prediction
# return a vector of log-prob
# p is the multiprocessing core
def p_sample(A,B,pi,I,data,hidden_state,obs_state,p):    
    #obs_state=HMM.obs_state
    #hidden_state=HMM.hidden_state
    # initialize the log likelihood
    
    
    # first compute log likelihood of B
    
    '''
    # Then compute the loglikelihood of Y|Z,B and Z
    for i in range(0,I.shape[0]):
    '''   
    # Note here p is the multiprocessing pool defined in global main() function
    
    # log_z is the log likelihood of the latent sequqnce
    log_z=p.starmap(p_seq,[(seq,A,pi,hidden_state,obs_state) for seq in I])
    # log_y is the log likelihood of the observed sequence
    log_y=p.starmap(p_observe,[(I[i],data[i],B,hidden_state,obs_state) for i in range(0,I.shape[0])])
    
    log_z=np.array(log_z)
    log_y=np.array(log_y)
    
    # Return the vector of log prob
    log_p=log_z+log_y
        
    
        
    return log_p

# Gibbs sampling using Metropolis within Gibbs algorithm (acceleration by parallel computing)
# input I,A,B,pi: initial guesses of the parameter
# n: number of samples to draw
# p: Pool
def parallel_Gibbs(data,I,A,B,pi,n,hidden_state,obs_state,p):
    post_A=[]
    post_B=[]
    log_prob=[]
    post_pi=[]
    
    
    # construct a buffer to store the latent sequence with largest likelihood
    I_buffer=I.copy()
    #log_p=p_evaluator(A,B,pi,I_buffer,data,hidden_state,obs_state,p)
    
    # log prob of each sample, help to select hidden state
    #selector=p_sample(A,B,pi,I_buffer,data,hidden_state,obs_state,p)
    #log_prob.append(log_p)
    
    for i in range(0,n):
        start=time.time()
        print(i)
        
        pi=sample_pi(I,A,pi,hidden_state,obs_state,p)
        print('pi',pi)
        post_pi.append(pi)
        B=sample_B(data,I,B,hidden_state,obs_state)
        A=sample_A(data,I,A,hidden_state,obs_state,p)
        
        post_A.append(A)
        print('A',A)
        post_B.append(B)
        
        print('B',B)
        
        
        I=p.starmap(f_b_sampling,[(A,B,pi,data[i],hidden_state,obs_state) for i in range(0,I.shape[0])])
        I=np.array(I)

        
        #new_log_p=p_evaluator(A,B,pi,I,data,hidden_state,obs_state,p)
        new_selector=p_sample(A,B,pi,I,data,hidden_state,obs_state,p)
        new_log_p=sum(new_selector)
        #new_log_p=1
        log_prob.append(new_log_p)
        
        '''
        if new_log_p>log_p:
            I_buffer=I.copy()
            log_p=new_log_p
        '''
        
        #indicator=new_selector>selector
        #print(np.sum(indicator))
        
        #I_buffer[indicator]=I[indicator]
        #selector[indicator]=new_selector[indicator]
        
        end=time.time()
        print(end-start)
        
        
        
    post_A=np.array(post_A)
    post_B=np.array(post_B)
    log_prob=np.array(log_prob)
    post_pi=np.array(post_pi)
    
    return post_A,post_B,post_pi,I



#Sample by a proposed stochastic gibbs sampler
#n: total sample generated
#batch_size: batch size
def batched_Gibbs(data,I,A,B,pi,n,batch_size,hidden_state,obs_state,p):
    post_A=[]
    post_B=[]
    post_pi=[]
    
    # prevent num being a float
    batch_num=int(data.shape[0]/batch_size)
    
    #indicator of what is being sampled
    #param=True
    for i in range(0,n):
        print(i)
        start=time.time()
        
            
        tau=np.random.choice(np.arange(batch_num),1,True)[0]
        data_batch=data[tau*batch_size:(tau+1)*batch_size,:].copy()
        I_batch=I[tau*batch_size:(tau+1)*batch_size,:].copy()
            
        # sample model parameters
        #pi=bsample_pi(I_batch,A,pi,hidden_state,obs_state,p,data.shape[0])
        pi=sample_pi(I_batch,A,pi,hidden_state,obs_state,p)
        print('pi',pi)
        post_pi.append(pi)
        #A=bsample_A(data_batch,I_batch,A,hidden_state,obs_state,p,data.shape[0])
        A=sample_A(data_batch,I_batch,A,hidden_state,obs_state,p)
        post_A.append(A)
        print('A',A)
        #B=bsample_B(data_batch,I_batch,B,hidden_state,obs_state,data.shape[0])
        B=sample_B(data_batch,I_batch,B,hidden_state,obs_state)
        post_B.append(B)
        print('B',B)
        
            
        
            
        tau=np.random.choice(np.arange(batch_num),1,True)[0]
        data_batch=data[tau*batch_size:(tau+1)*batch_size,:].copy()
        I_batch=I[tau*batch_size:(tau+1)*batch_size,:].copy()
            
        I_batch=sample_latent_seq(data_batch,I_batch,A,B,pi,hidden_state,obs_state)
        I[tau*batch_size:(tau+1)*batch_size,:]=I_batch
        print(tau)
        end=time.time()
        print(end-start)
    return post_A,post_B,post_pi,I

#Sample by a proposed stochastic gibbs sampler
#n: total sample generated
#batch_size: batch size
def random_batched_Gibbs(data,I,A,B,pi,n,batch_size,hidden_state,obs_state,p):
    post_A=[]
    post_B=[]
    post_pi=[]
    
    # prevent num being a float
    batch_num=int(data.shape[0]/batch_size)
    
    #indicator of what is being sampled
    #param=True
    for i in range(0,n):
        print(i)
        start=time.time()
        
        
        tau=np.random.choice(np.arange(batch_num),1,True)[0]
        data_batch=data[tau*batch_size:(tau+1)*batch_size,:].copy()
        I_batch=I[tau*batch_size:(tau+1)*batch_size,:].copy()
        
        
        # decide sample model parameters or latent variables
        toss=np.random.uniform(0,1,1)[0]
        if toss>0.5:
            tester=np.random.choice(batch_num+1,1,True)[0]
            if tester==1:
                print('pass')
            else:
                # sample model parameters
                #pi=bsample_pi(I_batch,A,pi,hidden_state,obs_state,p,data.shape[0])
                pi=sample_pi(I_batch,A,pi,hidden_state,obs_state,p)
                print('pi',pi)
                post_pi.append(pi)
                #A=bsample_A(data_batch,I_batch,A,hidden_state,obs_state,p,data.shape[0])
                A=sample_A(data_batch,I_batch,A,hidden_state,obs_state,p)
                post_A.append(A)
                print('A',A)
                #B=bsample_B(data_batch,I_batch,B,hidden_state,obs_state,data.shape[0])
                B=sample_B(data_batch,I_batch,B,hidden_state,obs_state)
                post_B.append(B)
                print('B',B)
        else:
            I_batch=sample_latent_seq(data_batch,I_batch,A,B,pi,hidden_state,obs_state)
            I[tau*batch_size:(tau+1)*batch_size,:]=I_batch
        print('tau:',tau)
        end=time.time()
        print(end-start)
    return post_A,post_B,post_pi,I


# batched gibbs, but select batch with a metropolis jump
def novel_batched_Gibbs(data,I,A,B,pi,n,batch_size,hidden_state,obs_state,p):
    post_A=[]
    post_B=[]
    post_pi=[]
    
    # prevent num being a float
    batch_num=int(data.shape[0]/batch_size)
    
    tau=np.random.choice(np.arange(batch_num),1,True)[0]
    
    data_batch=data[tau*batch_size:(tau+1)*batch_size,:].copy()
    I_batch=I[tau*batch_size:(tau+1)*batch_size,:].copy()
    for i in range(0,n):
        print(i)
        start=time.time()
            
        # sample model parameters
        #pi=bsample_pi(I_batch,A,pi,hidden_state,obs_state,p,data.shape[0])
        pi=sample_pi(I_batch,A,pi,hidden_state,obs_state,p)
        print('pi',pi)
        post_pi.append(pi)
        #B=bsample_B(data_batch,I_batch,B,hidden_state,obs_state,data.shape[0])
        B=sample_B(data_batch,I_batch,B,hidden_state,obs_state)
        post_B.append(B)
        print('B',B)
        #A=bsample_A(data_batch,I_batch,A,hidden_state,obs_state,p,data.shape[0])
        A=sample_A(data_batch,I_batch,A,hidden_state,obs_state,p)
        post_A.append(A)
        print('A',A)
            
        # propose new batch
        # propose new tau from a uniform distribution
        new_tau=np.random.choice(np.arange(batch_num),1,True)[0]
        
        # compute acceptance ratio
        nom=batch_likelihood(A,B,pi,I[new_tau*batch_size:(new_tau+1)*batch_size,:],
                             data[new_tau*batch_size:(new_tau+1)*batch_size,:],hidden_state,obs_state)
        dom=batch_likelihood(A,B,pi,I[tau*batch_size:(tau+1)*batch_size,:],
                             data[tau*batch_size:(tau+1)*batch_size,:],hidden_state,obs_state)
        print((nom-dom)/batch_size) 
        # metropolis step
        ratio=min(0,(nom-dom)/batch_size)
        u=np.random.uniform(0,1,1)[0]
        u=np.log(u)
        
        if u<ratio:
            tau=new_tau
            
        print('tau: ',tau)
        data_batch=data[tau*batch_size:(tau+1)*batch_size,:].copy()
        I_batch=I[tau*batch_size:(tau+1)*batch_size,:].copy()
            
        I_batch=sample_latent_seq(data_batch,I_batch,A,B,pi,hidden_state,obs_state)
        I[tau*batch_size:(tau+1)*batch_size,:]=I_batch
        
        
        # propose new batch again
        # propose new tau from a uniform distribution
        new_tau=np.random.choice(np.arange(batch_num),1,True)[0]
        
        # compute acceptance ratio
        nom=batch_likelihood(A,B,pi,I[new_tau*batch_size:(new_tau+1)*batch_size,:],
                             data[new_tau*batch_size:(new_tau+1)*batch_size,:],hidden_state,obs_state)
        dom=batch_likelihood(A,B,pi,I[tau*batch_size:(tau+1)*batch_size,:],
                             data[tau*batch_size:(tau+1)*batch_size,:],hidden_state,obs_state)
        
        # metropolis step
        ratio=min(0,(nom-dom)/batch_size)
        u=np.random.uniform(0,1,1)[0]
        u=np.log(u)
        
        if u<=ratio:
            tau=new_tau
        print('2nd tau: ',tau)
        end=time.time()
        
        print('Time used: ',end-start)
        
    return post_A,post_B,post_pi,I

# jump among batches sequentially
def Ordered_Minibatch_Gibbs(data,I,A,B,pi,n,batch_size,hidden_state,obs_state,p):
    post_A=[]
    post_B=[]
    log_prob=[]
    post_pi=[]
    tau=0
    batch_num=int(data.shape[0]/batch_size)
    for i in range(0,n):
        
        data_batch=data[tau*batch_size:(tau+1)*batch_size,:].copy()
        I_batch=I[tau*batch_size:(tau+1)*batch_size,:].copy()
        start=time.time()
        print(i)
        
        # sample model parameters
        pi=bsample_pi(I_batch,A,pi,hidden_state,obs_state,p,data.shape[0])
        print('pi',pi)
        post_pi.append(pi)
        B=bsample_B(data_batch,I_batch,B,hidden_state,obs_state,data.shape[0])
        A=bsample_A(data_batch,I_batch,A,hidden_state,obs_state,p,data.shape[0])
        
        post_A.append(A)
        print('A',A)
        post_B.append(B)
        
        print('B',B)
        print('tau',tau)
        # sample I
        I_batch=sample_latent_seq(data_batch,I_batch,A,B,pi,hidden_state,obs_state)
        I[tau*batch_size:(tau+1)*batch_size,:]=I_batch
        #sample tau
        tau=(tau+1)%batch_num
            
        
        
        
        

    return post_A,post_B,post_pi,I
    

def Naive_Minibatch_Gibbs(data,I,A,B,pi,n,batch_size,hidden_state,obs_state,p):
    post_A=[]
    post_B=[]
    log_prob=[]
    post_pi=[]
    batch_num=int(data.shape[0]/batch_size)
    tau=np.random.choice(np.arange(batch_num),1,True)[0]
    batch_num=int(data.shape[0]/batch_size)
    for i in range(0,n):
        start=time.time()
        data_batch=data[tau*batch_size:(tau+1)*batch_size,:].copy()
        I_batch=I[tau*batch_size:(tau+1)*batch_size,:].copy()
        start=time.time()
        print(i)
        
        # sample model parameters
        pi=bsample_pi(I_batch,A,pi,hidden_state,obs_state,p,data.shape[0])
        print('pi',pi)
        post_pi.append(pi)
        B=bsample_B(data_batch,I_batch,B,hidden_state,obs_state,data.shape[0])
        A=bsample_A(data_batch,I_batch,A,hidden_state,obs_state,p,data.shape[0])
        
        post_A.append(A)
        print('A',A)
        post_B.append(B)
        
        print('B',B)
        print('tau',tau)
        # sample I
        I_batch=sample_latent_seq(data_batch,I_batch,A,B,pi,hidden_state,obs_state)
        I[tau*batch_size:(tau+1)*batch_size,:]=I_batch
        #sample tau
        tau=np.random.choice(np.arange(batch_num),1,True)[0]
        end=time.time()
        print(end-start)
    
    return post_A,post_B,post_pi,I

# define the output class of the experiments
class Out:
    def __init__(self,data,post_A,post_B,post_pi,latent_seq, log_prob,true_hidden):
        self.data=data
        self.post_A=post_A
        self.post_B=post_B
        self.post_pi=post_pi
        self.latent_seq=latent_seq
        self.log_prob=log_prob
        self.true_hidden=true_hidden
        
# permute the output to the right position
# out is the output array
def permute(post_pi,post_A,post_B,pi,transition,obs_prob):
    if 1:
        est_pi=sum(post_pi[7500:])/len(post_pi[7500:])
        # if the estimated pi is not at the right order
        if np.any(-np.sort(-est_pi)!=est_pi):
            right_pi=-np.sort(-est_pi)
            permutation=[np.where(right_pi[i]==est_pi)[0][0] for i in range(0,len(est_pi))]
            for i in range(0,len(post_pi)):
                post_A[i]=post_A[i][permutation,:]
                post_A[i]=post_A[i][:,permutation]
                post_B[i]=post_B[i][permutation,:]
                #output.post_B[i]=output.post_B[i][:,permutation]
                post_pi[i]=post_pi[i][permutation]
    
    return post_pi,post_A,post_B