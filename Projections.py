"""
Created on Mon Apr 18 10:12:51 2016

@author: boris

Modified by: Jonas Bresch, M.Sc.
Project: Renyi-regularized optimal transport

Date: April. 24th, 2024
"""
import numpy as np
from Distances import inner_Frobenius

#Sinkhorn-Knopp's algorithm - performs Kullback-Leibler projection on U(r,c)
def Sinkhorn(A,r,c,precision):
    
    p = np.nansum(A,axis = 1)
    q = np.nansum(A,axis = 0)
    count = 0

    while not (check_2(p,q,r,c,precision)) and count <= 4000:
        
        A = np.diag(np.divide(r,p)).dot(A)
        q = np.nansum(A,axis = 0)
        A = A.dot(np.diag(np.divide(c,q)))
        p = np.nansum(A,axis = 1)
        count+=1
    
    if count >= 10000: #4000
        print('Unable to perform Sinkhorn-Knopp projection')
    return A

def OneSinkhorn(A,r,c):
    
    p = np.nansum(A,axis = 1)
    q = np.nansum(A,axis = 0)
        
    A = np.diag(np.divide(r,p)).dot(A)
    q = np.nansum(A,axis = 0)
    A = A.dot(np.diag(np.divide(c,q)))
    p = np.nansum(A,axis = 1)
    
    return A
    
    

#Euclidean projection on U(r,c) -- Assumes that A is a square matrix
def euc_projection(A,r,c,precision):
    
    n = A.shape[0]
    m = A.shape[0]
    
    assert (m == n),"Non-square matrix in euclidean projection"
    
    p = np.sum(A,axis = 1)
    q = np.sum(A,axis = 0)
    
    H = np.matrix(np.full((n,n),1))
    
    count = 0
    
    while not (check_inf(p,q,r,c,precision)) and count <=4000:
        
        #Projection on the transportation constraints
        A = A + (np.tile(r,(n,1)).transpose() +np.tile(c,(n,1)))/n - (np.tile(p,(n,1)).transpose() + np.tile(q,(n,1)))/n + (A.sum() - 1)* H/(n*n)
        
        #Projection on the positivity constraint
        A = np.maximum(A,0)
        
        p = np.sum(A,axis = 1)
        q = np.sum(A,axis = 0)
        
        count = count +1
        
    if count >= 4000:
        print('Unable to perform Euclidean projection')
    return A

    
#Returns true iff p and q approximate respectively r and c to a 'prec' ratio in infinite norm
def check_inf(p,q,r,c,prec):
    if (np.linalg.norm(np.divide(p,r)-1,np.inf)>prec) or (np.linalg.norm(np.divide(q,c)-1,np.inf)>prec):
        return False
    else: return True

def check_2(p,q,r,c,prec):
    if (np.linalg.norm(p-r)>prec) or (np.linalg.norm(q-c)>prec):
        return False
    else: return True
    

    