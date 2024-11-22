'''
Creator: Jonas Bresch, M.Sc.
Project: Renyi-regularized optimal transport

Date: April 24th, 2024
'''

import numpy as np
import matplotlib.pyplot as plt
import ot
import ot.plot
import math
from copy import deepcopy
from Projections import check_inf, check_2, Sinkhorn, OneSinkhorn
from Distances import inner_Frobenius


###### HELP-FUNCTIONS ######

def gen_P(n):

    r = np.random.uniform(n)
    c = np.random.uniform(n)

    r = r/np.sum(r)
    c = c/np.sum(c)

    P = np.random.uniform(0,1,(n,n))

    P = P/np.sum(P)

    c = P@np.ones(n)
    r = P.T@np.ones(n)

    return [P, c, r]

def renyi_entropy(x, a):

    s = 1/(1-a)*np.log(np.sum(x**a))

    return s

def renyi_divergence(x, y, a):

    s = 1/(a - 1)*np.log(np.sum(x**a*y**(1-a)))

    return s

def shannon_entropy(x):

    s = -np.sum(x*np.log(x))

    return s

def op_L(P):

    n,n = np.shape(P)

    p = P@np.ones(n)

    return p

def op_Lt(P):

    n,n = np.shape(P)

    p = P.T@np.ones(n)

    return p

def adj_op_L(u):

    n = np.size(u)

    U = np.zeros((n,n))

    for i in range(n):
        U[:,i] = np.squeeze(u.T)
    
    return U

def adj_op_Lt(u):

    n = np.size(u)

    U = np.zeros((n,n))

    for i in range(n):
        U[i,:] = np.squeeze(u)
    
    return U

##### ALGORITHMS #####

def grad_Renyi_conj(X, Q, alp):

    S = np.abs(X)**(alp/(alp-1))*Q

    s = np.sum(S)

    adjR = 1/s*(alp/(alp-1))*np.abs(X)**(1/(alp-1))*Q
    
    return -adjR

def rank1_prod(x, y):

    P = adj_op_L(x)*adj_op_Lt(y)

    return P

def LineSearch(x, X, a, tet, bet, lam, alp, M, r, c):

    d = np.size(r)

    t = a
    rc = rank1_prod(r,c)

    xx = lam*(adj_op_L(-x[0:d])+adj_op_Lt(-x[d:2*d]) - (M - X))

    y = x - t*(-np.append(op_L(grad_Renyi_conj(xx, rc, alp)), op_Lt(grad_Renyi_conj(xx, rc, alp))) + np.append(r, c))

    Y = np.maximum(0, X - t*grad_Renyi_conj(xx, rc, alp))

    yy = lam*(adj_op_L(-y[0:d])+adj_op_Lt(-y[d:2*d]) - (M - Y))

    while t*np.sqrt(np.linalg.norm(np.append(op_L(grad_Renyi_conj(yy, rc, alp)), op_Lt(grad_Renyi_conj(yy, rc, alp))) - np.append(op_L(grad_Renyi_conj(xx, rc, alp)), op_Lt(grad_Renyi_conj(xx, rc, alp))))**2 + np.linalg.norm(grad_Renyi_conj(yy, rc, alp) - grad_Renyi_conj(xx, rc, alp))**2) > bet*(np.linalg.norm(y - x,1)+np.linalg.norm(Y - X,1)):
        t = tet*t
    
    return t

def StepSearch( x, X, a, tet, bet, lam, alp, M, r, c):

    d = np.size(r)

    t = LineSearch(x, X, a, tet, bet, lam, alp, M, r, c)
    
    rc = rank1_prod(r,c)

    xx = lam*(adj_op_L(-x[0:d])+adj_op_Lt(-x[d:2*d]) - (M - X))

    y = x - t*(-np.append(op_L(grad_Renyi_conj(xx, rc, alp)), op_Lt(grad_Renyi_conj(xx, rc, alp))) + np.append(r, c))

    Y = np.maximum(0, X - t*grad_Renyi_conj(xx, rc, alp))

    return [y, Y]


####### SUBGRADIENT-METHOD #######

def inner_rc(q, r, c):
    return np.sum(q*np.append(r, c))

def renyi_div_conj(X, alp, r, c):

    QQ = rank1_prod(r, c)

    b = 1 + 1/(alp-1)

    S = np.abs(X)**b*QQ
    s = np.sum(S)

    a = (alp*(1+np.log((1-alp)/alp)))/(1-alp)

    if np.all(X <= 0):
        return np.log(s) - a
    else:
        return +np.inf

def mat_val(q, G, M, lam):

    d = int(np.size(q)/2)

    Y = lam*(adj_op_L(-q[0:d]) + adj_op_Lt(-q[d:2*d]) - (M - G))

    return Y

def func(Q, r, c, alp, lam, M):

    d = np.size(r)

    q = Q[0:2*d]

    G = np.reshape(Q[2*d:(2+d)*d], (d,d))
    
    Y = mat_val(q, G, M, lam)

    return inner_rc(q, r, c) + 1/lam*renyi_div_conj(Y, alp, r, c)

def funcEnt(q, r, c, alp, lam, M):

    d = np.size(r)
    
    Y = lam*(M - adj_op_L(q[0:d]) + adj_op_Lt(q[d:2*d]))

    return inner_rc(q, r, c) + 1/lam*renyi_div_conj(Y, alp, np.ones(d), np.ones(d))

def jac_renyi_div_conj(X, r, c, alp):

    QQ = rank1_prod(r, c)

    b = 1 + 1/(alp-1)

    S = np.abs(X)**b*QQ
    s = np.sum(S)

    c = -b

    J = 1/s*c*np.abs(X)**(1/(alp-1))*QQ

    return J

def jac_func(Q, r, c, alp, lam, M):

    d = np.size(r)

    q = Q[0:2*d]

    G = np.reshape(Q[2*d:(2+d)*d], (d,d))

    Y = mat_val(q, G, M, lam)

    jac_q1 = jac_renyi_div_conj(Y, r, c, alp)

    jac_q = np.append(r,c) - np.append(op_L(jac_q1), op_Lt(jac_q1))

    jac_G1 = -jac_renyi_div_conj(Y, r, c, alp)

    jac_G = jac_G1.flatten()

    return np.append(jac_q, jac_G)

def SubGradDes(r, c, lam, alp, M, iter):

    d = np.size(r)

    F = []

    #q = -np.ones(2*d)
    q = np.append(-np.ones(d), np.ones(d)*0.5)

    t = 0.5

    qlist = []

    if np.max(mat_val(-q, np.zeros((d,d)), M, lam)) >= 0:
        print('not in domain')

    print('F \t\t| step \t\t| q-res \t| update')
    print('--------------------------------------------------------------')

    for i in range(iter):

        Y = mat_val(-q, np.zeros((d,d)), M, lam)
        if np.max(Y) >= 0:
            print('not in domain')

        jac_q1 = jac_renyi_div_conj(Y, r, c, alp)

        jac_q = -np.append(r,c) + np.append(op_L(jac_q1), op_Lt(jac_q1))

        #jac_Q = -jac_renyi_div_conj(Y, r, c, alp)

        a = 10
        
        R = np.zeros(d*(d+2))
        R[0:2*d] = q
        R[2*d:d*(d+2)] = np.zeros(d*d)


        R1 = np.zeros(d*(d+2))
        R1[0:2*d] = q - a*jac_q
        R1[2*d:d*(d+2)] = np.zeros(d*d)

        #while np.max(mat_val(-(q - a*jac_q), np.zeros((d,d)), M, lam)) >= 0 or func(-R, r, c, alp, lam, M) < func(-R1, r, c, alp, lam, M):
        while func(-R, r, c, alp, lam, M) < func(-R1, r, c, alp, lam, M):
            a = a*t
            R1 = np.zeros(d*(d+2))
            R1[0:2*d] = q - a*jac_q
            R1[2*d:d*(d+2)] = np.zeros(d*d)
        
        q1 = q
        q = q - a*jac_q

        R = np.zeros(d*(d+2))
        R[0:2*d] = q
        R[2*d:d*(d+2)] = np.zeros(d*d)

        F = np.append(F, func(-R, r, c, alp, lam, M))

        if i > 0:
            if F[i] - F[i-1] < 0 and F[i] == np.min(F):
                qbest = q
                flag = 1
        
        qlist = np.append(qlist, q)

        if np.mod(i,1000) == 0:
            if i == 0:
                flag = np.nan
                
            print("%10.3e"% F[i] , '\t|', "%10.3e"% a, '\t|', "%10.1e"% np.linalg.norm(q - q1), '\t|', flag)
            flag = 0
    
    return [q, F, qbest, qlist]

####### MIRROR-DESCENT #######

def grad_F_alp(P, M, alp, lam, r, c):
    R = rank1_prod(r,c)
    
    #Q = np.where(np.isnan(1/P)==True, 0, 1/P)
    Q = np.where(P==0, 0, 1/P)

    Inf_alp = np.sum(np.multiply(P**alp,R**(1-alp)))

    return M + 1/lam*alp/(alp-1)*1/Inf_alp*np.multiply(Q**(1-alp),R**(1-alp))

def alp_obj(alp, P, M, lam, r, c):
    R = rank1_prod(r,c)

    return np.sum(P*M) + 1/lam*1/(alp-1)*np.log(np.sum(P**alp*R**(1-alp)))

'''
    mainly inspired by Muzellec, Boris 
'''

def KLprojMirrorDescent(alp,M,r,c,lam,precision, precision_sink,T , rate = None, rate_type = None, a = None):
    
    if (alp>=1): print("Warning: Rényi-divergence not convex")    
    
    omega = math.sqrt(2*np.log(M.shape[0]))
    ind_map = np.asarray(np.matrix(r).transpose().dot(np.matrix(c)))

    P = np.zeros((np.size(r),np.size(c)),dtype='float128')

    P = rank1_prod(r,c)
    P1 = np.random.randn(np.shape(P)[0],np.shape(P)[1])

    new_score = alp_obj(alp, P, M, lam, r, c)
    scores = []
    scores.append(new_score)
    
    best_score = new_score
    best_P = P

    p_min = []

    print('iteration \t| func-value \t| residuum \t| update \t| stepsize ')
    print('---------------------------------------------------------------------')
    print(0 ,'\t\t|', "%10.3e"%alp_obj(alp, P, M, lam, r, c), '\t|', np.nan, '\t\t|', np.nan)
    
    count = 1    
    
    while count<=T: #and np.linalg.norm(P - P1)>1e-6: 

        P = np.where(np.isnan(P)==True, 0, P)
        P = np.where(np.isinf(P)==True, 0, P)
        
        G = grad_F_alp(P, M, alp, lam, r, c)        
        G = np.where(np.isnan(G)==True, 0, G)
        
        if rate is None:
            tmp = np.exp(G*(-omega/(math.sqrt(T)*np.linalg.norm(G,np.inf)))) #Absolute horizon
            #tmp = np.exp(G*(-omega/(math.sqrt(count)*np.linalg.norm(G,np.inf)))) #Rolling horizon
        elif rate_type == "constant":
            tmp = np.exp(G*(-rate))
            rrate = rate
        elif rate_type == "constant_plus_P":
            tmp = np.exp((G+P)*(-rate))
            rrate = rate
        elif rate_type == "constant_length":
            tmp = np.exp(G*(-rate*np.linalg.norm(G,np.inf)))
            rrate = rate*np.linalg.norm(G,np.inf)
        elif rate_type == "diminishing":
            tmp = np.exp(G*(-rate/math.sqrt(count)))
            rrate = rate/math.sqrt(count)
        elif rate_type == "square_summable":
            tmp = np.exp(G*(-rate/count))
            rrate = rate/count
        elif rate_type == "square_summable_a":
            if a is None:
                a = 1
                tmp = np.exp(G*(-rate/(a+count)))
                rrate = rate/(a+count)
            else: 
                tmp = np.exp(G*(-rate/(a+count)))
                rrate = rate/(a+count)
        
            
        X = np.multiply(P,tmp) 
        P1 = P.copy()
        P = Sinkhorn(X,r,c,precision_sink)
        #P = OneSinkhorn(X,r,c)
        #P = ot.bregman.sinkhorn_stabilized(r, c, G.copy(), 1/rate)

        if np.min(P[P>0]) < 1e-30:
            break

        P = np.where(np.isnan(P)==True, 0, P)
        P = np.where(np.isinf(P)==True, 0, P)

        p_min = np.append(p_min, np.min(P[P>0]))

        #Update score list
        new_score = alp_obj(alp, P, M, lam, r, c)
        scores.append(new_score)  
        
        up = 0
        #Keep track of the best solution so far
        if (new_score < best_score):
            best_score = new_score
            best_P = P
            up = 1
        
        if np.mod(count, 100) == 0:
            print(count, '\t\t|', "%10.3e"%new_score, '\t|', "%10.3e"%np.linalg.norm(P - P1) ,'\t|', up ,'\t\t|', "%10.3e"%rrate, '\t|', "%10.3e"%np.sum(np.multiply(P,np.log(np.divide(P,rank1_prod(r,c))))))

        if np.linalg.norm(P - P1) < precision:
            break

        count+=1

    return best_P, P, scores, p_min

def KLprojMirrorDescentPolyak(alp,M,r,c,lam,precision,precision_sink,T,d1,B,cc):

    sig1=0
    l=1

    d = np.zeros(2*T+1)
    d[1] = d1

    k = np.zeros(2*T+1)
    k[1] = 1

    f_rec = np.zeros(2*T+1)
    
    if (alp>=1): print("Warning: Rényi-divergence not convex")    

    P = np.zeros((np.size(r),np.size(c)),dtype='float128')

    P = rank1_prod(r,c)
    P1 = np.random.randn(np.shape(P)[0],np.shape(P)[1])

    new_score = alp_obj(alp, P, M, lam, r, c)
    scores = []
    scores.append(new_score)
    
    best_score = new_score
    best_P = P


    print('iteration \t| func-value \t| residuum \t| update \t| stepsize ')
    print('--------------------------------------------------------------------------------------')
    print(0 ,'\t\t|', "%10.3e"%alp_obj(alp, P, M, lam, r, c), '\t|', np.nan, '\t\t|', np.nan)

    ####

    counter = 1

    while counter <= T:

        P = np.where(np.isnan(P)==True, 0, P)
        P = np.where(np.isinf(P)==True, 0, P)
        
        G = grad_F_alp(P, M, alp, lam, r, c)        
        G = np.where(np.isnan(G)==True, 0, G)
        G = np.where(np.isinf(G)==True, 0, G)


        # Polyak step size calculations 
        f_rec[counter] = best_score

        if alp_obj(alp, P, M, lam, r, c) <= f_rec[int(k[l])] + 1/2*d[l]:

            k[l+1] = counter
            sig = 0
            d[l+1] = l
            l = l+1
        if sig > B:
            k[l+1] = counter
            sig = 0
            d[l+1] = 1/2*d[l]
            l = l+1
        
        ff = f_rec[int(k[l])] - d[l]
        # eta = (new_score - ff)/(cc*np.linalg.norm(G)**2)
        eta = (new_score - ff)/(cc*np.max(np.abs(G))**2)

        tmp = np.exp(-eta*G)
        X = np.multiply(P,tmp)
        P1 = P.copy()
        P = Sinkhorn(X,r,c,precision_sink)
        # P = OneSinkhorn(X,r,c)
        # P = ot.bregman.sinkhorn_stabilized(r, c, G.copy(), 1/eta)

        P = np.where(np.isnan(P)==True, 0, P)
        P = np.where(np.isinf(P)==True, 0, P)

        # sig = sig + cc*eta*np.linalg.norm(G)
        sig = sig + cc*eta*np.max(np.abs(G))

        # Update score list
        new_score = alp_obj(alp, P, M, lam, r, c)
        scores.append(new_score)  
        
        up = 0
        # Keep track of the best solution so far
        if (new_score < best_score):
            best_score = new_score
            best_P = P
            up = 1
        
        if np.mod(counter, 100) == 0:
            print(counter, '\t\t|', "%10.3e"%new_score, '\t|', "%10.3e"%np.linalg.norm(P - P1) ,'\t|', up ,'\t\t|', "%10.3e"%eta)

        if np.linalg.norm(P - P1) < precision:
            break

        counter+=1

    return best_P, P, scores

####### SAVE-FILE-DUAL #######

def saverenyi(r, c, M, q, F, alp, lam, save, cut):

    d = np.size(r)
    p1 = q[0:d]
    p2 = q[d:2*d]

    if cut == 1:
        C = (np.minimum(0,M - adj_op_L(p1) - adj_op_Lt(p2)))**(1/(alp-1))*rank1_prod(r,c)
    else:
        C = (M - adj_op_L(p1) - adj_op_Lt(p2))**(1/(alp-1))*rank1_prod(r,c)

    P = C/np.sum(C)

    fig = plt.figure(figsize=(10,2))
    ax1 = fig.add_subplot(141)
    ax1.plot(op_L(P))
    ax1.plot(r)
    ax1.plot(op_Lt(P))
    ax1.plot(c)
    ax1.set_title('for/backward-OP')

    ax2 = fig.add_subplot(142)
    ax2.plot(op_L(P) - r, np.arange(0,d))
    ax2.plot(op_Lt(P) - c, np.arange(0,d))
    ax2.set_title('residuum')

    #print(np.linalg.norm(op_L(P) - r), np.linalg.norm(op_Lt(P) - c))

    ax3 = fig.add_subplot(143)
    ax3.set_title('reg-OT-plan')
    #ot.plot.plot1D_mat(r, c, P, 'Renyi-Reg-OT')
    ax3.imshow(P)

    ax4 = fig.add_subplot(144)
    ax4.set_title('func-value')
    ax4.semilogx(F)
    ax4.semilogx(np.zeros(np.size(F)),'--', color='gray', linewidth=0.5)
    fig.tight_layout()
    if save == 1:
        plt.savefig('mix_Gaussian_Renyi_reg_OT_alp=%s_lam=%s.pdf' % (alp ,lam), dpi=300)

    return P

############

#Squared Euclidean distance cost matrix
def sqeuc_costs(n,scale):
    indices = np.arange(n) / n
    return scale * (indices[:, None] - indices[None, :])**2

def euc_costs(n,scale):
    indices = np.arange(n) / n
    return scale * np.sqrt((indices[:, None] - indices[None, :])**2)
