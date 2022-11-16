from utils import ReplayMemory
import numpy as np
from env import Environment
from scipy.linalg import lstsq

# Still in work_bench



def New_Estimate(A_hat,B_hat,Q_hat,R_hat,D_real):
    if len(D_real)==0:
        return A_hat,B_hat,Q_hat,R_hat
    dim_state=D_real[0][0].shape
    dim_action=D_real[0][1].shape
    X=np.zeros(shape=(dim_state+dim_action,len(D_real)))
    Y=np.zeros(shape=(dim_state,len(D_real)))
    for i in range(len(D_real)):
        Y[:,i]=D_real[i][2]
        X[:,i]=np.concatenate(D_real[i][0],D_real[i][1])
    A=X.T
    b=Y.T
    total_hat=lstsq(A,b) # need to split
    return total_hat

def Sample_state(env_pool):
    space=[x[0] for x in env_pool] 
    return np.random.choice(space)

def gradient_with_model(A,B):
    
    grad=4
    return grad 

def gradient_with_exp(D_fake):
    
    grad=4
    return grad

def get_traj(S_t,horiz_len,A_hat,B_hat,Q_hat,R_hat,K_t):
    i=0
    holder=[]
    while i<horiz_len:
        prev=S_t
        u_T=K*S_t
        S_t=A_hat*S_t+B_hat*u_T+np.random.normal(0,1)
        R_t=0.5*[S_t]@Q_hat@[S_t].T + 0.5*[u_T]@R_hat@[u_T].T
        done_t=False
        if S_t==np.zeros_like(S_t):
            done_t=True
        holder.append((prev,u_T,S_t,R_t,done_t))
        i+=1
    return holder



# A ->>3*3
# B ->>3*3
# C->eye(3)
# K ->>3*3



np.random.seed(0)

# True parameters of the env

A=np.diag([1,-2,3])
B=np.diag([1,2,3])
C=np.eye(3)
Q=np.diag([1,2,3])
R=np.diag([5,1,3])



A_hat=np.random.rand(3,3)   # Initial theta
B_hat=np.random.rand(3,3)   # Initial theta
Q_hat=np.diag(np.ones(3))   # Initial theta
R_hat=np.diag(np.ones(3))   # Initial theta

K=np.random.rand(3,3)   # Initial phi
K_t=K

D_real = ReplayMemory()   # Real data
D_fake = ReplayMemory()  # Fake data


env=Environment(A,B,Q,R)

################################################################

num_of_epcoh=20
N=10
E=10
M=10
G=10
horiz_len=10
num_of_rollouts=10

################################################################

for n in range(num_of_epcoh):
    A_hat,B_hat,Q_hat,R_hat=New_Estimate(A_hat,B_hat,Q_hat,R_hat,D_real) # Regression 
    for e in range(E):
        S_t=np.random.rand(len(A_hat)) # Update D_real
        
        
        for m in range(M):
            S_t=Sample_state(D_real)    # Random sampling
            D_fake.push(get_traj(S_t,horiz_len,A_hat,B_hat,K_t))
            # Update D_fake with horiz_len
        for g in range(G):
            K_t+=gradient_with_model(A_hat,B_hat)    # Known parameter
            K_t+=gradient_with_exp(D_fake)    # unKnown parameter (From trajectories -off policy settings)
        
            
    
    
    