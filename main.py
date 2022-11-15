from utils import ReplayMemory,New_Estimate,Sample_state,gradient_At
import numpy as np
from env import Environment




# A ->>3*3
# B ->>3*3
# C->eye(3)
# K ->>3*3

np.random.seed(0)

A=np.random.rand(3,3)   # Initial theta
B=np.random.rand(3,3)   # Initial theta


K=np.random.rand(3,3)   # Initial phi
K_t=K

D_real = ReplayMemory()   # Real data
D_fake = ReplayMemory()  # Fake data

num_of_epcoh=20
N=10
E=10
M=10
G=10
horiz_len=10
num_of_rollouts=10

for n in range(num_of_epcoh):
    A,B=New_Estimate(A,B,D_real) # Regression 
    for e in range(E):
        # update D_env
        for m in range(M):
            S_t=Sample_state(D_real)    # Random sampling
            # Update D_fake with horiz_len
        for g in range(G):
            K_t+=gradient_At(A,B,D_fake)    # Known parameter
            K_t+=gradient_At(A,B,D_fake)    # unKnown parameter (From trajectories)
        
            
    
    
    