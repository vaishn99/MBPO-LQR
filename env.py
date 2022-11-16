import numpy as np
import gym




class ENV:
    def __init__(self,A,B,Q,R):
        self.A=A
        self.B=B
        self.Q=Q
        self.R=R
        
        self.current_action=None
        self.current_state=None
        self.observation_space=gym.spaces.Box(low=-1000*np.ones(len(self.A)),high=1000*np.ones(len(self.A)),dtype=np.float64)
        self.action_space=gym.spaces.Box(low=-1000*np.ones(len(self.B[0])),high=1000*np.ones(len(self.B[0])),dtype=np.float64)

    def reset(self):
        self.current_state=np.ones(shape=(len(self.A),1)).T
        return self.current_state

    def step(self,action):
        mean = self.A@self.current_state.T + self.B@action
        next_state=np.random.multivariate_normal(mean.T[0],np.eye(len(mean)))
        
        
        part_1=np.array([next_state])@self.Q@np.array([next_state]).T
        # part_2=np.array([action]).T@self.R@self.array([action])
        part_2=action*self.R*action
        return next_state,part_1+part_2,False,None