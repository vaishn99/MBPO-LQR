import numpy as np
import gym




class ENV:
    def __init__(self,A,B,Q,R,target_state=np.ones(3)):
        self.A=A
        self.B=B
        self.Q=Q
        self.R=R
        self.target_state=target_state
        self.current_action=None
        self.current_state=None

    def reset(self):
        self.current_state=np.ones(shape=(len(self.A),1)).T
        return self.current_state

    def step(self,action):
        mean = self.A@self.current_state.T + self.B@action
        next_state=np.random.multivariate_normal(mean.T[0],np.eye(len(mean)))
        part_1=np.array([next_state])@self.Q@np.array([next_state]).T
        part_2=action*self.R*action
        
        self.current_state=next_state
        self.current_action=action
        
        if next_state==self.target_state:
            return part_1+part_2,next_state,True
        return part_1+part_2,next_state,False