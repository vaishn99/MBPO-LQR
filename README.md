
# MBPO : A Review

## Algorithm proposed :(Simplified version)



## Algorithm proposed:(Implementations version)





# Current settings : 

0. True env : LQR (Linear dynamics with quadratic cost)<br />
1. Model is parametrised by A,B matrices.( No neaural network)<br />
2. Note that we need a policy gradient algorithm that operates on off-policy data.So chose to go with "Off-Policy policy gradient" (aka importance sampling based) <br />
3. Only Linear policies are considered. ( ie u=K*x )<br />
    - Problems encountered:<br />
        - Problems of with the structure of the distribution .Here it is dirac-delta kind of distribution.<br />      
        - More clearly ,we are dealing with deterministic policies.<br />
    - Solution:<br />
        - Go forward ..(One approach is based on the paper by David Silver,ICML 2014)<br />
        - Take a bypass ( Use gaussian policy with it's mean parametrised by linear function of state.) <br />

3. Policy gradient over linear policy setting is discussed in detail here: (https://arxiv.org/pdf/2011.10300.   pdf ,2021).
    In the paper they are discussing the the following settings:<br />
    - given model.(known parameter setting)<br />
    - we can query the env for say m trajectory for a set of "polices" .(BUT we dont have the laxury of using the trajectories originated from some other policies)<br />

    But this setting wont help us !!<br />

    Note:

    Their comparison may help us !!
4. https://arxiv.org/abs/1809.05870<br />


# In the original paper :(More complicated setting !!)

1. Modelling env : instead of a single model an ensemble model is considered. (They are citing a paper for the same)<br />
2. True env : open-AI gym env are used.<br />
2. updating the policy : SAC is used for policy optimization .<br />
3. While debugging the original code the following is noticed:<br />

    Since we are interested in how the real vs fake ratio's affect.In the program ,they are maintaining a a variable named real_ratio,but inside the program they are using this variable to adjust some of the other training parameters.But I didn't get why they are doing so. Additionally, they use this variable in some edge cases detection.

# Important note :

1. In a blog written by the first author (MBPO), he mention Dyna Algorithm(Sutton).<br />

   - The following sentence is taken that blog :<br />

    "An important detail in many machine learning success stories is a means of artificially increasing the size of a training set. It is difficult to define a manual data augmentation procedure for policy optimization, but we can view a predictive model analogously as a learned method of generating synthetic data. The original proposal of such a combination comes from the Dyna algorithm by Sutton, which alternates between model learning, data generation under a model, and policy learning using the model data. "


