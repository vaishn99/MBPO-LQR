
# MBPO : A Review

## Objective : <br/>
To simulate MBPO algorithm for LQR setting(ie linear dynamics and quadratic cost).<br/>

## Algorithm proposed :(Simplified version)
<img width="583" alt="Screenshot 2022-11-17 at 12 07 22 AM" src="https://user-images.githubusercontent.com/113635391/202267312-78099037-df2e-4f5a-8b32-37feb8cb9192.png">



## Algorithm proposed:(Implementations version)

<img width="579" alt="Screenshot 2022-11-17 at 12 07 33 AM" src="https://user-images.githubusercontent.com/113635391/202267366-fd1939e5-68d9-4440-a84b-e7cd288af667.png">

## Intuition :

MBPO optimizes a policy under a learned model, collects data under the updated policy, and uses that data to train a new
model.



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

# An important note :

1. In a blog written by the first author (MBPO), he mention Dyna Algorithm(Sutton).<br />

   - The following sentence is taken that blog :<br />

    "An important detail in many machine learning success stories is a means of artificially increasing the size of a training set. It is difficult to define a manual data augmentation procedure for policy optimization, but we can view a predictive model analogously as a learned method of generating synthetic data. The original proposal of such a combination comes from the Dyna algorithm by Sutton, which alternates between model learning, data generation under a model, and policy learning using the model data. "


# Updates on implementation:

- Done with the implementation for MBPO algorithm for linear setting.
- Tested Model updation section,Real and Fake data generation ,things are working fine.
- But there is a problem with the gradient update rule (Policy gradient is used))
- We need to working in the following settings
    - Linear policy (Dirac delta distribution)
    - Off policy setting (Note that there is an important sampling term)
- To handle the importance sampling term ,initialy planned to use the following strategy : use a  gaussian policy with mean equal to K@x term and with a fixed covariance matrix.The derivation and the expression for the gradient term is in the "main.ipynb" file.
- As a sanity check,I used the following test.

- In short MBPO, proposes the following approach for finding the optimal policy.Use fake data for performing policy gradient instead of real data.

## The vanilla policy gradient approach :<br />
- Use the Real data to construct a cost function( which is a function of the policy),
- Use the gradient descent to do the minimisation step.
### The drawback:
- we need a lot of real data(trajectories).

## MBPO proposes the following strategy:

- Use the real data to construct a model 
- generate fake data(trajectories) using this model
- Do the policy gradient step using this fake data.

## The Current problem :

- MBPO uses the fake data to update the policy,but instead I will use the real data to upate the policy.It should gradually converge to the optimal policy (which we know in LQR case).But it observed that the " update rule " obtained with the "gaussian hack" is not converging to the optimal policy.Here "gaussian hack" refers to using a gaussian policy for preventing numerically exploding.I have used comments in the script to denote what is going on for each term.
- The problem  that I'm facing now:
    For this particular setting "Linear policy" and "off policy trajectories",need to get an expression for grad to carry out the policy updation rule.

