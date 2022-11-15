

# Algorithm proposed :




# 





# Current settings :

0. True env : LQR.
1. Model is parametrised by A,B matrices.( No neaural network)
2. Only Linear policies are considered. ( ie u=K*x )
3. Policy gradient over linear policy setting follows the paper (https://arxiv.org/pdf/2011.10300.pdf ,2021)
    - given model.(known parameter setting)
    - given experience (Trying with Fake data approach)(unknwon parameter setting ())

    Note:
    Their comparison may help us !!
4. https://arxiv.org/abs/1809.05870


# In the original paper :(More complicated setting !!)

1. Modelling env : instead of a single model an ensemble model is considered. (They are citing a paper for the same)
2. True env : open-AI gym env are used.
2. updating the policy : SAC is used for policy optimization .
3. While debugging the original code the following is noticed:

    Since we are interested in how the real vs fake ratio's affect.In the program ,they are maintaining a a variable named real_ratio,but inside the program they are using this variable to adjust some of the other training parameters.But I didn't get why they are doing so. Additionally, they use this variable in some edge cases detection.

# Important note :

1. In a blog written by the first author (MBPO), he mention Dyna Algorithm(Sutton).

   - The following sentence is taken that blog :

    "An important detail in many machine learning success stories is a means of artificially increasing the size of a training set. It is difficult to define a manual data augmentation procedure for policy optimization, but we can view a predictive model analogously as a learned method of generating synthetic data. The original proposal of such a combination comes from the Dyna algorithm by Sutton, which alternates between model learning, data generation under a model, and policy learning using the model data. "

# Now :
