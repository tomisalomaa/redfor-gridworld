import numpy as np

# Base class for agents. Acts as a purely stochastic agent on its own.
class Agent:
    def __init__(self, actionSet, manpower=300):
        self.manpower = manpower
        self.actionSet = actionSet
    
    def decideAction(self):
        return np.random.choice(self.actionSet)