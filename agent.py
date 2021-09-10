import numpy as np

# Base class for agents. Acts as a purely stochastic agent.
class Agent:
    def __init__(self, environment, manpower=300, episodes=1000, maxSteps=1000):
        self.environment = environment
        self.manpower = manpower
        self.episodes = episodes
        self.maxSteps = maxSteps
        self.actionSet = self.environment.actionSet
    
    def decideAction(self):
        return np.random.choice(self.actionSet)

    def run(self):
        rewardLog = []
        iteration = 1
        for episode in range(self.episodes):
            cumulativeReward = 0
            step = 0
            terminate = False
            while not terminate:
                oldState = self.environment.agentLocation
                a = self.decideAction()
                reward = self.environment.step(a)
                deltaState = self.environment.agentLocation
                cumulativeReward += reward
                step += 1
                if self.environment.checkState() == "terminal" or step >= self.maxSteps:
                    self.environment.__init__()
                    terminate = True
            rewardLog.append(cumulativeReward)
            print("Episode", iteration, ":", cumulativeReward)
            iteration += 1
        return rewardLog