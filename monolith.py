import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from q_agent import QAgent

class Environment:
    def __init__(self, height=10, width=10):
        # grid dimensions
        self.height = height
        self.width = width
        # init grid with zeros
        self.grid = np.zeros((self.height, self.width))
        # set redfor agent start location
        self.agentLocation = (9, np.random.randint(0,9))
        # set bluefor starting location(s)
        self.blueforLocs = (np.random.randint(1,6), np.random.randint(1,6))
        # set the redfor target location
        self.targetAreaLoc = (0,5)
        # define terminal states
        self.terminalStates = [self.targetAreaLoc]
        # define rewards for special states
        self.grid[self.blueforLocs[0],self.blueforLocs[1]] = -10
        self.grid[self.targetAreaLoc[0],self.targetAreaLoc[1]] = 10
        # define actions
        self.actionSet = ["UP", "RIGHT", "DOWN", "LEFT", "ATTACK"]

    def getActionSet(self):
        return self.actionSet

    def getAgentLoc(self):
        grid = np.zeros((self.height, self.width))
        grid[self.agentLocation[0], self.agentLocation[1]] = 1
        return grid

    def getReward(self, locDelta):
        return self.grid[locDelta[0], locDelta[1]]

    def step(self, a):
        lastLoc = self.agentLocation
        if a == "UP":
            if lastLoc[0] == 0:
                reward = self.getReward(lastLoc)
            else:
                self.agentLocation = (self.agentLocation[0]-1, self.agentLocation[1])
                reward = self.getReward(self.agentLocation)
        elif a == "DOWN":
            if lastLoc[0] == self.height-1:
                reward = self.getReward(lastLoc)
            else:
                self.agentLocation = (self.agentLocation[0]+1, self.agentLocation[1])
                reward = self.getReward(self.agentLocation)
        elif a == "LEFT":
            if lastLoc[1] == 0:
                reward = self.getReward(lastLoc)
            else:
                self.agentLocation = (self.agentLocation[0], self.agentLocation[1]-1)
                reward = self.getReward(self.agentLocation)
        elif a == "RIGHT":
            if lastLoc[1] == self.width-1:
                reward = self.getReward(lastLoc)
            else:
                self.agentLocation = (self.agentLocation[0], self.agentLocation[1]+1)
                reward = self.getReward(self.agentLocation)
        elif a == "ATTACK":
            # For now, later the Lanchester model needs to be implemented
            reward = -1
        return reward

    def checkState(self):
        if self.agentLocation in self.terminalStates:
            return "terminal"

def run(environment, agent, episodes=1000, maxSteps=1000, learn=False):
    rewardLog = []
    iteration = 1
    for episode in range(episodes):
        cumulativeReward = 0
        step = 0
        terminate = False
        while step < maxSteps and not terminate:
            oldState = environment.agentLocation
            a = agent.decideAction(environment.actionSet)
            reward = environment.step(a)
            deltaState = environment.agentLocation
            if learn:
                agent.updateQtable(oldState, reward, deltaState, a)
            cumulativeReward += reward
            step += 1
            if environment.checkState() == "terminal":
                environment.__init__()
                terminate = True
        rewardLog.append(cumulativeReward)
        print("Episode", iteration, ":", cumulativeReward)
        iteration += 1
    return rewardLog

env = Environment()
#agent = Agent()
qAgent = QAgent(env, env.actionSet)
#rewardsAchieved = run(env, agent, 100)
rewardsAchieved = run(env, qAgent, 10, learn=True)
#plt.plot(rewardsAchieved)
#plt.show()