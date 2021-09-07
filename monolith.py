import numpy as np
import matplotlib.pyplot as plt

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

class StochasticAgent():
    def decideAction(self, actionSet):
        return np.random.choice(actionSet)

class QAgent():
    def __init__(self, environment, epsilon=0.01, alpha=0.1, gamma=1):
        self.environment = environment
        self.Qtable = dict()
        for x in range(environment.height):
            for y in range(environment.width):
                self.Qtable[(x,y)] = {"UP":0, "RIGHT":0, "DOWN":0, "LEFT":0, "ATTACK":0}
                self.epsilon = epsilon
                self.alpha = alpha
                self.gamma = gamma
    
    def decideAction(self, actionSet):
        if np.random.uniform(0,1) < self.epsilon:
            a = actionSet[np.random.randint(0, len(actionSet))]
        else:
            QValues = self.Qtable[self.environment.agentLocation]
            greedyValue = max(QValues.values())
            possibleActions = [k for k,v in QValues.items() if v == greedyValue]
            a = np.random.choice(possibleActions)
        return a

    def updateQtable(self, oldState, reward, deltaState, action):
        QValues = self.Qtable[deltaState]
        maxQinDelta = max(QValues.values())
        presentQvalues = self.Qtable[oldState][action]
        self.Qtable[oldState][action] = (1-self.alpha)*presentQvalues+self.alpha*(reward+self.gamma*maxQinDelta)

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

#env = Environment()
#agent = StochasticAgent()
#Qagent = QAgent(env)
#rewardsAchieved = run(env, agent, 5000)
#rewardsAchieved = run(env, Qagent, 5000, learn=True)
#plt.plot(rewardsAchieved)
#plt.show()