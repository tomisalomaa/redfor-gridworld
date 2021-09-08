import numpy as np

class QAgent:
    def __init__(self, environment, manpower=300, episodes=1000, maxSteps=1000, epsilon=0.01, alpha=0.1, gamma=1):
        self.environment = environment
        self.actionSet = self.environment.actionSet
        self.manpower = manpower
        self.episodes = episodes
        self.maxSteps = maxSteps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Qtable = dict()

        for x in range(environment.height):
            for y in range(environment.width):
                self.Qtable[(x,y)] = {}
                for a in self.actionSet:
                    self.Qtable[(x,y)][a] = 0
    
    def decideAction(self):
        if np.random.uniform(low=0, high=1) < self.epsilon:
            a = np.random.choice(self.actionSet)
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

    def run(self):
        rewardLog = []
        iteration = 1
        for episode in range(self.episodes):
            cumulativeReward = 0
            step = 0
            terminate = False
            while step < self.maxSteps and not terminate:
                oldState = self.environment.agentLocation
                a = self.decideAction()
                reward = self.environment.step(a)
                deltaState = self.environment.agentLocation
                self.updateQtable(oldState, reward, deltaState, a)
                cumulativeReward += reward
                step += 1
                if self.environment.checkState() == "terminal":
                    self.environment.__init__()
                    terminate = True
            rewardLog.append(cumulativeReward)
            print("Episode", iteration, ":", cumulativeReward)
            iteration += 1
        return rewardLog