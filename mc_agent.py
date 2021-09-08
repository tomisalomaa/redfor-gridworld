import numpy as np

class FirstVisitMCAgent:
    def __init__(self, environment, manpower=300, episodes=1000, maxSteps=1000, epsilon=0.05, alpha=0.1, gamma=1):
        self.environment = environment
        self.actionSet = self.environment.actionSet
        self.manpower = manpower
        self.episodes = episodes
        self.maxSteps = maxSteps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.actionStateValues = {}
        self.actionStateVisits = {}
        self.actionStateRewards = {}

        for x in range(self.environment.height):
            for y in range(self.environment.width):
                self.actionStateValues[(x,y)] = {}
                self.actionStateVisits[(x,y)] = {}
                self.actionStateRewards[(x,y)] = {}
                for a in self.actionSet:
                    self.actionStateValues[(x,y)][a] = 0
                    self.actionStateVisits[(x,y)][a] = 0
                    self.actionStateRewards[(x,y)][a] = 0

    def decideAction(self):
        if np.random.uniform(low=0,high=1) < self.epsilon:
            a = np.random.choice(self.actionSet)
        else:
            greedyValue = max(self.actionStateValues[self.environment.agentLocation].values())
            greedyActions = [k for k,v in self.actionStateValues[self.environment.agentLocation].items() if v == greedyValue]
            a = np.random.choice(greedyActions)
        return a

    def predictEpisode(self, episodeSAR):
        uniqueStateVisits = []
        for idx, sar in enumerate(episodeSAR):
            if (sar[0],sar[1]) in uniqueStateVisits:
                continue
            else:
                g = 0
                for i in range(idx, len(episodeSAR)):
                    g += episodeSAR[i][2]
                uniqueStateVisits.append((sar[0],sar[1]))
                self.actionStateVisits[sar[0]][sar[1]] += 1
                self.actionStateRewards[sar[0]][sar[1]] += g
                self.actionStateValues[sar[0]][sar[1]] = self.actionStateRewards[sar[0]][sar[1]] / self.actionStateVisits[sar[0]][sar[1]]

    def run(self):
        rewardLog = []
        iteration = 1
        for episode in range(self.episodes):
            step = 0
            terminate = False
            cumulativeReward = 0
            episodeSAR = []
            while not terminate:
                oldState = self.environment.agentLocation
                a = self.decideAction()
                reward = self.environment.step(a)
                episodeSAR.append((oldState,a,reward))
                cumulativeReward += reward
                step += 1
                if self.environment.checkState() == "terminal" or step >= self.maxSteps:
                    self.predictEpisode(episodeSAR)
                    self.environment.__init__()
                    terminate = True
            rewardLog.append(cumulativeReward)
            if (iteration == 1 or iteration%10 == 0):
                print("Episode", iteration, ":", cumulativeReward)
            iteration += 1
        return rewardLog