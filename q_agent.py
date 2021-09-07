import numpy as np

class QAgent:
    def __init__(self, environment, actionSet, manpower=300, epsilon=0.01, alpha=0.1, gamma=1):
        self.environment = environment
        self.actionSet = actionSet
        self.manpower = manpower
        self.Qtable = dict()
        self.QtableInitKeyValues = dict()
        for a in actionSet:
            self.QtableInitKeyValues[a] = 0
        for x in range(environment.height):
            for y in range(environment.width):
                self.Qtable[(x,y)] = self.QtableInitKeyValues
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