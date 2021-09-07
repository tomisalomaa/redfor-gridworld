import numpy as np

class Environment:
    def __init__(self, height=10, width=10, show=False):
        # grid dimensions
        self.height = height
        self.width = width
        self.show = show
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
        self.grid[self.blueforLocs[0],self.blueforLocs[1]] = -100
        self.grid[self.targetAreaLoc[0],self.targetAreaLoc[1]] = 10
        # define actions
        self.actionSet = ["UP", "RIGHT", "DOWN", "LEFT", "ATTACK"]

    def getActionSet(self):
        return self.actionSet

    def getAgentandSpecialLoc(self):
        # This will be for drawing the image later
        grid = np.zeros((self.height, self.width))
        grid[self.agentLocation[0], self.agentLocation[1]] = 1
        grid[self.blueforLocs[0], self.blueforLocs[1]] = 2
        grid[self.targetAreaLoc[0], self.targetAreaLoc[1]] = 3
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