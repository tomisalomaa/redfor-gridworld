import numpy as np

class MCAgent:
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

        self.locationSnapShots = []

        for x in range(self.environment.height):
            for y in range(self.environment.width):
                self.actionStateValues[(x,y)] = {}
                self.actionStateVisits[(x,y)] = {}
                self.actionStateRewards[(x,y)] = {}
                for a in self.actionSet:
                    self.actionStateValues[(x,y)][a] = 0
                    self.actionStateVisits[(x,y)][a] = 0
                    self.actionStateRewards[(x,y)][a] = 0

    # Choosing an action:
    # Compare a random float between 0 and 1 to the epsilon.
    # If lower than epsilon, pick a random action from the action set
    # to explore possibilities. Otherwise choose the current best
    # performing action from the statewise action pool.
    # If current policy has multiple best choices, choose at random
    # between said choices.
    def decideAction(self):
        if np.random.uniform(low=0,high=1) < self.epsilon:
            a = np.random.choice(self.actionSet)
        else:
            greedyValue = max(self.actionStateValues[self.environment.agentLocation].values())
            greedyActions = [k for k,v in self.actionStateValues[self.environment.agentLocation].items() if v == greedyValue]
            a = np.random.choice(greedyActions)
        return a

    def run(self):
        rewardLog = []
        iteration = 1
        for episode in range(self.episodes):
            step = 0
            terminate = False
            cumulativeReward = 0
            episodeSAR = []
            # Episode snaps are for agent action simulating purposes only;
            # allows for rendering of episodes.
            episodeStepSnaps = []
            episodeStepSnaps.append(self.environment.getAgentandSpecialLoc())
            while not terminate:
                oldState = self.environment.agentLocation # Save current state to variable
                a = self.decideAction() # Decide action; explore or follow policy
                reward = self.environment.step(a) # Calculate and save reward based on action
                episodeSAR.append((oldState,a,reward)) # Keep building the S,A,R, ... array
                cumulativeReward += reward # Add gained reward to episode total reward
                step += 1
                episodeStepSnaps.append(self.environment.getAgentandSpecialLoc())
                if self.environment.checkState() == "terminal" or step >= self.maxSteps:
                    self.predictEpisode(episodeSAR)
                    self.environment.__init__() # Re-initialize the environment
                    terminate = True
            rewardLog.append(cumulativeReward)
            self.locationSnapShots.append(episodeStepSnaps)
            print("Episode", iteration, ":", cumulativeReward)
            iteration += 1
        return rewardLog, self.locationSnapShots

class FirstVisitMCAgent(MCAgent):
    def predictEpisode(self, episodeSAR):
        uniqueStateVisits = [] # To keep book of first visits ONLY
        for idx, sar in enumerate(episodeSAR):
            if (sar[0],sar[1]) in uniqueStateVisits:
                # If the state-action pair has already been visited,
                # discard and continue.
                continue
            else:
                g = 0
                # Iterate through state->action->reward -tuples of the episode
                # starting from a unique state-action -visit and
                # calculate the cumulative reward.
                for i in range(idx, len(episodeSAR)):
                    g += episodeSAR[i][2]
                # Mark the state-action pair as visited
                uniqueStateVisits.append((sar[0],sar[1]))
                # Keep track of how many times the pair has been visited
                # over all the episodes.
                self.actionStateVisits[sar[0]][sar[1]] += 1
                # Keep track of the over all cumulative reward of the pair
                # over all the episodes.
                self.actionStateRewards[sar[0]][sar[1]] += g
                # Calculate value of the state-action -pair.
                self.actionStateValues[sar[0]][sar[1]] = self.actionStateRewards[sar[0]][sar[1]] / self.actionStateVisits[sar[0]][sar[1]]

class EveryVisitMCAgent(MCAgent):
    def predictEpisode(self, episodeSAR):
        uniqueStateVisits = []
        for idx, sar in enumerate(episodeSAR):
            if (sar[0],sar[1]) in uniqueStateVisits:
                self.actionStateVisits[sar[0]][sar[1]] += 1
                continue
            else:
                g = 0
                for i in range(idx, len(episodeSAR)):
                    g += episodeSAR[i][2]
                uniqueStateVisits.append((sar[0],sar[1]))
                self.actionStateVisits[sar[0]][sar[1]] += 1
                self.actionStateRewards[sar[0]][sar[1]] += g
                self.actionStateValues[sar[0]][sar[1]] = self.actionStateRewards[sar[0]][sar[1]] / self.actionStateVisits[sar[0]][sar[1]]