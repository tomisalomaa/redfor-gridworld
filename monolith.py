import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from agent import Agent
from q_agent import QAgent
from environment import Environment

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

env = Environment(show=True)

agent = Agent(env.actionSet)
rewardsAchieved = run(env, agent, episodes=500)
plt.plot(rewardsAchieved)
plt.show()

agent = QAgent(env, env.actionSet)
rewardsAchieved = run(env, agent, 500, learn=True)
plt.plot(rewardsAchieved)
plt.show()