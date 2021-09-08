import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from agent import Agent
from q_agent import QAgent
from environment import Environment

env = Environment()

# agent = Agent(env)
# rewardsAchieved = agent.run()
# plt.plot(rewardsAchieved)
# plt.show()

agent = QAgent(env)
rewardsAchieved = agent.run()
plt.plot(rewardsAchieved)
plt.show()