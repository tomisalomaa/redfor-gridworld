import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from agent import Agent
from q_agent import QAgent
from mc_agent import FirstVisitMCAgent
from environment import Environment

env = Environment()

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle("Agent score: Random, Q-learning, MC first visit")

agent = Agent(environment=env, episodes=5000)
rewardsAchieved = agent.run()
ax1.plot(rewardsAchieved)

agent = QAgent(environment=env, episodes=5000, epsilon=0.01)
rewardsAchieved = agent.run()
ax2.plot(rewardsAchieved)

agent = FirstVisitMCAgent(environment=env, episodes=5000, epsilon=0.01)
rewardsAchieved = agent.run()
ax3.plot(rewardsAchieved)

plt.savefig("test_runs/5000_episodes", bbox_inches="tight")