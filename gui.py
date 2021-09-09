import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk, IntVar
from PIL import Image
from agent import Agent
from q_agent import QAgent
from mc_agent import FirstVisitMCAgent
from environment import Environment

class Window:
    def __init__(self,root):
        self.root = root
        self.root.title("REDFOR REINFORCEMENT LEARNING")
        self.root.geometry("600x400")
        self.root.resizable(True,True)

        agents = ["Choose agent", 
                    "Random", 
                    "MC first visit", 
                    "MC every visit", 
                    "Q-learning", 
                    "SARSA"]

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=10)

        episodesLabel = ttk.Label(self.root, text="Episodes:")
        episodesLabel.grid(column=0, row=0, sticky=tk.W)
        episodeEntry = ttk.Entry(self.root)
        episodeEntry.grid(column=1, row=0, sticky=tk.W)

        stepsLabel = ttk.Label(self.root, text="Steps:")
        stepsLabel.grid(column=0, row=1, sticky=tk.W)
        stepsEntry = ttk.Entry(self.root)
        stepsEntry.grid(column=1, row=1, sticky=tk.W)

        epsilonLabel = ttk.Label(self.root, text="Epsilon:")
        epsilonLabel.grid(column=0, row=2, sticky=tk.W)
        epsilonEntry = ttk.Entry(self.root)
        epsilonEntry.grid(column=1, row=2, sticky=tk.W)

        alphaLabel = ttk.Label(self.root, text="Alpha:")
        alphaLabel.grid(column=0, row=3, sticky=tk.W)
        alphaEntry = ttk.Entry(self.root)
        alphaEntry.grid(column=1, row=3, sticky=tk.W)

        gammaLabel = ttk.Label(self.root, text="Gamma:")
        gammaLabel.grid(column=0, row=4, sticky=tk.W)
        gammaEntry = ttk.Entry(self.root)
        gammaEntry.grid(column=1, row=4, sticky=tk.W)

        agentLabel = ttk.Label(self.root, text="Agent:")
        agentLabel.grid(column=0, row=5, sticky=tk.W)
        var = tk.StringVar(self.root)
        ttk.OptionMenu(self.root, var, *agents).grid(
                                                    column=1,
                                                    row=5, 
                                                    sticky=tk.W)
        
def main():
    root = tk.Tk()
    gui = Window(root)
    gui.root.mainloop()
    return None

main()

# env = Environment()

# fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.suptitle("Agent score: Random, Q-learning, MC first visit")

# agent = Agent(environment=env, episodes=5000)
# rewardsAchieved = agent.run()
# ax1.plot(rewardsAchieved)

# agent = QAgent(environment=env, episodes=5000, epsilon=0.01)
# rewardsAchieved = agent.run()
# ax2.plot(rewardsAchieved)

# agent = FirstVisitMCAgent(environment=env, episodes=5000, epsilon=0.01)
# rewardsAchieved = agent.run()
# ax3.plot(rewardsAchieved)

# plt.savefig("test_runs/5000_episodes", bbox_inches="tight")