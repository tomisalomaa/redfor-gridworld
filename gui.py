import tkinter as tk
from tkinter import ttk
from environment import Environment
from agent import Agent as RandomAgent
from temporal_learning_agent import QAgent as QAgent, SARSAAgent as SarsaAgent
from mc_agent import FirstVisitMCAgent as MCAgentFirstVisit

class Window:
    def __init__(self,root):
        self.root = root
        self.root.title("REDFOR REINFORCEMENT LEARNING")
        self.root.geometry("600x400")
        self.root.resizable(True,True)

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=10)
        
        agents = ["Choose agent", 
                    "Random", 
                    "MC first visit", 
                    "MC every visit", 
                    "Q-learning", 
                    "SARSA"]

        episodesLabel = ttk.Label(self.root, text="Episodes:")
        episodesLabel.grid(column=0, row=0, sticky=tk.W)
        episodeEntry = ttk.Entry(self.root)
        episodeEntry.insert(0, "1000")
        episodeEntry.grid(column=1, row=0, sticky=tk.W)

        stepsLabel = ttk.Label(self.root, text="Steps:")
        stepsLabel.grid(column=0, row=1, sticky=tk.W)
        stepsEntry = ttk.Entry(self.root)
        stepsEntry.insert(0, "1000")
        stepsEntry.grid(column=1, row=1, sticky=tk.W)

        epsilonLabel = ttk.Label(self.root, text="Epsilon:")
        epsilonLabel.grid(column=0, row=2, sticky=tk.W)
        epsilonEntry = ttk.Entry(self.root)
        epsilonEntry.insert(0, "0.01")
        epsilonEntry.grid(column=1, row=2, sticky=tk.W)

        alphaLabel = ttk.Label(self.root, text="Alpha:")
        alphaLabel.grid(column=0, row=3, sticky=tk.W)
        alphaEntry = ttk.Entry(self.root)
        alphaEntry.insert(0, "0.01")
        alphaEntry.grid(column=1, row=3, sticky=tk.W)

        gammaLabel = ttk.Label(self.root, text="Gamma:")
        gammaLabel.grid(column=0, row=4, sticky=tk.W)
        gammaEntry = ttk.Entry(self.root)
        gammaEntry.insert(0, "1")
        gammaEntry.grid(column=1, row=4, sticky=tk.W)

        agentLabel = ttk.Label(self.root, text="Agent:")
        agentLabel.grid(column=0, row=5, sticky=tk.W)
        agentVariable = tk.StringVar(self.root)
        agentDropDown = ttk.OptionMenu(self.root,
                                        agentVariable,
                                        *agents).grid(
                                                    column=1,
                                                    row=5, 
                                                    sticky=tk.W)

        runButton = ttk.Button(self.root, text="Run", 
                                command=lambda: self.runAgent(
                                    int(episodeEntry.get()),
                                    int(stepsEntry.get()),
                                    float(epsilonEntry.get()),
                                    float(alphaEntry.get()),
                                    float(gammaEntry.get()),
                                    agentVariable.get()
                                ))
        runButton.grid(column=0, row=6)
        
    def runAgent(self, episodes, steps, epsilon, alpha, gamma, agentType):
        env = Environment()
        if agentType == "Random":
            #DEBUG
            print("RANDOM AGENT")
            agent = RandomAgent(environment=env, episodes=episodes, 
                                maxSteps=steps)
        elif agentType == "MC first visit":
            #DEBUG
            print("MCFV AGENT")
            agent = MCAgentFirstVisit(environment=env, episodes=episodes, 
                                        maxSteps=steps, epsilon=epsilon, 
                                        alpha=alpha, gamma=gamma)
        elif agentType == "MC every visit":
            print("Not yet implemented")
            return 0
        elif agentType == "Q-learning":
            #DEBUG
            print("Q-LEARNING AGENT")
            agent = QAgent(environment=env, episodes=episodes, 
                            maxSteps=steps, epsilon=epsilon, 
                            alpha=alpha, gamma=gamma)
        elif agentType == "SARSA":
            agent = SarsaAgent(environment=env, episodes=episodes, 
                                maxSteps=steps, epsilon=epsilon, 
                                alpha=alpha, gamma=gamma)
        else:
            print("No agent type selected!")
            return 0
        #DEBUG
        print("AGENT WITH PARAMS:")
        print("     episodes:",episodes)
        print("     steps:",steps)
        print("     epsilon:",epsilon)
        print("     alpha:",alpha)
        print("     gamma:",gamma)
        print("     agent type:",agentType)
        print("RUNNING AGENT...")
        rewardsAchieved = agent.run()
        return rewardsAchieved

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