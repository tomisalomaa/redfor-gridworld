import tkinter as tk
import numpy as np
import time
import threading
from tkinter import ttk
from environment import Environment
from agent import Agent as RandomAgent
from temporal_learning_agent import QAgent as QAgent, SARSAAgent as SarsaAgent
from mc_agent import FirstVisitMCAgent as FVAgent, EveryVisitMCAgent as EVAgent

class Window:
    def __init__(self,root):
        self.root = root
        self.root.title("REDFOR REINFORCEMENT LEARNING")
        self.root.geometry("600x400")
        self.root.resizable(True,True)

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=10)
        
        self.createdThreads = []
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
                                        *agents)
        agentDropDown.grid(column=1,row=5,sticky=tk.W)

        runButton = ttk.Button(self.root, text="Run", 
                                command=lambda: self.getResultFromRun(
                                    int(episodeEntry.get()),
                                    int(stepsEntry.get()),
                                    float(epsilonEntry.get()),
                                    float(alphaEntry.get()),
                                    float(gammaEntry.get()),
                                    agentVariable.get()
                                ))
        runButton.grid(column=0, row=6)

        self.runResults = []
        self.resultsVar = tk.StringVar(value=self.runResults)
        self.resultsList = tk.Listbox(self.root, height=4, width=40, listvariable=self.resultsVar)
        self.resultsList.grid(column=2,row=0,rowspan=4, sticky=tk.W)
        
        self.imgCanvas = tk.Canvas(self.root, height=260, width=260, bg="white")
        self.imgCanvas.grid(column=2,row=4,rowspan=4,sticky=tk.W)
        self.episodeStatesArray = []
        self.stepAndEpisodeLabel = ttk.Label(self.root)
        self.stepAndEpisodeLabel.grid(column=2,row=8,rowspan=4,sticky=tk.W)

        simulateResultsButton = ttk.Button(self.root, text="Simulate", command=lambda: self.startNewThread(self.drawGridCanvas))
        simulateResultsButton.grid(column=0,row=7)

    def startNewThread(self, targetFunction):
        gridDrawThread = threading.Thread(target=self.drawGridCanvas)
        self.createdThreads.append(gridDrawThread)
        gridDrawThread.start()
        for thread in self.createdThreads:
            if not thread.is_alive():
                print("Closing",thread,"down..")
                thread.join()
                # TODO: Do stopped threads need attention (kill, remove?)

    # Current notes:
    # Last arr is a no-use array (doesnt contain an actual playthrough but is an end state with no redfor instead),
    # actual episode number needs to be contained with the episode sample arrays to tie it in the simulation.
    def drawGridCanvas(self, episodeArrModulus=10):
        episodes = self.episodeStatesArray
        for idx,arr in enumerate(episodes):
            for matrix in arr:
                if idx==0 or idx+1%episodeArrModulus == 0 or idx == len(episodes)-1:
                    self.imgCanvas.delete("all")
                    self.stepAndEpisodeLabel["text"]="Episode: "+str(idx+1)
                    for x in range(len(matrix)):
                        for y in range(len(matrix[x])):
                            if matrix[x][y] == 1:
                                rectColor = "red"
                            elif matrix[x][y] == 2:
                                rectColor = "blue"
                            elif matrix[x][y] == 3:
                                rectColor = "grey"
                            else:
                                 rectColor = "white"
                            self.imgCanvas.create_rectangle(x*25+5,y*25+5,(x+1)*25+5,(y+1)*25+5, fill=rectColor)
                    timeoutStart = time.time()
                    while time.time() < timeoutStart+0.05:
                        pass
        
    def getResultFromRun(self, episodes, steps, epsilon, alpha, gamma, agentType):
        returns = self.runAgent(episodes, steps, epsilon, alpha, gamma, agentType)
        self.runResults.append(str(agentType)+" "+str(episodes)+" episode avg: "+str(np.mean(returns[0])))
        self.resultsList.delete(0, tk.END)
        for result in self.runResults:
            self.resultsList.insert(0, result)
        self.episodeStatesArray = returns[1]

    def runAgent(self, episodes, steps, epsilon, alpha, gamma, agentType):
        env = Environment()
        if agentType == "Random":
            agent = RandomAgent(environment=env, episodes=episodes, 
                                maxSteps=steps)
        elif agentType == "MC first visit":
            agent = FVAgent(environment=env, episodes=episodes, 
                                        maxSteps=steps, epsilon=epsilon, 
                                        alpha=alpha, gamma=gamma)
        elif agentType == "MC every visit":
            agent = FVAgent(environment=env, episodes=episodes, 
                                        maxSteps=steps, epsilon=epsilon, 
                                        alpha=alpha, gamma=gamma)
        elif agentType == "Q-learning":
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
        print("RUNNING AGENT...")
        rewardsAchieved = agent.run()
        return rewardsAchieved

def main():
    root = tk.Tk()
    gui = Window(root)
    gui.root.mainloop()
    return None

main()