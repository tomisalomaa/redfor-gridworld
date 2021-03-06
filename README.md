# redfor-gridworld

![coarseSimCapabilities](https://user-images.githubusercontent.com/52319409/133126973-ff480521-1f2d-4dfd-b7b6-be5b578f963c.PNG)

Last update of this readme on 13/09/2021: added features TODO.

## Description
This application is a personal project motivated by a better understanding of classical reinforcement learning methods. The goal is to implement four different classical reinforcement learning agents in a gridworld environment with a problem set specified below. At the same time the application should produce statistics and simulation of the chosen agent's performance and offer the user the means to extract these statistics in order to perform comparative analysis between the learning methods available.

### Functionalities
Supported reinforcement learning agents are:
- Completely stochastic (ie. completely random choice, probably won't make it into production but acts as a "bottom-line")
- First-visit Monte Carlo
- Every-visit Monte Carlo
- Q-learning
- SARSA

Application is Python based.

### Problem environment
Goal of this application is to demonstrate whether tactical principles such as mass, manouver and economoy of force are achieved by means of classical reinforcement learning methods. The environment for the problem at hand is in general envisioned as such:

The area of operations (AO) for red force (REDFOR) contains a single target area that is in size relative to the total force of the REDFOR. AO also contains 1-3 units of blue force (BLUFOR) depending on the scenario at hand. The task of REDFOR is to beat BLUFOR from the area - if able to - and reach the target zone.

REDFOR achieves positive reinforcement based on force ratio in engagements and the efficiency of reaching the terminal state which is the target area.

## Current state / implemented features from TODO
- Includes a GUI template with agent parameter inputs
- Agent classes are at core functional and able to be ran
- Coarse simulation of ran episodes is possible

## TODO
- Implement further mathemathics to all learning algorithms: include alpha and gamma properly; allow for n-step variations for Q-learning and SARSA, implement the possibility of decaying epsilon
- Implement off-policy features for all learning algorithms
- Create a class to provide data handling functions
- Create a class to provide analytics tools
- Improve GUI to provide the user the means to use above functionalities
- Define and document the reinforcement problem as Markov Decision Process and implement a rule set in the environment based on said definition
- Implement ATTACK -action functionality, including Lanchester's laws based outcome deciding
- Improve UX
- (Eventually refactor to separate GUI and the rest of the application into a server + client architecture model)
