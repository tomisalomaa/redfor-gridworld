import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.5
alpha = 0.1
gamma = 0.95
EPS_DECAY = 0.999
SHOW_EVERY = 1000

start_q_table = None

PLAYER_N = 1
FOOD_N = 2
ENEMY = 3

d = {
    1: (255,175,0),
    2: (0,255,0),
    3: (0,0,255)
    }

class MilitaryUnit:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, destination):
        return (self.x - destination.x, self.y - destination.y)
    
    def checkGridBounds(self, x, y):
        if x < 0:
            self.x = 0
        elif x > SIZE-1:
            self.x = SIZE-1
        if y < 0:
            self.y = 0
        elif y > SIZE-1:
            self.y = SIZE-1
    
    def action(self, a):
        if a == 0:
            self.move(x=1, y=1)
        elif a == 1:
            self.move(x=-1, y=-1)
        elif a == 2:
            self.move(x=-1, y=1)
        elif a == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x
        
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y
        self.checkGridBounds(self.x,self.y)

