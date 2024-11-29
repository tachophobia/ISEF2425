import numpy as np
import random
import matplotlib.pyplot as plt
from abc import ABC,abstractmethod

class Environment(ABC):
    @abstractmethod
    def set(self, rewards, actions):
        """sets rewards and actions for the environment"""
        pass

    @abstractmethod
    def setState(self,s):
        """sets a new state (often the position of the agent)"""
        pass

    @abstractmethod
    def getState(self):
        """returns the current state"""
        pass

    @abstractmethod
    def isTerminal(self,s):
        """evaluates whether it is a terminal state"""
        pass

    @abstractmethod
    def getNextState(self,s,a):
        """given a current state and an action, returns the next state"""
        pass

    @abstractmethod
    def action(self,a):
        """executes an action"""
        pass

    @abstractmethod
    def undoAction(self,a):
        """undoes action a"""
        pass

    @abstractmethod
    def allStates(self):
        """returns the set of all states and corresponding rewards"""
        pass

class Submarine:
    def __init__(self,rows,cols,startPos):
        self.rows=rows
        self.cols=cols
        self.pos=startPos
        self.actions={'U','R','D','L'}
    def set(self,rewards,actions):
        self.rewards,self.actions = rewards,actions
    def setState(self, s):
        self.pos=s  
    def getState(self):
        return self.pos
    def isTerminal(self,s):
        return s not in self.actions
    def getNextState(self, s, a):
        i,j=s[0],s[1]
        if a in self.actions[s]:
            if a=='U': i-=1
            elif a=='D': i+=1
            elif a=='R': j+=1
            elif a=='L': j-=1
        return (i,j)
    def action(self,a):
        if a in self.actions[self.pos]:
            if a=='U': self.pos[0]-=1
            elif a=='D': self.pos[0]+=1
            elif a=='R': self.pos[1]+=1
            elif a=='L': self.pos[1]-=1
        return self.rewards.get(self.pos,0)
    def undoAction(self, a):
        if a in self.actions[self.pos]:
            if a=='U': self.pos[0]+=1
            elif a=='D': self.pos[0]-=1
            elif a=='R': self.pos[1]-=1
            elif a=='L': self.pos[1]+=1
        assert(self.getState() in self.allStates())
    def allStates(self):
        return set(self.actions.keys()) | set(self.rewards.keys())
    def generateRandom(rowNum,colNum,chanceToDecrease):
        rewards = {(0,1):1}  #all indexed column, row (i,j)
        actions = {}
        topOfCol={0:1}
        j=1
        r = [1,2,3,5,8,16,24,50,74,124]
        for i in range(1,colNum): #setting terminal states
            toss = random.randint(1,100)
            if toss<=chanceToDecrease and (j+1)<rowNum:
                j+=1
            topOfCol[i]=j
            rewards[(i,j)]=r[i]
        for i in range(0,colNum): #setting state-action dict and negative time rewards
            j=0
            while (i,j) not in rewards:
                rewards[(i,j)]=-1
                ac= set()
                if i-1>0 and j<=topOfCol[i-1]: ac.add('L')
                if i+1<colNum and j<=topOfCol[i+1]: ac.add('R')
                if j+1<=topOfCol[i]: ac.add('D')
                if j-1>=0 and j-1<=topOfCol[i]: ac.add('U')
                actions[(i,j)]=ac
                j+=1
        return actions,rewards

    def standardSubmarine():
   #     e = Submarine(11,10,(0,0))
    #    actions,rewards = e.generateRandom(11,10,95)
     #   e.set(rewards,actions)
      #  e.printSub
       # return e
        pass
    
    def printSub(self):
        for j in range(self.rows):
            for i in range(self.cols):
                if (i,j) in self.rewards: 
                    st = f"{self.rewards[i,j]}"
                    a=len(st)
                    print(f"{self.rewards[i,j]}{' '*(3-a)}",end="")
                else:print("X  ",end="")
            print()

    rows=11
    cols=10
    actions,rewards = generateRandom(11,10,80)
    for j in range(rows):
            for i in range(cols):
                if (i,j) in rewards: 
                    st = f"{rewards[i,j]}"
                    a=len(st)
                    print(f"{rewards[i,j]}{' '*(3-a)}",end="")
                else:print("X  ",end="")
            print()