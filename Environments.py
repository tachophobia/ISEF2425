import numpy as np
import random
import matplotlib.pyplot as plt
from abc import ABC,abstractmethod

class Environment(ABC):
    @abstractmethod
    def set(self, rewards, actions):
        """Set rewards and actions for the environment."""
        pass

    @abstractmethod
    def setState(self,s):
        """Set to a new state s (often the position of the agent)."""
        pass

    @abstractmethod
    def getState(self):
        """Return the current state."""
        pass

    @abstractmethod
    def isTerminal(self,s):
        """Evaluate whether state s is terminal."""
        pass

    @abstractmethod
    def getNextState(self,s,a):
        """Return next state given current state s and action a."""
        pass

    @abstractmethod
    def action(self,a):
        """Execute action a."""
        pass

    @abstractmethod
    def undoAction(self,a):
        """Undo action a."""
        pass

    @abstractmethod
    def allStates(self):
        """Returns the set of all states and corresponding rewards."""
        pass

class Submarine:
    def __init__(self,rows,cols,startPos):
        """Initialize the size of the environment and the starting position of the agent.
        
        Args:
            rows (int): the height of the environment.
            cols (int): the width of the environment.
            startPos (tuple): the starting position of the agent.
        """
        self.rows=rows
        self.cols=cols
        self.pos=startPos
        self.actions={'U','R','D','L'}
    def set(self,rewards,actions):
        self.rewards,self.actions = rewards,actions
    def setState(self, s):
        self.pos=s  
    def getState(self):
        """Return the state the agent is in as a tuple representing position."""
        return self.pos
    def isTerminal(self,s):
        """Return a bool of whether the state is terminal, based on whether there are any possible actions in the state s."""
        return s not in self.actions
    def getNextState(self, s, a):
        """
        Return the next state.
        
        Args:
            s (tuple): the initial state.
            a (string): the action taken.
            
        Returns:
            tuple: the final state.
        """
        i,j=s[0],s[1]
        if a in self.actions[s]:
            if a=='U': i-=1
            elif a=='D': i+=1
            elif a=='R': j+=1
            elif a=='L': j-=1
        return (i,j)
    def action(self,a):
        """Update the position of the agent based on action a and return the reward associated with the new position."""
        if a in self.actions[self.pos]:
            if a=='U': self.pos[0]-=1
            elif a=='D': self.pos[0]+=1
            elif a=='R': self.pos[1]+=1
            elif a=='L': self.pos[1]-=1
        return self.rewards.get(self.pos,0)
    def undoAction(self, a):
        """Update the position of the agent by doing the reverse of action a."""
        if a in self.actions[self.pos]:
            if a=='U': self.pos[0]+=1
            elif a=='D': self.pos[0]-=1
            elif a=='R': self.pos[1]-=1
            elif a=='L': self.pos[1]+=1
        assert(self.getState() in self.allStates())
    def allStates(self):
        """Return the set of all possible states, including terminal ones."""
        return set(self.actions.keys()) | set(self.rewards.keys())
    def generateRandom(rowNum,colNum,chanceToDecrease):
        """
        Generate a random environment of specified size.
        
        Args:
            rowNum (int): the height of the environment.
            colNum (int): the width of the environment.
            chanceToDecrease (float): the probability (1-100) of going down a level when assigning rewards.
            
        Returns:
            dict {tuple: int}: the value of the reward given for being in each state.
            dict {tuple: set}: the set of possible actions in each state.
        """
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
        """Print the environment grid with associated rewards in each position, or an X for impossible states."""
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