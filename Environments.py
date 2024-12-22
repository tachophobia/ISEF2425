import numpy as np
import random
import matplotlib.pyplot as plt
import gym
from abc import ABC,abstractmethod

class Environment(ABC):
    @abstractmethod
    def set_rewards(self, rewards, actions):
        """Set rewards and actions for the environment."""
        pass

    @abstractmethod
    def set_state(self,s):
        """Set to a new state s (often the position of the agent)."""
        pass

    @abstractmethod
    def get_state(self):
        """Return the current state."""
        pass

    @abstractmethod
    def is_terminal(self,s):
        """Evaluate whether state s is terminal."""
        pass

    @abstractmethod
    def get_next_state(self,s,a):
        """Return next state given current state s and action a."""
        pass

    @abstractmethod
    def action(self,a):
        """Execute action a."""
        pass

    @abstractmethod
    def undo_action(self,a):
        """Undo action a."""
        pass

    @abstractmethod
    def all_states(self):
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
    def generateRandom(self, rowNum,colNum,chanceToDecrease):
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

    def standardSubmarine(self):
        rows=11
        cols=10
        actions,rewards = self.generateRandom(11,10,80)
        for j in range(rows):
                for i in range(cols):
                    if (i,j) in rewards: 
                        st = f"{rewards[i,j]}"
                        a=len(st)
                        print(f"{rewards[i,j]}{' '*(3-a)}",end="")
                    else:print("X  ",end="")
                print()
    
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
    


class ContinuousSubmarine(gym.Env): 
    def __init__(self,width,height,start_pos,tWidth,tHeight,max_y,action_space):
        self.w=width #width of environment
        self.h=height #height of environment
        self.tW=tWidth #width of terminal areas
        self.tH=tHeight #height of terminal areas
        self.state=[start_pos,0,0] #always in [(y,x),Vy,Vx] form, this time it's floats rather than ints
        self.max_y=max_y #dictionary describing the height of the "seafloor" in form {xPos:maxY} -- xPos and maxY are both ints
        self.action_space=action_space
        self.time_steps=0

    def set_rewards(self,rewards):
        self.rewards = rewards

    def set_state(self, s):
        self.state=s  

    def get_state(self):
        """Return the state the agent is in as a tuple representing position."""
        return self.state

    def is_terminal(self,s): #given pos s, return if it's terminal
        return self.to_box(s) in self.rewards

    def get_next_state(self, s, a): #s-->current state; a-->action in form (theta,magnitude) tuple; a vector representing the action; a will always be valid given s and the environment
        next_pos=s
        delta_X,delta_Y,vX,vY = self.CAKE(a)
        next_pos[0][0]=s[0][0]+delta_Y
        next_pos[0][1]=s[0][1]+delta_X
        next_pos[1]=vY
        next_pos[2]=vX
        return next_pos
    
    def step(self, a): #a-->action in form (theta,magnitude) tuple; a vector representing the action; action will always be valid given s and the environment; magnitude is magnitude of the force
        if self.is_valid_action(a): 
            self.last_pos = self.state
            delta_X,delta_Y,vX,vY = self.CAKE(a)
            self.state[0][0]+=delta_Y
            self.state[0][1]+=delta_X
            self.state[1]=vY
            self.state[2]=vX
            self.time_steps+=1
            return (self.state, self.rewards.get(self.to_box(self.state[0]),-1*a[1]), self.is_terminal(self.state[0]),self.time_steps>1000,{})  #the agent loses reward scaling with the size of the movement; the return is in line with Env.step's API
        delta_X,delta_Y,vX,vY = self.CAKE(a)
        self.state[0][0]+=delta_Y
        self.state[0][1]+=delta_X
        self.state[1]=vY
        self.state[2]=vX
        self.time_steps+=1
        return (self.state,0,True,self.time_steps>1000,{})  #the return if action isn't valid


    def to_box(pos): #takes pos and converts it to a coordinate representing the larger box
        y,x=int(pos[0]),int(pos[1])
        return ((y//5)*5,(x//5)*5)

    def undo_action(self, a):
        """Update the position of the agent by doing the reverse of action a."""
        self.state=self.last_pos
        assert(self.get_State()[0][0]>=0 and self.get_State()[0][0]<=self.h and self.get_State()[0][1]>=0 and self.get_State()[0][1]<=self.w)

    def get_terminal_states(self):  #returns the top-left corners of each terminal box; combined with tW and tH, this can be used to determine if a pos is terminal
        return [key for key in self.rewards]

    def is_valid_action(self,a):
        b= a[0]<2*np.pi and a[0]>0 and a[1]<1.5 and a[1]>.5
        if not b:return False
        next_state = self.get_next_state(self,self.state,a)
        if next_state[1]>self.w or next_state[1]<0 or next_state[0]<0: return False
        next_state_to_box=self.to_box(next_state)
        if next_state[0]>=self.max_y[next_state_to_box[1]]:return False
        return True

    def CAKE(self,a):
        x_accel=a[1]*np.cos(a[0])
        y_accel = a[1]*np.sin(a[0])
        delta_t=2
        delta_x = self.state[2] * delta_t + (x_accel * (delta_t^2))
        delta_y=self.state[1]*delta_t + y_accel*(delta_t^2)
        vX=self.state[2] +delta_t*x_accel
        vY=self.state[1] +delta_t*y_accel
        return delta_x,delta_y,vX,vY


    def standard_env():
        tW=5
        tH=5
        w=50
        h=55
        state=[(0,0),0,0]
        rewards={(5,0):1,(10,5):2,(15,10):3,(20,15):5,(20,20):8,(20,25):16,(35,30):24,(35,35):50,(45,40):74,(50,45):124}
        max_y={0:10,5:15,10:20,15:25,20:25,25:25,30:40,35:40,40:50,45:55} 
        action_space=(gym.spaces.Box(0,2*(np.pi)),gym.spaces.Box(0.05,5.5))

        grid = ["."]*(w*h)
        for x in range(w):
            for y in range(h):
                if ((y//5)*5,(x//5)*5) in rewards:
                    grid[x+y*w]="t"
                elif y>=max_y[(x//5)*5]:
                    grid[x+y*w]="X"
        print("\n".join(["".join(grid[i:i+w]) for i in range(0,w*h,w)]))

    
    def print_grid():
        pass




    
    