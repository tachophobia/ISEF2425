# The Lichtenberg Figure Metaheuristic for Reinforcement Learning

## Project Overview
This repository is the product of a research project exploring the use of **Lichtenberg Figures,** the patterns created by electrical discharge, as a metaheuristic for improving the balance between **exploration and exploitation** in reinforcement learning (RL). The goal is to enhance the rate of convergence and stability of RL algorithms, particularly in environments with **continuous action spaces.** The research compares the performance of a Lichtenberg-inspired algorithm to established RL techniques, such as epsilon-greedy and UCB, through empirical evaluation and analysis.

## Introduction
Machine learning is being increasingly used to solve problems in many areas. One key aspect to consider with respect to training agents in both simulated and real-world environments is the amount of trials, or "episodes," necessary for the agent to find a solution. Therefore, it is important to find ways to decrease the amount of training needed in applications.

Reinforcement learning is a branch of machine learning in which an agent learns within the framework of the Markov decision process, meaning that it takes actions to change its state and accordingly gain rewards. The agent’s ultimate goal is to maximize the total reward, which it learns about through its experience. However, in the process of learning, the agent must repeatedly decide whether to continue exploiting rewards from following what it has currently determined to be the optimal set of actions, or to explore another set of actions to see if higher rewards might be found with them. This is known as the explore-exploit dilemma or tradeoff.


![A diagram of the Markov decision loop](https://ars.els-cdn.com/content/image/1-s2.0-S0029801822008666-gr3.jpg "The loop by which the agent interacts with the environment in RL")


Traditional methods of tackling this tradeoff involve a simple probabilistic choice, using a parameter called epsilon, but this is an inneficient approach. Heuristics attempt to expedite and guide this decision making by providing a rough approximation of when to explore and when to exploit. Metaheuristics, in turn, are heuristics which are adaptable and typically not tailored to a specific problem. Some of these metaheuristics are inspired by the physical world. One of these is the Lichtenberg algorithm (LA), which is based on Lichtenberg figures, the fractal patterns often created when electricity discharges on a surface.


**[place Lichtenberg figure image here]**
![An example of a Lichtenberg figure](eta5.dim1000.gif "An example Lichtenberg figure generated in this project using DBM (see below for methods)")


This project utilizes the LA as a metaheuristic to address the explore-exploit tradeoff in a novel way. We modify the actor-critic format of the deep deterministic policy gradient (DDPG), a variety of deep Q-learning for continuous action spaces, by using the LA for exploration in place of the actor. Normally, the actor (i.e., the policy function approximator) interacts with the environment to gain experience based on information that is not up to date with the critic (i.e., the Q-function approximator), and thus training takes more episodes. We hypothesize that using the Lichtenberg-based metaheuristic to explore the search space instead of the usual actor, thus keeping the experience information more up to date and searching the environment more efficiently, will decrease the amount of training episodes needed for the agent to converge on a solution compared to current prominent methods.

## Project Content
This repository is divided into folders based on the phases of the project and components of the algorithm. Below is an explanation of each:

### Tree Generation
The key to the algorithm used in this project is the generation of a Lichtenberg figure. There are two approaches to this generation contained in the Tree Generation folder: diffusion-limited aggregation (DLA) and dielectric breakdown model (DBM). Both are explained below:

#### DLA
DLA is

#### DBM
As opposed to DLA, DBM generation

### Experimental Results

This folder contains the materials associated with an electrochemical experiment conducted in a laboratory. We placed zinc onto a surface damp with copper(ii) sulfate solution, causing solid copper to deposit in a fractal pattern similar to a Lichtenberg figure. Images of two of these figures are shown in the "results" subfolder.

These images were then processed to be dual-colored with the **process_images.ipynb** file, and then individual branches were cropped off (these can be seen for one of the images in the files called fractal_branch# in the **branches** subfolder). Analysis was then conducted on each individual branch with the **fractal_analysis.py** script to determine each branch's fractal dimensionality, a number typically between 1 and 2. The higher the dimensionality, the more complex the fractal. A similar analysis was conducted for our DBM computer-generated figures, each generated with a different value of the eta parameter (see DBM explanation above), ranging from 4.0 to 5.5. We ran no statistical tests for comparison of the two, but the results showed that the physical and computer-generated figures both had dimensionalities from about 1.4-1.6, which we considered to be acceptably similar. In our later trials of our Lichtenberg agent, we used a Lichtenberg figure generated with eta set to 5.0, within the range of testing of this exercise. The full results are displayed here:

Fractal Dimensionalities of Each Branch of Lab-Created Lichtenberg Figure
| Branch Number | Dimensionality Estimate |
| ------------- | ----------------------- |
| 1 | 1.483 |
| 2 | 1.527 |
| 3 | 1.563 |
| 4 | 1.438 |
| 5 | 1.524 |

Fractal Dimensionalities for DBM-generated Figures with Different Values of eta Parameter
| eta value | Dimensionality Estimate |
| --------- | ----------------------- |
| 4.0 | 1.577 |
| 4.5 | 1.521 |
| 5.0 | 1.489 |
| 5.5 | 1.396 |


### Lichtenberg Optimization

This folder contains three files: 

- **func.py,** containing implementations of example functions to optimize
- **la.py,** containing the implementation of the Lichtenberg algorithm (LA, see below)
- **optimize.ipynb,** containing an annotated example of optimizing the Ackley function via the LA

#### Algorithm design

Contained in the file la.py, the Lichtenberg algorithm (LA) was implemented based on the 2021 paper “Lichtenberg algorithm: A novel hybrid physics-based meta-heuristic for global optimization" by Pereira et al. ***To be continued***

In this project, the LA is not used in isolation to optimize objective functions, but is instead used to partially optimize the action value function approximation in our modified deep deterministic policy gradient method. For more, see the Agents section.

### Agents
The Agents folder contains most of the critical project files. Here is a brief explanation of each:

| File/Subfolder  | Description |
| --------------- | ----------- |
| environments    | contains the classes of the environments used to test the agent, including submarine |
| simulations     | contains gifs showing the results of experimental runs of our agent in various environments |
| ddpg.py         | the benchmark DDPG agent (see below), our LA-based agent, and associated classes |
| dql.py          | deep Q-learning, an established RL technique for discrete action spaces |
| figure2d.npy    | the 2d Lichtenberg figure generated via DBM that was arbitrarily chosen for use in experimental trials |
| framework.py    | the abstract "Agent" class and the setup of the neural network used as a value function approximator |
| test_agents.ipynb | the script used to test our Lichtenberg agent against the benchmark DDPG agent |

The Deep Deterministic Policy Gradient (DDPG) is a

## Experimentation Methods
As shown in the test_agents.ipynb file, we tested our LA-based agent against the benchmark DDPG agent, which is similar to the well-established method of Q-learning. Both agents were set to the same parameters (shown in the table directly below) and underwent 30 full training runs for each of the environments until convergence. The number of episodes and the time taken to converge were recorded. The data for the submarine environment are shown in the results section below.

Controlled Parameters for Agents during Trials

| Parameter | Value |
| --------- | ----- |
| gamma     | 0.99  |
| tau       | 0.05  |
| noise     | 0.1   |
| batch_size | 32   |
| alpha     | 0.001 |
| hidden_dim | 64   |
| hidden_layers | 1 |
For more information on what wach parameter does, view the documented files in the Agents folder.

Additionally, within the Lichtenberg algorithm, the parameters for our agent during trials were set to:

- n_iter = 3
- pop = 15

## Results

Submarine Environment Convergence Data

***Add table here***

### Statistical Testing