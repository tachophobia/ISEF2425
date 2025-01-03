# The Lichtenberg Figure Metaheuristic for Reinforcement Learning

## Project Overview
This repository is the product of a research project exploring the use of **Lichtenberg Figures,** the patterns created by electrical discharge, as a metaheuristic for improving the balance between **exploration and exploitation** in reinforcement learning (RL). The goal is to enhance the rate of convergence and stability of RL algorithms, particularly in environments with **continuous action spaces.** The research compares the performance of a Lichtenberg-inspired algorithm to established RL techniques, such as epsilon-greedy and UCB, through empirical evaluation and analysis.

## Introduction
Reinforcement learning is a branch of machine learning in which an agent learns within the framework of the Markov decision process, meaning that it takes actions to change its state and accordingly gain rewards. The agent’s ultimate goal is to maximize the total reward, which it learns about through its experience. However, in the process of learning, the agent must repeatedly decide whether to continue exploiting rewards from following what it has currently determined to be the optimal set of actions, or to explore another set of actions to see if higher rewards might be found with them. Traditional methods of tackling this tradeoff involve a simple probabilistic choice, using a parameter called epsilon. Heuristics attempt to expedite this decision making by providing a rough approximation of when to explore and when to exploit, and metaheuristics are heuristics which are adaptable and typically not tailored to a specific problem. Some of these metaheuristics are inspired by the physical world

## Project Content
This repository is divided into folders based on the phases of the project and components of the algorithm. Below is an explanation of each:

### Tree Generation

The key to the algorithm used in this project is the generation of a Lichtenberg figure. 

### Lichtenberg Optimization

This folder contains three files: 

| func.py  | implementations of example functions to optimize  |
| la.py   | the implementation of the Lichtenberg algorithm (LA, see below) |
| optimize.ipynb | containing an annotated example of optimizing the Ackley function via the LA |

#### Algorithm design

Contained in the file la.py, the Lichtenberg algorithm (LA) was implemented based on the 2021 paper “Lichtenberg algorithm: A novel hybrid physics-based meta-heuristic for global optimization" by Pereira et al. **To be continued**

In this project, the LA is not used in isolation to optimize objective functions, but is instead used to partially optimize the value function approximation in place of the actor in the deep deterministic policy gradient method. For more, see the Agents section.

### Agents

### Experimental Results

## Experimentation Methods

## Results

