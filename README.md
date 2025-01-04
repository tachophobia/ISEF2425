# The Lichtenberg Figure Metaheuristic for Reinforcement Learning

## Project Overview
This repository is the product of a research project exploring the use of **Lichtenberg Figures,** the patterns created by electrical discharge, as a metaheuristic for improving the balance between **exploration and exploitation** in reinforcement learning (RL). The goal is to enhance the rate of convergence and stability of RL algorithms, particularly in environments with **continuous action spaces.** The research compares the performance of a Lichtenberg-inspired algorithm to established RL techniques, such as epsilon-greedy and UCB, through empirical evaluation and analysis.

## Introduction
Reinforcement learning is a branch of machine learning in which an agent learns within the framework of the Markov decision process, meaning that it takes actions to change its state and accordingly gain rewards. The agent’s ultimate goal is to maximize the total reward, which it learns about through its experience. However, in the process of learning, the agent must repeatedly decide whether to continue exploiting rewards from following what it has currently determined to be the optimal set of actions, or to explore another set of actions to see if higher rewards might be found with them. This is known as the explore-exploit dilemma or tradeoff.

![A diagram of the Markov decision loop](https://ars.els-cdn.com/content/image/1-s2.0-S0029801822008666-gr3.jpg)

Traditional methods of tackling this tradeoff involve a simple probabilistic choice, using a parameter called epsilon, but this is a slow approach. Heuristics attempt to expedite and guide this decision making by providing a rough approximation of when to explore and when to exploit. Metaheuristics, in turn, are heuristics which are adaptable and typically not tailored to a specific problem. Some of these metaheuristics are inspired by the physical world. One of these is the Lichtenberg algorithm (LA), which is based on Lichtenberg figures, the fractal patterns often created when electricity discharges on a surface.

This project utilizes the LA as a metaheuristic to address the explore-exploit tradeoff. **Add more** 

## Project Content
This repository is divided into folders based on the phases of the project and components of the algorithm. Below is an explanation of each:

### Tree Generation

The key to the algorithm used in this project is the generation of a Lichtenberg figure. There are two approaches to this generation contained in the Tree Generation folder: diffusion-limited aggregation (DLA) and dielectric breakdown model (DBM). Both are explained below:

#### DLA

#### DBM

### Lichtenberg Optimization

This folder contains three files: 

- **func.py,** containing implementations of example functions to optimize
- **la.py,** containing the implementation of the Lichtenberg algorithm (LA, see below)
- **optimize.ipynb,** containing an annotated example of optimizing the Ackley function via the LA

#### Algorithm design

Contained in the file la.py, the Lichtenberg algorithm (LA) was implemented based on the 2021 paper “Lichtenberg algorithm: A novel hybrid physics-based meta-heuristic for global optimization" by Pereira et al. **To be continued**

In this project, the LA is not used in isolation to optimize objective functions, but is instead used to partially optimize the value function approximation in place of the actor in the deep deterministic policy gradient method. For more, see the Agents section.

### Agents

### Experimental Results

This folder contains the materials associated with an electrochemical experiment conducted in a laboratory. We placed zinc onto a surface damp with copper(ii) sulfate solution, causing solid copper to deposit in a fractal pattern similar to a Lichtenberg figure. Images of two of these figures are shown in the "results" subfolder.

These images were then processed to be dual-colored with the process_images.ipynb file, and then individual branches were cropped off (these can be seen for one of the images in the files called fractal_branch# in the branches subfolder). Analysis was then conducted on each individual branch with the fractal_analysis.py script to determine each branch's fractal dimensionality, a number typically between 1 and 2.

## Experimentation Methods

## Results