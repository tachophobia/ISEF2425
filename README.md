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
The key to the algorithm used in this project is the generation of a Lichtenberg figure. There are two implementations of approaches to this generation contained in the Tree Generation folder: diffusion-limited aggregation (DLA) and dielectric breakdown model (DBM). Both are explained below:

#### DLA
In the DLA model, particles, termed "walkers," are released one by one from random points and drift randomly until they reach the developing fractal or, in some cases, a set radius or boundary line. If the particle reaches the fractal, it will attach to it with a probability termed "stickiness" which on a larger scale controls the density of the resulting figure (Pereira et al., 2021, p. 4).

#### DBM
As opposed to aggregating particles from outside like in DLA, DBM generation builds outward. Each particle or component of the developing figure (which initially is a point) is set to have an electrical potential according to Laplace's equation. These potentials influence the probabilities of a new particle being added at each neighboring location to an existing point, and thus according to these probabilities the model tends to add new points farther away from existing branches to build a Lichtenberg figure (Irurzun et al., 2002). A parameter symbolized by the Greek letter **eta** is used as the power in the equation which derives the probabilities, and eta is roughly the equivalent of the stickiness factor from DLA, affecting the dimensionality of the resulting fractal.

Past research has shown that there is no significant physical difference between figures generated via DLA vs. DBM (Irurzun et al., 2002). For the purposes of this project, we chose DBM as our LF generator because it is both more a more efficient algorithm and more true to the natural process of figure formation.

### Experimental Results

This folder contains the materials associated with an electrochemical experiment conducted in a laboratory. We placed zinc onto a surface damp with copper(ii) sulfate solution, causing solid copper to deposit in a fractal pattern similar to a Lichtenberg figure. Images of two of these figures are shown in the "results" subfolder.

These images were then processed to be dual-colored with the **process_images.ipynb** file, and then individual branches were cropped off (these can be seen for one of the images in the files called fractal_branch# in the **branches** subfolder). Analysis was then conducted on each individual branch with the **fractal_analysis.py** script to determine each branch's fractal dimensionality, a number between 1 and 2 for flat-surface fractals. The higher the dimensionality, the more complex the fractal. A similar analysis was conducted for our DBM computer-generated figures, each generated with a different value of the eta parameter (see DBM explanation above), ranging from 4.0 to 5.5. We ran no statistical tests for comparison of the two, but the results showed that the physical and computer-generated figures both had dimensionalities from about 1.4-1.6, which we considered to be acceptably similar. In our later trials of our Lichtenberg agent, we used a Lichtenberg figure generated with eta set to 5.0, within the range of testing of this exercise. The full results are displayed here:

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

Contained in the file la.py, the Lichtenberg algorithm (LA) was implemented based on the 2021 paper “Lichtenberg algorithm: A novel hybrid physics-based meta-heuristic for global optimization" by Pereira et al. The LA functions by placing a LF in the search space of the function to be optimized, finding the optimum function value point among a representative "population" group of a pre-determined number of points on the current figure, and then starting the next iteration by placing a LF (either the same one or a new one) with its origin on the previous optimal point. Because the figures are necessarily generated on matrices with pixels and not actual points, there is not actually an infinitely small distance between each point. This means the function is at some level discrete, not continuous, and therefore cannot search the space fully without adaptations. To greatly mitigate this, the researchers applied changes to the figure between iterations. Namely, they applied a random scaling and a random rotation, as well as the option to generate a new figure each iteration if desired. They also built in a refinement parameter to create a smaller figure each time as a means of bettering local search, but in our implementation we removed this capability by keeping ref=0 (therefore not generating a smaller figure). Below is a brief explanation of each LA parameter:

| Parameter | Explanation |
| --------- | ----------- |
| n_iter | the power of ten of the number of iterations of search desired |
| pop | the number of points in the population group of each iteration that is intended to be a representative sample of the figure from which the current optimum point is determined |
| ref | the scaling factor (between 0 and 1, inclusive) to create the smaller LF used for local search |

Lichtenberg figures serve as good metaheuristics because of their ability to scan the search space efficiently. They spread out widely in a pattern with complexity at all levels. In the LA, scaled down LFs in some iterations allow the algorithm to exploit, while larger scaled LFs allow the function to explore, so as not to get stuck in local extrema. This process helps the LA to optimize objective functions in fewer iterations (Pereira et al., 2021, p. 6).

[**Here include an image - probably best is a gif of LA sampling?**]

In this project, we did not use the LA in isolation to optimize objective functions like the researchers did, but instead used it to partially optimize the action value function approximation in our modified deep deterministic policy gradient method, using LA to explore the search space in place of the typical "actor." For more, see the Agents section.

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

The deep deterministic policy gradient (DDPG) has been described as a counterpart of deep Q-learning (DQL) for continuous action spaces as opposed to discrete ones (INSERT SOURCE HERE). DDPG adapts DQL by using function approximators for both the action-value function, often symbolized as Q, and the policy, instead of using tables as DQL does. [**Add stuff here about actor and critic**]

[**HERE EXPLAIN HOW LFs FIT INTO DDPG - Nikita?**]

## Experimentation Methods
As shown in the test_agents.ipynb file, we tested our LA-based agent against the benchmark DDPG agent, which is similar to the well-established method of deep Q-learning. Both agents were set to the same parameters (shown in the table directly below) and underwent 30 full training runs for each of the environments until convergence. The number of episodes and the time taken to converge were recorded. The data for the submarine environment are shown in the results section below.

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

For more information on what each parameter does, view the documented files in the Agents folder.

Additionally, within the Lichtenberg algorithm, the parameters for our agent's exploration during trials were set to:

- n_iter = 3
- pop = 15

See the explanation of the Lichtenberg algorithm above for more on these.

## Results

We chose to focus on the number of episodes each agent took to converge to a solution. Time to convergence and solution stability were not analyzed in this study, but these and other factors may be addressed in future revisions.

Epidodes before Convergence Summary Statistics for the Submarine Environment

|     | Benchmark Agent | Lichtenberg Agent |
| --- | --------------- | ----------------- |
| n | 30 | 30 |
| mean | 6137.667 | 2153.667 |
| SD | 6408.816 | 1016.951 |
| min | 591 | 597 |
| Q1 | 2191 | 1594 |
| median | 3641 | 1998 |
| Q3 | 7891 | 2599 |
| max | 24491 | 4999 |

### Statistical Testing
As a preliminary measure, a Shapiro-Wilk Normality test was conducted for both distributions. We found that both were right-skewed and not normal distributions. However, because of the large sample sizes of n=30 for both, the statistical test discussed next was deemed able to continue.

Because there was only one independent variable, the **One-Way ANOVA test** was chosen to analyze the differences between the means. The probability that the means were the same was found to be **p=0.001371**. Because this value was less than 0.05, we concluded that there was a **statistically significant difference** between the means.

In summary, we found that, on average, the Lichtenberg agent was able to converge to a solution in fewer episodes than the benchmark agent in the submarine environment.

## Conclusion and Discussion
This study explored the application of the pre-existing Lichtenberg algorithm of Pereira et al. (2021) to the explore-exploit tradeoff in reinforcement learning environments with continuous action spaces. The LA was used as the explorer rather than the typical actor, a function approximator of the policy, in the deep deterministic policy gradient (DDPG) method. In trials in the submarine environment, the LA-modified agent was found to perform better on average than the benchmark DDPG agent with respect to the number of episodes of training needed to converge to a solution.

[**ADD DISCUSSION**]

## References

- Irurzun, I., Bergero, P., Mola, V., Cordero, M., Vicente, J., & Mola, E. (2002).
     Dielectric breakdown in solids modeled by DBM and DLA. Chaos, Solitons &
     Fractals, 13(6), 1333-1343. https://doi.org/10.1016/
     s0960-0779(01)00142-4
- Pereira, J. L. J., Francisco, M. B., Diniz, C. A., Oliver, G. A., Cunha, S. S.,
     Jr, & Gomes, G. F. (2021). Lichtenberg algorithm: A novel hybrid
     physics-based meta-heuristic for global optimization. Expert Systems With
     Applications, 170, 114522. https://doi.org/10.1016/j.eswa.2020.114522