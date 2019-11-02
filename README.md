# Reward function adaptation: maze env

### Background
We consider a variation of both the standard Reinforcement Learning problem formulation and the [Hierarchical Reinforcement Learning formulation](https://arxiv.org/pdf/1910.04450.pdf).  We consider an MDP without an explicit rewardfunction tailored to it.  Instead, MDP possesses a fitness function that evaluates an agent according to someobjective criterion.  An example of such a criterion we discuss further.  The reward function, in its turn, becomesa decision variable, subject to an agent.  This provides us with an opportunity to fit a reward function to thegiven MDP and the given fitness criterion.
### Current progress
We decided to start with a single-agent environment [gym-maze](https://github.com/MattChanTK/gym-maze).  In this case, the fitness criterion is the numberof training epochs it takes an agent to learn the optimal maze-running policy for a given maze (which is theshortest path).  We re-implement the entire Reinforcement Learning pipeline to accommodate the changes inthe problem formulation2.  The current implementation exploits q-learning algorithm with a tabular q-function.The reward function is given in a tabular form as well.For the first experiment, we optimize the reward function of an agent using a [genetic optimization algorithm](https://github.com/DEAP/deap).
### Future work
We would like to shift towards multi-agent reinforcement learning.

<kbd>![Simple 2D maze environment](http://i.giphy.com/Ar3aKxkAAh3y0.gif)</kbd> <kbd>![Solving 20x20 maze with loops and portals using Q-Learning](http://i.giphy.com/rfazKQngdaja8.gif)</kbd>



## Installation
It should work on both Python 2.7+ and 3.4+. It requires pygame and numpy.

```bash
cd cs285-project
python setup.py install
```
## Examples
```bash
python RunMaze_script.py
```


