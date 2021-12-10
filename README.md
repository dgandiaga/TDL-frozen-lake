# TDL-frozen-lake
This repository allows training Reinforcement Learning models for different variables of the OpenAI's Frozen Lake environment. The algorithm used is vanilla SARSA, in the exact same way that it is stated in **Reinforcement Learning, An Introduction (Richard S. Sutton and Andrew G. Barto)**.

SARSA is an online version of **Temporal Difference Learning**. That means that it uses the same policy for taking actions than for calculating expected returns when updating the policy. The loop is as follows:
![image](https://user-images.githubusercontent.com/26325749/145639861-01aee871-37f0-489c-a176-c15a882021b0.png)

At the beginning it initializes all the action values Q(s, a) when a new state is discovered. In this version I use surrealistic high values for initialization in order to encourage exploration.
Then everytime an action is taken, it chooses the action that is going to be taken in the new state and use its Q value for updating the expected value of the previous state. That only affects to non-terminal states, for terminal states the return is just the reward gotten in this last step.

This means that it is an on-line algorithm (the same policy that is being used for choosing the actions is also used for calculating returns during the update). This means it's sensitive to bad choices due to a non-greedy policy's decision for exploring. In this case I use an epsilon-greedy policy whose epsilon is multiplied by a factor lower than one at the end of every episode.

## Usage

The project is dockerized. You can check the **docker-compose.yml**, where you'll find all the possible services defined. Every one of them is already tuned with the best parameters for every situation. In order to launch them you only need to build and run the services. For example, for service **train-4x4-det**:
```
docker-compose build train-4x4-det
docker-compose run train-4x4-det
```

## Services
Every version has its train and test service, already tuned with some suggested parameters. The environment can be set to a 4x4 or a 8x8 grid. It also can be set as deterministic or probabilistic. That means 2x2x2=8 services.

### Deterministic versions of the environment:

The services for the deterministic versions of the environment are:
* train-4x4-det
* test-4x4-det
* train-8x8-det
* test-8x8-det

These configurations are not a real challenge since the model only needs to find the best sequence of actions once and then exploit it as much as possible. Here we can see that convergence to the optimal policy is quite fast (~15-50 episodes):
![img.png](img.png)![img_1.png](img_1.png)

As soon as it has taken the best path a couple of times it never fails again. This is because for these environments I choose a high alpha since there's no noise in the returns and a full-greedy policy since encouraging exploration is not required.

Here we can see by launching the test-4x4-det service how it follows the best path possible:
![img_2.png](img_2.png)~~

