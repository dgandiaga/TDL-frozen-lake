# TDL-frozen-lake
This repository allows training Reinforcement Learning models for different variables of the OpenAI's Frozen Lake environment. The algorithm used is vanilla SARSA, in the exact same way that it is stated in **Reinforcement Learning, An Introduction (Richard S. Sutton and Andrew G. Barto)**.

SARSA is an online version of **Temporal Difference Learning**. That means that it uses the same policy for taking actions than for calculating expected returns when updating the policy. The loop goes as follows:
![image](https://user-images.githubusercontent.com/26325749/145645431-62e30720-fe43-4e02-8319-eb319b025124.png)

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

![frozen-lake-4x4-det](https://user-images.githubusercontent.com/26325749/145644502-7787bbfb-ba0b-4da4-8a27-5879eb4c21a1.png)![frozen-lake-8x8-det](https://user-images.githubusercontent.com/26325749/145644517-f8b1f8c0-ec43-40cd-8fbd-af80d0137724.png)



As soon as it has taken the best path a couple of times it never fails again. This is because for these environments I choose a high alpha since there's no noise in the returns and a full-greedy policy since encouraging exploration is not required.

Here we can see by launching the test-4x4-det service how it follows the best path possible:
![image](https://user-images.githubusercontent.com/26325749/145644552-d7c0eca9-b514-4a19-bd05-14f133b0754a.png)


### Stochastic versions of the environment:

The services for the stochastic versions of the environment are:
* train-4x4-det
* test-4x4-det
* train-8x8-det
* test-8x8-det

For these versions I've used a small alpha and a high epsilon decay in order to encourage the exploration and make the agent robust to the noise. Convergence is much harder to achieve and it never gets a perfect result.

Let's analyze the 4x4 example:
![frozen-lake-4x4](https://user-images.githubusercontent.com/26325749/145644576-e800a3b1-aef1-4da3-b821-f6f2acf4bcb1.png)

Convergence is much slower. You may think that accumulated reward is not as good as it should be, but keep in mind that this policy always keeps an epsilon higher than zero, so the average reward is lower than if we'd switch to the full-greedy policy once we now the agent is trained. The result when the agent is trained and we switch to the greedy policy is in fact about **75% of success**.

The agent gets really conservative in its choices due to the high degree of stochasticity introduced by both the slippery floor and the exploration ratio. For example, for the initial state it tries to go to the left:
![image](https://user-images.githubusercontent.com/26325749/145644801-d62ff22d-f314-4d76-ad45-fa2a02a07eed.png)

That may seem counterintuitive, but in fact trying this movement as many times as needed is the only way of eventually going down while eliminating the risk of going right, which is a non-desirable situation.

Here the agent has learn that it's better to go up with the hope of it ending up going to the right, even if the most possible outcome is it getting further away from the goal, instead of just trying directly to go to the right and facing the chance of end up going down to the hole:
![image](https://user-images.githubusercontent.com/26325749/145645034-b7d7355f-bed7-4412-9d14-9d397dd79911.png)

This behavior given the stochastic nature of the environment is one of the reasons I chose this algorithm instead of **Q-learning** or other off-policy versions of** Temporal Difference Learning**. In this paragraph from Sutton and Barto's book they explain the point over a similar problem but with less randomness, since there the stochasticity only comes from the exploration and not from the slippery floor:
![image](https://user-images.githubusercontent.com/26325749/145646053-5c2c6b41-3764-4761-8399-26994b4581c8.png)

Finally here are the results for the 8x8 stochastic frozen-lake:
![frozen-lake-8x8](https://user-images.githubusercontent.com/26325749/145646701-238769b7-9706-4068-87a0-abc18bc7db2e.png)

Here I had to raise gamma and alpha compared to the 4x4 version since runs are longer and there are more different states.

Performance over a thousand runs once switched to test-mode is **~72%**. The behavior follows the same patterns that I explained above for the stochastic 4x4 grid.
