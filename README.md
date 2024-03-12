# POLICY EVALUATION

## AIM

To develop a Python program to evaluate the given policy.



## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.
### STATES:

The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.
Five Transition states / Non-terminal States including S: The starting state.

### ACTIONS:
The agent can take two actions:

 * R: Move right.
* L: Move left.
* 
### Transition Probabilities:
The transition probabilities for each action are as follows:

* 50% chance that the agent moves in the intended direction.
* 33.33% chance that the agent stays in its current state.
* 6.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards:

The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.


## POLICY EVALUATION FUNCTION
### FORMULA:
![image](https://github.com/Aravindsamy04/rl-policy-evaluation/assets/113497037/3039af8a-5ab5-41de-8e0d-09e56c5b7212)

## PROGRAM:
### POLICY EVALUATION:
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
        V = np.zeros(len(P), dtype=np.float64)
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()


    return V
```

### Code to evaluate the first policy:
```
pi_1 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
print_policy(pi_1, P, action_symbols=('<', '>'), n_cols=7)
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_1, goal_state=goal_state)*100,
    mean_return(env, pi_1)))
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)
V1

```

### Code to evaluate the second policy:
```
pi_2 = lambda s: {
    0:RIGHT, 1:RIGHT, 2:LEFT, 3:RIGHT, 4:RIGHT, 5:LEFT, 6:RIGHT
}[s]
print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)
     

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state)*100,
    mean_return(env, pi_2)))

V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)
V2
```
### COMPARISON:

```
V1>=V2

print('ARAVIND SAMY.P')
print('212222230011')
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")


```




## OUTPUT:
## POLICY 1:

![Screenshot 2024-03-12 143355](https://github.com/Aravindsamy04/rl-policy-evaluation/assets/113497037/a9c88f7c-e7bf-42b3-97b6-ee00509e315d)

### POLICY 2:

![Screenshot 2024-03-12 143543](https://github.com/Aravindsamy04/rl-policy-evaluation/assets/113497037/6a81cd15-3aa9-49e0-a93a-c0ba724addc7)

### COMPARISION:
![Screenshot 2024-03-12 143630](https://github.com/Aravindsamy04/rl-policy-evaluation/assets/113497037/15b8e59c-0403-4407-b29d-149aff903912)
























## RESULT:

Thus, a Python program is developed to evaluate the given policy.

