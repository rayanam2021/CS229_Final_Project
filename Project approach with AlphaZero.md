**Goal:**
Learn a *neural policy* $\pi_{\theta}(a\mid s)$ and *value function* $V_{\phi}(s)$ that approximate what the MCTS would do

From:
[AlphaZero paper](https://arxiv.org/pdf/1712.01815) 
[AlphaZero tutorial](https://suragnair.github.io/posts/alphazero.html ) 
DM under Uncertainty book

## 1. Monte Carlo Tree Search to gather data
Calculate $U(s,a)$, the upper confidence bound on the Q-values as:

$$a^* = \arg\max_a \Big[ Q(s,a) + c_{puct}\, P(s,a) \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} \Big]$$

- $Q(s,a)=W(s,a)/N(s,a)$: mean action value (average of all simulated returns from taking $a$ in $s$)
- $N(s,a)$: the number of times we took action a from state s (**visit counts**)
- $P(a \mid s)=\pi_{\theta}(a \mid s)$: parameterized policy network (prior probability)
- $c_{puct}$: hyperparameter that controls the degree of exploration
- Use $V_\phi(s_t)$ to avoid rollouts (bootstrapped value)

### Implementation
For each sampled root state $s_t$​:
1. Run MCTS for several simulations.  
	Store:
    - $N(s_t,a)$
    - $Q(s_t,a)$
    - $z_{t}$: rollout return (total InfoGain–Cost).
2. Construct **MCTS policy**

$$
\pi_{MCTS}(a|s_t) = \frac{N(s_{t},a)^{1/\tau}}{\sum_{b}N(s_{t},b)^{1/\tau}}
$$

	where $\tau$ is a hyperparameter controlling how sharp/greedy the distribution is:
	    - $\tau \to 0$ → nearly one-hot (greedy)
	    - $\tau = 1$→ smooth imitation of visit proportions
	    - $\tau > 1$→ softer, more exploratory
3. Generate dataset with $(s_t, \pi_{MCTS}, z_t)$ tuples with MCTS over propagator and observation model

### **Bootstrapped Backup**
To do once a $V_\phi(s')$ is obtained through training
When expanding new nodes, use $V_\phi(s')$ instead of rollouts:

$$
Q(s,a) \leftarrow \frac{N(s,a) Q(s,a) + [R(s,a) + \gamma V_\phi(s')]}{N(s,a) + 1}
$$

This replaces the need for explicit long rollouts in the search.

## 2. Neural Network selection
Neural network $f_{\psi}(s)$ has two outputs:
- Policy $\pi_{\theta}(a \mid s)$: softmax logits
- Value function $V_\phi(s_t)$ (scalar): predicted expected return

### Loss function (regression + classification)

#### Policy imitation loss | classification
You want the learned policy $\pi_{\theta}(a \mid s)$ to match the MCTS policy

$$\mathcal{L}_{policy} = - \sum_a \pi_{MCTS}(a|s_t) \log \pi_\theta(a|s_t)$$

This is the *soft* [cross-entropy](https://www.geeksforgeeks.org/machine-learning/what-is-cross-entropy-loss-function/) between the MCTS target and the network’s output, equivalent to minimizing the KL divergence:

$$D_{KL}(\pi_{MCTS}\|\pi_\theta) = \sum_a \pi_{MCTS}(a|s) \log \frac{\pi_{MCTS}(a|s)}{\pi_\theta(a|s)}$$

The constant term is dropped, yielding the cross-entropy loss.

#### Value regression loss | regression
Train the value function $V_\phi(s_t)$ to predict the **total discounted return** $z_t$​ from node $t$:

$$\mathcal{L}_{value} = (z_t - V_\phi(s_t))^2$$

with the return defined as:

$$z_t = \sum_{k=t}^{T} \gamma^{k-t} [\text{InfoGain}(b_k,b_{k+1}) - c\|\Delta v_{a_k}\|]$$

or recursively:

$$z_t = r_t + \gamma z_{t+1}, \quad r_t = \text{InfoGain}(b_t,b_{t+1}) - c\|\Delta v_{a_t}\|$$

#### Full loss function

$$\mathcal{L} = \mathcal{L}_{policy} + \mathcal{L}_{value} + c_{reg} \, \|\theta,\phi\|^2$$

with $L_{2}$ regularization and $c_v$ controlling the weight between policy and value components.

This is a standard supervised learning problem, so networks reviewed in class should work.

### Datasets
Options for splitting MCTS-generated data:
- 80/10/10 for train/validation/test, or
- 70/30 for train/test (tune hyperparameters within the 70%).

### Training
Minimize $\mathcal L$ using SGD/Adam/AdamW optimizer with typical setup:

| Hyperparameter | Typical value |
|----------------|----------------|
| Learning rate | $3\times10^{-4}$ with decay |
| Weight decay | $1\times10^{-4}$ |
| Batch size | 128–512 |
| Gradient clipping | 5.0 |
| Early stopping | yes |

### Optimizer Notes
AdamW is preferred for stability and adaptive learning rate management.
- $\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$
- Learning rate scheduling (cosine or step decay) recommended.
- Regularization via $c_{reg}$ keeps weights bounded.

## 3. Training Loop

#### Refinement / Self-play Loop

**Stage 1 – Before any training**  
Use MCTS with full rollouts to compute $z_t$. This data trains the initial network.
	You have no reliable $V_{\phi}(s)$.  
	MCTS must run full rollouts to compute returns ztz_tzt​.  
	That’s slow → used only to generate initial training data.

**Stage 2 – After first training cycles**  
Once $V_\phi(s)$ is roughly trained, use it as a **bootstrap** value in MCTS, avoiding rollouts.

This modifies return estimation from:

$$
Q(s,a) = \frac{1}{n}\!\sum_i [r^{(i)}_0 + \gamma r^{(i)}_1 + \cdots + \gamma^{T_i-1} r^{(i)}_{T_i-1}]
$$

to

$$
Q(s,a) = \frac{1}{n}\!\sum_i [r^{(i)}_0 + \gamma r^{(i)}_1 + \cdots + \gamma^{T_i-1}(r^{(i)}_{T_i-1} + \gamma V_\phi(s^{(i)}_{T_i}))].
$$

That lets you run _many more searches per second_ and still collect high-quality $(s, \pi, z)$ pairs for the next training round.  
Each iteration, the network’s predictions get sharper, which improves the _next_ MCTS
So  $V_\phi$ is used to evaluate leaves faster.

**Stage 3 – After convergence**  
When $\pi_\theta$ and $V_\phi$ approximate MCTS well, MCTS becomes unnecessary.  
You deploy the network directly for real-time control.

At that point, replacing rollouts with $V_\phi(s)$ was primarily a training-efficiency mechanism.
## 4. After training

Onboard execution uses only $\pi_\theta$ (no MCTS).
### Neural Network Outputs
#### Policy
The policy outputs **logits** for each discrete action. Use of softmax to turn logits into categorical probability distribution:

$$
\vec{y}(s)=f_{policy}(s), \quad
\pi_{\theta}(a\mid s)=\text{softmax}(y(s))_{a}=\frac{\exp(y_{a}(s))}{\sum_{b}\exp(y_{b}(s))}.
$$

During deployment:
- pick *greedy action* $a=\arg \max_{a} \pi_{\theta}(a \mid s)$
- or sample stochastically for exploration.

#### Value function
$V_{\phi}(s)$ shows how much total InfoGain–Cost the agent expects from a given state.
