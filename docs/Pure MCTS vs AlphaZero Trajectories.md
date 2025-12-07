  Analysis: Jagged vs Smooth Trajectories

  Root Cause Identified

  Pure MCTS produces jagged trajectories because it uses greedy action selection:
  # mcts/mcts.py:54
  best_idx = int(np.argmax(root.Q_sa))  # Always picks action with highest Q-value

  AlphaZero produces smooth trajectories because it uses stochastic sampling from visit counts:
  # learning/training_loop.py:145
  action_idx = np.random.choice(len(pi), p=pi)  # Samples from visit distribution

  Where pi is the visit count distribution:
  # mcts/mcts_alphazero_controller.py:67
  pi = counts / np.sum(counts)  # Probability proportional to visit counts

  Why This Matters

  1. Pure MCTS (Greedy):
    - Always chooses the single "best" action based on Q-values
    - Small changes in state can flip which action has max Q
    - Results in abrupt direction changes → jagged trajectory
    - More exploitative, less exploration during execution
  2. AlphaZero (Stochastic):
    - Samples actions proportionally to how much MCTS "trusts" them (visit counts)
    - If two actions have similar visit counts, it might pick either
    - Smoother transitions between maneuvers → smooth trajectory
    - Maintains exploration during training (important for self-play)

  Additional Factors

  Looking at the code, there's also a difference in exploration during search:

  AlphaZero adds Dirichlet noise at root for exploration:
  # mcts/mcts_alphazero_controller.py:45
  noise = np.random.dirichlet([0.3] * len(actions))
  root.children[action_idx].prior = 0.95 * prior + 0.05 * noise[i]

  Pure MCTS uses random rollouts:
  # mcts/mcts.py:106
  action = self.mdp.rollout_policy(state)  # Random/heuristic policy

  Recommendation

  If you want smoother Pure MCTS trajectories, you could modify mcts/mcts.py to sample from a Boltzmann distribution instead of
  argmax:

  # Instead of:
  best_idx = int(np.argmax(root.Q_sa))

  # Use temperature-based sampling:
  tau = 0.1  # Lower = more greedy, higher = more exploratory
  pi = np.exp(root.Q_sa / tau) / np.sum(np.exp(root.Q_sa / tau))
  best_idx = np.random.choice(len(pi), p=pi)

  Pure MCTS (Greedy Selection)

  - Purpose: Find and execute the single best action based on tree search
  - Selection: argmax(Q) - deterministic, always picks highest value
  - Behavior: Exploitation-focused during execution
  - Trajectory: Jagged because it commits 100% to whatever action has highest Q-value at that moment
  - This is correct behavior for classical MCTS

  AlphaZero (Stochastic Sampling)

  - Purpose: Generate diverse training data through self-play
  - Selection: Sample from visit count distribution - stochastic
  - Behavior: Maintains exploration during training (critical for learning)
  - Trajectory: Smoother because it probabilistically samples among good actions
  - This is correct behavior for AlphaZero's training phase

  Why AlphaZero Uses Stochastic Sampling During Training

  1. Exploration: Needs diverse experiences to train the network effectively
  2. Visit counts = confidence: If MCTS visits action A 80 times and action B 20 times, both might be reasonable, just A is slightly better
  3. Prevents overfitting: Training on only greedy actions would make the network too narrow

  Important Note

  During deployment/testing (after training), AlphaZero typically switches to:
  - Greedy selection (argmax of visit counts or policy), OR
  - Low temperature sampling (nearly greedy)

  So the jagged vs smooth difference you're seeing is:
  - ✅ Expected - different algorithms, different selection strategies
  - ✅ Correct - Pure MCTS should be greedy, AlphaZero training should explore
  - ✅ By design - not a bug, but a feature of how each algorithm works

  The trajectories reflect the algorithms working as intended!
