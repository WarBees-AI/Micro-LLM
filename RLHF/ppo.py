"""
PPO (Proximal Policy Optimization) — from-scratch, heavily commented, PyTorch + Gymnasium.

This file is intentionally verbose in comments to teach PPO *deeply*.

What PPO is doing (big picture)
-------------------------------
We want to optimize a stochastic policy πθ(a|s) to maximize expected return, but:
- pure policy gradient updates can be unstable (too large steps destroy performance)
- PPO stabilizes training by restricting how much the policy can change per update
  using a *clipped surrogate objective* (and/or KL penalty)

Core PPO ingredients
--------------------
1) Actor-Critic: we learn both
   - policy πθ(a|s)  (actor)
   - value Vφ(s)     (critic / baseline)

2) Rollouts (on-policy data):
   - collect trajectories using the *current* policy πθ_old
   - compute advantages Â_t and returns R_t (targets for critic)

3) PPO Update:
   - update actor to maximize:
       L_clip(θ) = E[ min( r_t(θ) * Â_t, clip(r_t(θ), 1-ε, 1+ε) * Â_t ) ]
     where r_t(θ) = πθ(a_t|s_t) / πθ_old(a_t|s_t)
   - update critic to minimize value error (e.g., MSE(R_t, Vφ(s_t)))
   - add entropy bonus to encourage exploration
   - often add approximate KL monitoring + early stopping if update too large

Mapping to RLHF (important connection)
--------------------------------------
In RLHF with PPO:
- state s_t is "prompt + generated tokens so far"
- action a_t is "next token"
- reward is from a reward model (plus KL penalty vs reference model)
- PPO math stays the same, only the environment changes (text-generation as environment)

Dependencies
-----------
pip install torch gymnasium

If gymnasium isn't available, this script tries `import gym` as fallback.

This is a teaching script: it aims for clarity > performance.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Gymnasium is the actively maintained Gym fork.
try:
    import gymnasium as gym
except ImportError:
    import gym  # fallback (older)


# ---------------------------
# 0) Reproducibility / device
# ---------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# 1) PPO hyperparameters
# ---------------------------
@dataclass
class PPOArgs:
    env_id: str = "CartPole-v1"
    seed: int = 42

    # rollout collection
    num_steps: int = 2048            # steps collected per iteration (batch size in time)
    num_envs: int = 1                # keep 1 for clarity; vector envs are faster
    gamma: float = 0.99              # discount factor
    gae_lambda: float = 0.95         # GAE smoothing (bias-variance tradeoff)

    # PPO update
    update_epochs: int = 10          # number of passes over collected data
    minibatch_size: int = 256        # SGD minibatch size
    clip_eps: float = 0.2            # PPO clipping epsilon (policy ratio bound)
    vf_coef: float = 0.5             # value loss weight
    ent_coef: float = 0.01           # entropy bonus weight
    max_grad_norm: float = 0.5       # gradient clipping for stability
    lr: float = 3e-4

    # KL monitoring (optional but very useful)
    target_kl: float = 0.02          # if approx KL exceeds this, stop policy updates early

    # training length
    total_timesteps: int = 200_000   # total environment interactions (approx)


args = PPOArgs()
set_seed(args.seed)


# -----------------------------------------
# 2) Environment creation (single env)
# -----------------------------------------
def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    # Gymnasium uses env.reset(seed=...)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    return env


env = make_env(args.env_id, args.seed)

# Observation and action dimensions for discrete-action environment (CartPole)
obs_shape = env.observation_space.shape  # e.g. (4,)
assert obs_shape is not None
obs_dim = obs_shape[0]
assert hasattr(env.action_space, "n"), "This minimal PPO example expects discrete actions."
act_dim = env.action_space.n


# -----------------------------------------
# 3) Actor-Critic network (shared backbone)
# -----------------------------------------
class ActorCritic(nn.Module):
    """
    A minimal shared network with two heads:
      - policy logits over actions (for categorical distribution)
      - value estimate V(s)

    For PPO, we need:
      - log_prob of chosen actions under current policy
      - entropy of policy distribution (exploration bonus)
      - value predictions for advantage estimation + critic training
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 64

        # Shared torso: maps observation -> latent features
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Policy head: outputs logits for categorical distribution over actions
        self.policy_head = nn.Linear(hidden, act_dim)

        # Value head: outputs scalar V(s)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits: shape [batch, act_dim]
          value:  shape [batch]    (squeezed)
        """
        x = self.backbone(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from πθ(a|s) and return:
          action:      int tensor
          log_prob:    log πθ(a|s)
          value:       V(s)

        During rollout collection, we treat policy parameters as "fixed"
        and store log_prob from πθ_old. Later, in PPO update, we recompute
        log_prob under the updated policy πθ to form the ratio r_t(θ).
        """
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the policy for given (obs, actions) under *current* θ:
          log_prob: log πθ(a|s)
          entropy:  H[πθ(.|s)]  (encourages exploration, discourages collapse)
          value:    V(s)        (critic prediction)
        """
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value


model = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# -----------------------------------------
# 4) Rollout buffer (stores on-policy batch)
# -----------------------------------------
class RolloutBuffer:
    """
    Stores one on-policy batch of transitions for PPO.

    We store:
      obs_t, action_t, logprob_t, reward_t, done_t, value_t
    Then compute:
      advantages_t, returns_t

    NOTE:
    - PPO is on-policy: the collected data must come from πθ_old
    - We store logprob_t from πθ_old so the PPO ratio can be computed later:
        r_t(θ) = exp( logprob_new - logprob_old )
    """

    def __init__(self, num_steps: int, obs_dim: int):
        self.obs = torch.zeros(num_steps, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, device=device, dtype=torch.long)
        self.logprobs = torch.zeros(num_steps, device=device)
        self.rewards = torch.zeros(num_steps, device=device)
        self.dones = torch.zeros(num_steps, device=device)
        self.values = torch.zeros(num_steps, device=device)

        self.advantages = torch.zeros(num_steps, device=device)
        self.returns = torch.zeros(num_steps, device=device)

        self.num_steps = num_steps
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        done: float,
        value: torch.Tensor,
    ):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        """
        Generalized Advantage Estimation (GAE-λ)

        Advantage is roughly:
          A_t = Q(s_t,a_t) - V(s_t)

        But Q is unknown. We estimate advantages using temporal-difference (TD) errors:
          δ_t = r_t + γ V(s_{t+1}) - V(s_t)

        GAE computes:
          Â_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + ...

        This reduces variance compared to Monte Carlo returns,
        while keeping bias manageable via λ.
        """
        adv = 0.0
        for t in reversed(range(self.num_steps)):
            next_value = last_value if t == self.num_steps - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]  # 0 if done, else 1
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + gamma * lam * next_nonterminal * adv
            self.advantages[t] = adv

        self.returns = self.advantages + self.values

        # Optional but common: normalize advantages to stabilize updates
        # (scale invariance for policy gradient)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_minibatches(self, minibatch_size: int):
        """
        Yield randomized minibatches for SGD.
        PPO typically does multiple epochs over the same rollout batch.
        """
        idxs = torch.randperm(self.num_steps, device=device)
        for start in range(0, self.num_steps, minibatch_size):
            mb = idxs[start : start + minibatch_size]
            yield (
                self.obs[mb],
                self.actions[mb],
                self.logprobs[mb],
                self.advantages[mb],
                self.returns[mb],
                self.values[mb],
            )


# -----------------------------------------
# 5) PPO loss function (the heart of PPO)
# -----------------------------------------
def ppo_update(
    buffer: RolloutBuffer,
    model: ActorCritic,
    optimizer: optim.Optimizer,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    update_epochs: int,
    minibatch_size: int,
    target_kl: float,
) -> Dict[str, float]:
    """
    Run PPO updates using the collected rollout buffer.

    Policy loss (clipped surrogate):
      L^CLIP(θ) = E[ min( r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t ) ]

    Where:
      r_t(θ) = πθ(a_t|s_t) / πθ_old(a_t|s_t)
             = exp( logπθ - logπθ_old )

    Intuition:
      - If new policy increases probability of good actions (Â_t > 0),
        we want r_t > 1. But we clip so it can't grow too much.
      - If new policy decreases probability of bad actions (Â_t < 0),
        we want r_t < 1. But we clip so it can't shrink too much.
      - This yields a "trust-region-like" constraint without complex TRPO math.

    Critic loss:
      L^VF = MSE( V(s_t), R_t )

    Entropy bonus:
      L^ENT = -E[ H(πθ(.|s_t)) ]   (minus because we minimize total loss)
      - higher entropy -> more exploration
      - prevents premature collapse to deterministic policy

    Total loss (minimized):
      L = -(L^CLIP) + vf_coef * L^VF - ent_coef * E[entropy]

    KL monitoring:
      PPO implementations often compute approximate KL between old and new policies:
        approx_kl = E[ logπ_old - logπ_new ]
      If KL is too large, early stop epochs to avoid destructive update.
    """
    # For logging
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_clipfrac = 0.0
    num_updates = 0

    for epoch in range(update_epochs):
        # Each epoch uses shuffled minibatches of the same on-policy rollout data
        for (obs, actions, old_logprobs, adv, returns, old_values) in buffer.get_minibatches(minibatch_size):
            # Evaluate current policy on the minibatch
            new_logprobs, entropy, new_values = model.evaluate_actions(obs, actions)

            # 1) Compute PPO ratio r_t(θ)
            # ratio = π_new / π_old = exp(logπ_new - logπ_old)
            ratio = torch.exp(new_logprobs - old_logprobs)

            # 2) Compute unclipped and clipped objectives
            # unclipped: ratio * advantage
            pg_loss1 = ratio * adv

            # clipped ratio: force ratio into [1-ε, 1+ε]
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            pg_loss2 = clipped_ratio * adv

            # We maximize the *minimum* of the two (conservative update)
            # In PyTorch, we minimize loss, so take negative.
            policy_loss = -torch.mean(torch.min(pg_loss1, pg_loss2))

            # 3) Value function loss (critic)
            # Returns are the regression targets.
            # Many PPO variants also "clip" value updates; we keep it simple here.
            value_loss = torch.mean((new_values - returns) ** 2)

            # 4) Entropy bonus (encourages exploration)
            entropy_loss = -torch.mean(entropy)  # negative because we add -ent_coef * entropy to total loss

            # 5) Combine losses
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping: avoids exploding updates
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # 6) Approximate KL for monitoring
            # A common approximation:
            #   KL(old || new) ≈ E[ logπ_old - logπ_new ]
            # This is not exact but correlates well with step size.
            approx_kl = torch.mean(old_logprobs - new_logprobs).item()

            # 7) Clip fraction: how often ratio was outside clip range
            # Useful signal: if clipfrac is high, updates are pushing too hard.
            clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).item()

            # Accumulate logs
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += torch.mean(entropy).item()
            total_kl += approx_kl
            total_clipfrac += clipfrac
            num_updates += 1

        # Early stopping on KL (very practical)
        # If KL is too big, further epochs over same data might overfit / destabilize.
        avg_kl = total_kl / max(1, num_updates)
        if avg_kl > target_kl:
            break

    # Return averaged stats for printing
    return {
        "policy_loss": total_policy_loss / max(1, num_updates),
        "value_loss": total_value_loss / max(1, num_updates),
        "entropy": total_entropy / max(1, num_updates),
        "approx_kl": total_kl / max(1, num_updates),
        "clipfrac": total_clipfrac / max(1, num_updates),
        "updates": num_updates,
    }


# -----------------------------------------
# 6) Training loop
# -----------------------------------------
"""
Training loop structure:
- We repeatedly:
  A) Collect rollouts with current policy πθ_old for num_steps
  B) Compute GAE advantages + returns
  C) Run PPO update epochs over the collected batch (on-policy)
  D) Repeat until total_timesteps reached

Key difference vs off-policy RL (e.g., DQN):
- PPO is on-policy: old data becomes stale after each policy update.
- So we collect fresh rollouts frequently.
"""

obs, _info = env.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=device)

global_step = 0
episode_return = 0.0
episode_len = 0
num_episodes = 0

while global_step < args.total_timesteps:
    buffer = RolloutBuffer(args.num_steps, obs_dim)

    # ---- A) Rollout collection ----
    for t in range(args.num_steps):
        global_step += 1
        episode_len += 1

        # sample action from current policy
        action, logprob, value = model.act(obs)

        # step environment
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = float(terminated or truncated)

        episode_return += float(reward)

        # store transition in buffer
        buffer.add(
            obs=obs,
            action=action,
            logprob=logprob,
            reward=float(reward),
            done=done,
            value=value,
        )

        # prepare next obs
        if done:
            num_episodes += 1
            # print episode summary occasionally
            if num_episodes % 10 == 0:
                print(f"[Episode {num_episodes}] return={episode_return:.1f} len={episode_len}")

            next_obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        if global_step >= args.total_timesteps:
            break

    # ---- B) Compute advantages / returns ----
    # For GAE we need V(s_{T}) for the last state after rollout
    with torch.no_grad():
        _logits, last_value = model.forward(obs.unsqueeze(0))
        last_value = last_value.squeeze(0)

    buffer.compute_gae(last_value=last_value, gamma=args.gamma, lam=args.gae_lambda)

    # ---- C) PPO Update ----
    stats = ppo_update(
        buffer=buffer,
        model=model,
        optimizer=optimizer,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
    )

    # ---- D) Print training statistics ----
    print(
        f"[Step {global_step}] "
        f"policy_loss={stats['policy_loss']:.4f} "
        f"value_loss={stats['value_loss']:.4f} "
        f"entropy={stats['entropy']:.4f} "
        f"kl={stats['approx_kl']:.4f} "
        f"clipfrac={stats['clipfrac']:.3f} "
        f"updates={stats['updates']}"
    )

print("\nTraining finished.")


"""
How to adapt this PPO code to RLHF (very explicitly)
---------------------------------------------------
To turn this into PPO-RLHF for an LLM, you would replace the Gym environment with a "text environment":

1) State s_t:
   - prompt + generated tokens so far

2) Action a_t:
   - next token chosen from vocabulary (categorical distribution)

3) Episode termination:
   - EOS token or max length

4) Reward r:
   - reward_model(prompt, full_response) at the end of the sequence
   - optionally add per-step shaping or penalties
   - IMPORTANT: add KL penalty vs reference model:
       r_total = r_rm - beta * KL(πθ || π_ref)

5) Value function:
   - can be token-level value head that predicts expected final reward from partial sequence

All PPO math stays identical:
- ratio = exp(logπ_new - logπ_old)
- clipped objective with ε
- critic regression to returns
- entropy bonus
- KL monitoring and/or KL penalty in reward

If you tell me:
- which base model you want (Qwen/Llama/Pangu/etc.)
- whether you use LoRA
- your reward model format (pairwise preference RM or direct scoring)
I can rewrite this into a *token-level PPO-RLHF training loop* (still with deep comments),
either using TRL or implementing the PPO update manually for sequences.
"""

