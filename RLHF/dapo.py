"""
DAPO (Decoupled Clip + Dynamic Sampling Policy Optimization) — LLM RL skeleton
============================================================================

What this file is:
- A *teaching-quality* implementation that makes DAPO mechanics explicit.
- It is not a full production RL system (no distributed rollout workers, no vLLM, etc.).

What DAPO (the LLM-RL DAPO) is known for:
- Built on a group-based RL update (GRPO-like), i.e., multiple completions per prompt.
- Adds techniques such as "Clip-Higher" and "Dynamic Sampling" to improve stability/efficiency. :contentReference[oaicite:2]{index=2}

Algorithm overview (per iteration):
1) Sample a batch of prompts: x_1..x_B
2) For each prompt x_b, sample K_b completions from policy πθ_old:
      y_{b,1}, ..., y_{b,K_b}
   Here K_b is *dynamic* (depends on uncertainty/variance).
3) Compute rewards r_{b,k} = R(x_b, y_{b,k})
4) Compute group baseline and group-relative advantage A_{b,k}:
      baseline_b = mean_k r_{b,k}
      A_{b,k} = (r_{b,k} - baseline_b) / (std_b + eps)   (optional normalization)
5) Policy update with PPO-like clipped objective but with:
   - decoupled clipping (different behavior for A>0 vs A<0)
   - clip-higher: allow larger upside ratio to maintain diversity / avoid entropy collapse

Important modeling choice:
- Token-level policy gradient:
  For each (prompt, completion), we sum log-probs only over completion tokens.
  This matches standard LLM RLHF practice.

This script includes:
- A simple dynamic-K sampler based on group reward std.
- A DAPO loss with decoupled, asymmetric clipping (educational).
- KL regularization toward a reference model (optional).

Dependencies:
  pip install torch transformers

You should replace:
- model_name: with your SFT checkpoint (Qwen/Llama/etc.)
- reward_fn: with a real verifier-based reward
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# 0) Reproducibility / device
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1) Hyperparameters
# -----------------------------
@dataclass
class DAPOConfig:
    model_name: str = "gpt2"          # replace with your SFT model checkpoint
    ref_model_name: Optional[str] = None  # for KL regularization; often same as SFT snapshot

    # Rollout / generation controls
    max_prompt_len: int = 256
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95

    # Group sampling
    K_min: int = 2         # minimum completions per prompt
    K_max: int = 8         # maximum completions per prompt
    target_std: float = 0.35  # dynamic sampling target: want enough reward variance estimate

    # Optimization
    lr: float = 1e-5
    train_steps: int = 200
    batch_prompts: int = 2  # number of prompts per step

    # DAPO clipping behavior
    eps_low: float = 0.2        # lower clip bound (1 - eps_low)
    eps_high: float = 0.6       # *higher* upper clip bound (1 + eps_high) = "Clip-Higher"
    # Clip-Higher is described as helping avoid entropy collapse / maintain diversity. :contentReference[oaicite:3]{index=3}

    # Decoupled clip:
    # - For positive advantages, allow bigger increases (upper bound bigger).
    # - For negative advantages, keep stricter clipping to avoid over-penalizing and instability.
    # This is an intuitive form of "decoupled clip" used in practice.
    neg_clip_upper: float = 0.2  # when A<0, upper bound = 1+neg_clip_upper (typically small)

    # Advantage normalization
    adv_norm: bool = True
    adv_eps: float = 1e-6

    # Optional KL regularization to reference policy (like RLHF)
    beta_kl: float = 0.0  # set >0 to keep policy close to ref

cfg = DAPOConfig()


# -----------------------------
# 2) Load tokenizer / models
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
policy.train()

# Snapshot / reference model for KL penalty (optional)
ref = None
if cfg.ref_model_name is not None or cfg.beta_kl > 0:
    ref_name = cfg.ref_model_name or cfg.model_name
    ref = AutoModelForCausalLM.from_pretrained(ref_name).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

optimizer = optim.AdamW(policy.parameters(), lr=cfg.lr)


# -----------------------------
# 3) Prompt dataset (toy)
# -----------------------------
PROMPTS = [
    "Solve: If 3x + 5 = 20, what is x? Give only the final number.",
    "Compute: 17 * 19. Give only the final number.",
    "Write exactly JSON: {\"answer\": <number>}. Compute 12^2.",
]


# -----------------------------
# 4) Reward function (replace!)
# -----------------------------
def reward_fn(prompt: str, completion: str) -> float:
    """
    Replace this with a REAL verifier-based reward.

    DAPO/GRPO-style RL works best when reward is:
    - verifiable (math checker / unit tests / exact match)
    - low-noise, not purely heuristic
    """
    import re
    gold = {
        PROMPTS[0]: 5,
        PROMPTS[1]: 323,
        PROMPTS[2]: 144,
    }

    # Simple format rule for prompt 3
    if "exactly JSON" in prompt:
        ok = re.fullmatch(r'\{\s*"answer"\s*:\s*-?\d+\s*\}', completion.strip()) is not None
        if not ok:
            return 0.0

    m = re.search(r"-?\d+", completion.strip())
    if not m:
        return 0.0
    pred = int(m.group(0))
    return 1.0 if pred == gold[prompt] else 0.0


# -----------------------------
# 5) Utilities: tokenize + generation
# -----------------------------
def tokenize_prompts(prompts: List[str]) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_prompt_len,
    )
    return {k: v.to(device) for k, v in enc.items()}

@torch.no_grad()
def generate_k_completions(prompt: str, K: int) -> List[str]:
    """
    Sample K completions from current policy.
    In a scalable system, this is done with fast inference engines (vLLM, etc.).
    Here we keep it simple.

    We decode full text then strip the prompt prefix.
    """
    inputs = tokenize_prompts([prompt])
    input_ids = inputs["input_ids"]

    completions = []
    for _ in range(K):
        out_ids = policy.generate(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask", None),
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # strip prompt (best-effort)
        comp = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        completions.append(comp)
    return completions


# -----------------------------
# 6) Token-level log-prob sum over completion tokens
# -----------------------------
def build_prompt_completion_ids(prompt: str, completion: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build concatenated ids:
      [prompt_ids] + [completion_ids] + [EOS]
    and a mask that marks which tokens are completion tokens (1) vs prompt (0).

    We'll compute log-probs for the completion tokens only.
    """
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][: cfg.max_prompt_len]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"][: cfg.max_new_tokens]
    c_ids = c_ids + [tokenizer.eos_token_id]

    ids = (p_ids + c_ids)[: (cfg.max_prompt_len + cfg.max_new_tokens)]
    mask = [0] * min(len(p_ids), len(ids))
    mask += [1] * (len(ids) - len(mask))

    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )

def logprob_sum_completion(model: AutoModelForCausalLM, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute sum_t log π(token_t | prefix_{<t}) over completion tokens.

    For causal LMs:
      logits[:, t] predicts ids[t+1]
    So:
      logits[:-1] aligns with labels[1:].
    We must shift mask similarly: mask[1:].
    """
    # Add batch dimension [1, T]
    input_ids = ids.unsqueeze(0)
    attn = torch.ones_like(input_ids, device=device)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]            # [1, T-1, V]
    labels = input_ids[:, 1:]                 # [1, T-1]
    m = mask[1:].unsqueeze(0)                 # [1, T-1], marks completion tokens among labels

    logp = torch.log_softmax(logits, dim=-1)  # [1, T-1, V]
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    return (token_logp * m).sum(dim=-1).squeeze(0)  # scalar


# -----------------------------
# 7) Dynamic sampling rule for K
# -----------------------------
def choose_K_from_rewards(rewards: List[float]) -> int:
    """
    A simple dynamic sampling heuristic:

    - Start with K_min samples
    - Estimate reward std; if std is too low (uncertain / no signal),
      we sample more completions up to K_max.

    Rationale:
    - If all rewards are identical, group-relative advantages become ~0,
      producing weak gradients. Increasing K can expose better samples.
    - This is a practical interpretation of "Dynamic Sampling" to improve efficiency/stability. :contentReference[oaicite:4]{index=4}
    """
    if len(rewards) < 2:
        return cfg.K_min
    std = float(torch.tensor(rewards).std(unbiased=False).item())
    if std < cfg.target_std:
        # If reward variance is too low, increase K
        return min(cfg.K_max, len(rewards) + 2)
    return len(rewards)


# -----------------------------
# 8) DAPO loss: decoupled asymmetric clipping + clip-higher
# -----------------------------
def dapo_policy_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantage: torch.Tensor,
) -> torch.Tensor:
    """
    PPO-like clipped objective with DAPO-style tweaks.

    Standard PPO uses:
      ratio = exp(logp_new - logp_old)
      L = -E[ min(ratio*A, clip(ratio, 1-eps, 1+eps)*A) ]

    DAPO-style modifications (educational but aligned with described ideas):
    1) Clip-Higher:
       - Use a larger *upper* clip bound (1+eps_high) than typical PPO,
         to avoid entropy collapse / maintain diversity. :contentReference[oaicite:5]{index=5}
    2) Decoupled clipping:
       - When A > 0: allow bigger upward changes (upper = 1+eps_high)
       - When A < 0: restrict upward changes more (upper = 1+neg_clip_upper)
         because for negative advantage you mainly want to *reduce* probability.

    Lower bound is kept at (1-eps_low).

    NOTE:
    - There are multiple legitimate ways to implement "decoupled clip".
      This version is simple and stable in practice.
    """
    ratio = torch.exp(logp_new - logp_old)

    # Decide per-sample upper clip based on advantage sign
    upper_pos = 1.0 + cfg.eps_high
    upper_neg = 1.0 + cfg.neg_clip_upper
    upper = torch.where(advantage >= 0, torch.tensor(upper_pos, device=device), torch.tensor(upper_neg, device=device))

    lower = 1.0 - cfg.eps_low
    ratio_clipped = torch.clamp(ratio, lower, upper)

    # conservative surrogate
    surrogate1 = ratio * advantage
    surrogate2 = ratio_clipped * advantage

    # We minimize negative of (min of surrogates)
    loss = -torch.mean(torch.min(surrogate1, surrogate2))
    return loss


# -----------------------------
# 9) Training loop
# -----------------------------
for step in range(1, cfg.train_steps + 1):
    # ---- sample prompts batch ----
    batch_prompts = random.sample(PROMPTS, k=min(cfg.batch_prompts, len(PROMPTS)))

    all_losses = []
    avg_reward = 0.0
    n_samples = 0

    for prompt in batch_prompts:
        # ---- Dynamic sampling: start with K_min, then expand if needed ----
        K = cfg.K_min
        completions = generate_k_completions(prompt, K=K)
        rewards = [reward_fn(prompt, c) for c in completions]

        # Possibly expand K based on observed reward variance
        newK = choose_K_from_rewards(rewards)
        while newK > K:
            extra = generate_k_completions(prompt, K=(newK - K))
            completions += extra
            rewards += [reward_fn(prompt, c) for c in extra]
            K = newK
            newK = choose_K_from_rewards(rewards)

        # ---- Group baseline + advantages ----
        r = torch.tensor(rewards, dtype=torch.float32, device=device)  # [K]
        baseline = r.mean()
        if cfg.adv_norm:
            std = r.std(unbiased=False) + cfg.adv_eps
            adv = (r - baseline) / std
        else:
            adv = (r - baseline)

        # ---- Compute old logprobs under current policy snapshot (π_old) ----
        # In a scalable system you would freeze θ_old per iteration.
        # Here we approximate π_old as "policy before optimizer.step() in this loop body".
        # So we compute logp_old with torch.no_grad() *before* update.
        with torch.no_grad():
            logp_old_list = []
            for c in completions:
                ids, m = build_prompt_completion_ids(prompt, c)
                logp_old_list.append(logprob_sum_completion(policy, ids, m))
            logp_old = torch.stack(logp_old_list)  # [K]

        # ---- Compute new logprobs (requires grads) ----
        logp_new_list = []
        for c in completions:
            ids, m = build_prompt_completion_ids(prompt, c)
            logp_new_list.append(logprob_sum_completion(policy, ids, m))
        logp_new = torch.stack(logp_new_list)  # [K]

        # ---- Optional KL penalty toward reference policy ----
        # A common RLHF-style regularizer: KL(πθ || πref).
        # Here we approximate it via logprob difference on sampled completions:
        #   KL proxy ≈ E[ logπθ - logπref ]
        # and subtract beta*KL from the advantage signal (reward shaping).
        if ref is not None and cfg.beta_kl > 0:
            with torch.no_grad():
                logp_ref_list = []
                for c in completions:
                    ids, m = build_prompt_completion_ids(prompt, c)
                    logp_ref_list.append(logprob_sum_completion(ref, ids, m))
                logp_ref = torch.stack(logp_ref_list)  # [K]
            kl_proxy = (logp_new.detach() - logp_ref)  # [K]
            # shape advantage: penalize drift from reference
            adv = adv - cfg.beta_kl * kl_proxy

        # ---- DAPO policy loss ----
        loss = dapo_policy_loss(logp_new=logp_new, logp_old=logp_old, advantage=adv)
        all_losses.append(loss)

        avg_reward += float(r.mean().item()) * len(r)
        n_samples += len(r)

    # ---- Backprop / step ----
    total_loss = torch.stack(all_losses).mean()
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    # ---- logging ----
    if step % 10 == 0:
        print(
            f"[step {step:04d}] loss={total_loss.item():.4f} "
            f"avg_reward={avg_reward/max(1,n_samples):.4f}"
        )

print("DAPO training finished.")
