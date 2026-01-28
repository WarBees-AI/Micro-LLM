"""
GSPO — Group Sequence Policy Optimization (LLM RL) — from-scratch teaching skeleton
=================================================================================

References / key ideas
----------------------
GSPO (Qwen team) argues GRPO instability comes from *misusing token-level importance ratios*,
which causes high-variance noise that accumulates with response length and can trigger collapse.
GSPO fixes this by:
  1) Using a *sequence-level* importance ratio derived from sequence likelihood (importance sampling)
  2) Doing *sequence-level* clipping, rewarding, and optimization
  3) Using *group-normalized rewards* across multiple responses for the same prompt as advantages
These points are described in the GSPO paper and the Qwen blog. :contentReference[oaicite:1]{index=1}

Most common GSPO sequence-level ratio form (length-normalized):
  w_i = [ πθ(y_i|x) / π_old(y_i|x) ]^(1/|y_i|)
      = exp( (logπθ(y_i|x) - logπ_old(y_i|x)) / |y_i| )
This exact form is also summarized in several GSPO explainers/docs. :contentReference[oaicite:2]{index=2}

What this script is
-------------------
A clear, hackable GSPO loop for LLM post-training:
- sample K completions per prompt (group)
- compute rewards r_{i} = R(prompt, completion)
- compute group-relative advantages A_i (mean/std normalize within group)
- compute GSPO sequence ratios w_i (length-normalized)
- apply PPO-style clipped surrogate at *sequence level*
- backprop through policy only (no critic/value model)

What this script is NOT
-----------------------
- Not distributed / not vLLM / not production rollouts
- Not multi-epoch on the same rollouts (you can add that)
- Reward function is a toy verifier—replace with real verifiable reward (unit tests, checkers)

Dependencies
------------
pip install -U torch transformers

Practical notes
---------------
- For real training: use an SFT checkpoint as initialization
- Add LoRA/QLoRA + bf16 + gradient checkpointing for 7B+ models
- Use a real verifier reward (math check, code tests, etc.) for RLVR

"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

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
# 1) Hyperparameters (GSPO)
# -----------------------------
@dataclass
class GSPOConfig:
    # Start from an SFT model checkpoint in real runs (instruction-tuned).
    model_name: str = "gpt2"

    # Optional frozen reference (for KL regularization like RLHF).
    # GSPO itself focuses on sequence-level importance ratio vs π_old;
    # KL-to-ref is an *additional* stabilizer in many RLHF setups.
    ref_model_name: Optional[str] = None
    beta_kl: float = 0.0  # set >0 to penalize drift toward ref model

    # Rollout sampling
    group_size_K: int = 8              # number of completions per prompt
    batch_prompts: int = 2             # prompts per training step
    train_steps: int = 200

    # Generation controls (exploration)
    max_prompt_len: int = 256
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95

    # GSPO clipping (sequence-level)
    clip_eps: float = 0.2              # clip w into [1-eps, 1+eps]

    # Advantage normalization inside group
    adv_norm: bool = True
    adv_eps: float = 1e-6

    # Optimization
    lr: float = 1e-5
    grad_clip: float = 1.0


cfg = GSPOConfig()


# -----------------------------
# 2) Load tokenizer / models
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
policy.train()

# Optional frozen reference for KL regularization
ref = None
if cfg.beta_kl > 0:
    ref_name = cfg.ref_model_name or cfg.model_name
    ref = AutoModelForCausalLM.from_pretrained(ref_name).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

optimizer = optim.AdamW(policy.parameters(), lr=cfg.lr)


# -----------------------------
# 3) Prompt “dataset” (toy)
# -----------------------------
PROMPTS = [
    "Solve: If 3x + 5 = 20, what is x? Give only the final number.",
    "Compute: 17 * 19. Give only the final number.",
    "Write exactly JSON: {\"answer\": <number>}. Compute 12^2.",
]


# -----------------------------
# 4) Reward function (toy verifier)
# -----------------------------
def extract_first_int(text: str) -> Optional[int]:
    m = re.search(r"-?\d+", text.strip())
    return int(m.group(0)) if m else None

def reward_fn(prompt: str, completion: str) -> float:
    """
    Replace with a REAL verifier reward for your task:
      - math verifier
      - code sandbox + unit tests
      - constrained formatting checks
      - etc.

    For RLVR / reasoning, verifiable reward is the critical ingredient.
    """
    gold = {
        PROMPTS[0]: 5,      # 3x+5=20
        PROMPTS[1]: 323,    # 17*19
        PROMPTS[2]: 144,    # 12^2
    }

    # Strict JSON requirement for the third prompt
    if "exactly JSON" in prompt:
        ok = re.fullmatch(r'\{\s*"answer"\s*:\s*-?\d+\s*\}', completion.strip()) is not None
        if not ok:
            return 0.0

    pred = extract_first_int(completion)
    if pred is None:
        return 0.0
    return 1.0 if pred == gold[prompt] else 0.0


# -----------------------------
# 5) Generation (rollouts)
# -----------------------------
@torch.no_grad()
def sample_completions(prompt: str, K: int) -> List[str]:
    """
    Sample K completions for a single prompt from the *current policy*.

    In real RL infrastructure, you'd separate:
      - rollout workers (fast inference engine, e.g., vLLM)
      - training workers (backprop)
    Here we keep it single-process for clarity.
    """
    enc = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_prompt_len,
    ).to(device)

    completions = []
    for _ in range(K):
        out_ids = policy.generate(
            **enc,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # naive strip prompt
        comp = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
        completions.append(comp)
    return completions


# -----------------------------
# 6) Tokenization + log-prob sums
# -----------------------------
def build_prompt_completion(prompt: str, completion: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build concatenated token ids and a mask that marks completion tokens.

    ids = [prompt_ids] + [completion_ids] + [EOS]
    mask = 0 for prompt tokens, 1 for completion tokens

    We'll compute logπ(y|x) by summing token log-probs for completion tokens only.
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
    Compute logπ(y|x) as the sum of log-probabilities over completion tokens.

    Causal LM alignment:
      logits[:, t] predicts ids[t+1]
    So:
      logits[:-1] aligns with labels[1:]
      mask[1:] aligns with those labels

    Returns:
      scalar log-prob sum for the completion tokens
    """
    input_ids = ids.unsqueeze(0)                    # [1, T]
    attn = torch.ones_like(input_ids, device=device)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]                 # [1, T-1, V]
    labels = input_ids[:, 1:]                      # [1, T-1]
    m = mask[1:].unsqueeze(0)                      # [1, T-1]

    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    return (token_logp * m).sum(dim=-1).squeeze(0)                  # scalar

def completion_length(mask: torch.Tensor) -> torch.Tensor:
    """
    |y| used for length normalization in GSPO ratio.
    Here completion length is number of completion tokens contributing to logprob:
      sum(mask[1:]) (aligned with label positions)
    """
    return mask[1:].sum()


# -----------------------------
# 7) GSPO: sequence-level ratio + clipping + loss
# -----------------------------
def gspo_sequence_ratio(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    y_len: torch.Tensor,
) -> torch.Tensor:
    """
    GSPO uses a *sequence-level* importance ratio.

    Length-normalized ratio (common in GSPO discussions/docs):
      w = exp( (logp_new - logp_old) / |y| )
    This is the per-token geometric mean of token-level ratios, but crucially,
    it is defined as a *sequence likelihood* ratio and is clipped at sequence level,
    which avoids token-level high-variance noise accumulation. :contentReference[oaicite:3]{index=3}
    """
    return torch.exp((logp_new - logp_old) / torch.clamp(y_len, min=1.0))

def gspo_loss(
    w: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """
    PPO-style clipped surrogate, but at the *sequence level*:

      L = -E[ min( w*A, clip(w, 1-eps, 1+eps)*A ) ]

    Interpretation when treating the WHOLE sequence y as one "action":
    - w is the importance ratio for that action (sequence)
    - A is the sequence-level advantage (from reward)
    - We clip w so policy doesn't change too much on this sequence
    """
    w_clip = torch.clamp(w, 1.0 - clip_eps, 1.0 + clip_eps)
    s1 = w * advantage
    s2 = w_clip * advantage
    return -torch.mean(torch.min(s1, s2))


# -----------------------------
# 8) Training loop (one-step updates for clarity)
# -----------------------------
"""
Implementation detail: π_old
----------------------------
In full PPO-like algorithms, you:
  1) snapshot π_old at rollout time
  2) do several SGD epochs using (logp_old) computed from the snapshot
Here, for simplicity, we treat π_old as "policy before the update of this step"
and compute logp_old under no_grad BEFORE computing the loss.

For higher fidelity:
- keep a separate frozen copy policy_old = deepcopy(policy) each iteration
- compute logp_old with policy_old
- optionally do multiple optimization epochs over the same rollouts
"""

for step in range(1, cfg.train_steps + 1):
    batch_prompts = random.sample(PROMPTS, k=min(cfg.batch_prompts, len(PROMPTS)))
    step_losses = []
    step_rewards = []

    for prompt in batch_prompts:
        # ---- 1) Sample a GROUP of K completions (rollouts) ----
        completions = sample_completions(prompt, K=cfg.group_size_K)

        # ---- 2) Compute rewards for each completion ----
        rewards = torch.tensor([reward_fn(prompt, c) for c in completions],
                               dtype=torch.float32, device=device)
        step_rewards.append(rewards.mean().item())

        # ---- 3) Group-normalize rewards to get advantages ----
        # GSPO uses normalized rewards across multiple responses to the same query
        # as "advantages" (baseline = group mean). :contentReference[oaicite:4]{index=4}
        baseline = rewards.mean()
        if cfg.adv_norm:
            std = rewards.std(unbiased=False) + cfg.adv_eps
            adv = (rewards - baseline) / std
        else:
            adv = (rewards - baseline)

        # ---- 4) Compute logp_old under π_old (no gradients) ----
        with torch.no_grad():
            logp_old_list, ylen_list = [], []
            for c in completions:
                ids, m = build_prompt_completion(prompt, c)
                logp_old_list.append(logprob_sum_completion(policy, ids, m))
                ylen_list.append(completion_length(m))
            logp_old = torch.stack(logp_old_list)     # [K]
            y_len = torch.stack(ylen_list)            # [K]

        # ---- 5) Compute logp_new under current πθ (with gradients) ----
        logp_new_list = []
        for c in completions:
            ids, m = build_prompt_completion(prompt, c)
            logp_new_list.append(logprob_sum_completion(policy, ids, m))
        logp_new = torch.stack(logp_new_list)         # [K]

        # ---- 6) Sequence-level ratio w_i (GSPO) ----
        w = gspo_sequence_ratio(logp_new=logp_new, logp_old=logp_old, y_len=y_len)

        # ---- 7) Optional KL regularization toward a frozen reference π_ref ----
        # Many RLHF pipelines penalize deviation from a reference model.
        # We apply it as *advantage shaping* (subtract beta * KL proxy).
        if ref is not None and cfg.beta_kl > 0:
            with torch.no_grad():
                logp_ref_list = []
                for c in completions:
                    ids, m = build_prompt_completion(prompt, c)
                    logp_ref_list.append(logprob_sum_completion(ref, ids, m))
                logp_ref = torch.stack(logp_ref_list)  # [K]
            # KL proxy on sampled sequences:
            #   E[logπθ - logπref]
            kl_proxy = (logp_new.detach() - logp_ref) / torch.clamp(y_len, min=1.0)
            adv = adv - cfg.beta_kl * kl_proxy

        # ---- 8) GSPO clipped objective (sequence-level) ----
        loss = gspo_loss(w=w, advantage=adv, clip_eps=cfg.clip_eps)
        step_losses.append(loss)

    # ---- Backprop ----
    total_loss = torch.stack(step_losses).mean()
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
    optimizer.step()

    # ---- Logging ----
    if step % 10 == 0:
        print(
            f"[step {step:04d}] loss={total_loss.item():.4f} "
            f"avg_group_reward={sum(step_rewards)/len(step_rewards):.4f}"
        )

print("GSPO training finished.")


"""
How GSPO differs from GRPO in one line (important)
-------------------------------------------------
- GRPO: token-level ratios => variance grows with length, clipping can amplify noise
- GSPO: sequence-level ratio from sequence likelihood, clipped at sequence level => more stable,
  especially for long responses and MoE training. :contentReference[oaicite:5]{index=5}

What to do next for a real run
------------------------------
1) Replace reward_fn with a verifiable checker (math/coding/format + correctness)
2) Implement explicit π_old snapshot per iteration (deepcopy(policy))
3) Add multiple epochs over the same rollouts (like PPO) for better sample efficiency
4) Use LoRA/QLoRA + bf16 + gradient checkpointing
5) Use a fast sampler (vLLM) and disaggregated rollout/training workers
"""
