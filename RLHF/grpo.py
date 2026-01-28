"""
GRPO (Group Relative Policy Optimization) — practical LLM post-training with TRL
================================================================================

What GRPO does (conceptually)
-----------------------------
Given a prompt x, GRPO samples a *group* of K responses:
    y_1, y_2, ..., y_K ~ π_theta_old(. | x)

Then it scores each response with a reward function:
    r_i = R(x, y_i)

Instead of training a value function (critic) to estimate a baseline b(x),
GRPO uses the group statistics as the baseline, e.g.:
    b = mean(r_1..r_K)
(or sometimes normalized with std)

Then it computes a *group-relative advantage* for each response:
    A_i = r_i - b
(or A_i = (r_i - mean)/std for normalization)

Finally it performs a PPO-style policy update with clipping:
    ratio_i = π_theta(y_i|x) / π_theta_old(y_i|x)
    L = E_i[ min( ratio_i * A_i, clip(ratio_i, 1-eps, 1+eps) * A_i ) ]
Optionally add KL regularization toward a reference policy π_ref (often the SFT model).

Key difference vs PPO-RLHF:
- PPO-RLHF: needs a critic (value head) to compute advantages (or at least learned baseline)
- GRPO: no critic; advantage comes from group-relative rewards => memory-efficient
  (this is a core point from DeepSeekMath). :contentReference[oaicite:2]{index=2}

Implementation strategy (TRL)
-----------------------------
TRL provides GRPOTrainer + GRPOConfig and handles:
- sampling multiple generations per prompt (group size K)
- computing old-policy logprobs
- computing rewards (via your reward functions)
- building group baseline and group advantages
- applying PPO-style clipped loss updates

See TRL GRPOTrainer docs. :contentReference[oaicite:3]{index=3}

Dependencies
------------
pip install -U torch transformers datasets accelerate trl

(For real models you likely also want: peft, bitsandbytes, flash-attn, deepspeed)

Notes
-----
- This script is *LLM-centric*. If you want GRPO in classic Gym RL, tell me and I’ll adapt.
- For best results: start from an SFT model checkpoint (policy_init).
- Reward functions can be "verifiable rewards" (math correctness, unit tests, exact match)
  which is where GRPO shines in practice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

# TRL provides GRPOTrainer / GRPOConfig
from trl import GRPOTrainer, GRPOConfig


# --------------------------
# 0) Config you will edit
# --------------------------
@dataclass
class Args:
    # Start from an instruction-tuned/SFT model if you have it
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # small example; replace with your SFT checkpoint

    # Optional: reference model for KL regularization (often same as SFT init)
    # If you don't want KL, TRL can run with beta=0.0 (docs mention this default choice). :contentReference[oaicite:4]{index=4}
    ref_model_name: str | None = None

    # Group size K: number of responses sampled per prompt (the "group")
    num_generations: int = 8

    # Sequence lengths
    max_prompt_length: int = 512
    max_completion_length: int = 256

    # GRPO / PPO-ish knobs
    clip_eps: float = 0.2            # PPO clipping range
    beta_kl: float = 0.0             # KL coefficient (0.0 disables KL term)
    learning_rate: float = 2e-6

    # Training regime
    per_device_train_batch_size: int = 1   # prompts per device per step (each prompt expands into K generations)
    gradient_accumulation_steps: int = 8
    max_steps: int = 200

    # Logging / saving
    output_dir: str = "./grpo_out"
    logging_steps: int = 10
    save_steps: int = 100

    # Precision
    bf16: bool = torch.cuda.is_available()
    fp16: bool = False


args = Args()


# ------------------------------------
# 1) Minimal dataset of prompts (x)
# ------------------------------------
"""
GRPO needs prompts; for each prompt it will sample K completions.

In practice, you’ll use:
- math problems, coding prompts, tool-use tasks, etc.
- or your own instruction dataset

Here we create a tiny dataset inline for demonstration.
"""

prompts = [
    "Solve: If 3x + 5 = 20, what is x? Give only the final number.",
    "Solve: What is 17 * 19? Give only the final number.",
    "Format exactly as JSON: {\"answer\": <number>}. Compute 12^2.",
]

train_ds = Dataset.from_dict({"prompt": prompts})


# ------------------------------------
# 2) Tokenizer + policy model
# ------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

# Many chat models define special padding; ensure we have a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16 if args.bf16 else None,
    device_map="auto",
)

# Optional reference model for KL term
# - If you set beta_kl>0, you usually want a frozen reference (e.g., SFT model snapshot)
# - If beta_kl==0.0, you can omit it
ref_model = None
if args.ref_model_name is not None:
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map="auto",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)


# ------------------------------------
# 3) Reward functions (the "R(x,y)")
# ------------------------------------
"""
GRPO is only as good as your reward.
Common reward types:
- Verifiable reward: exact match / unit tests / execution success / checker
- Format reward: strict output constraints (JSON, tags, schema)
- Safety reward: refusal correctness, policy constraints (careful, can be gamed)

Below we implement two simple reward functions:
A) correctness_reward: checks if final numeric answer is correct for known prompts
B) format_reward: rewards if output matches a specific required format

TRL GRPOTrainer can accept a list of reward functions. Each function typically returns:
- a list[float] rewards aligned with the generated samples
"""

def extract_first_int(text: str) -> int | None:
    m = re.search(r"-?\d+", text.strip())
    return int(m.group(0)) if m else None

def correctness_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    A toy verifiable reward:
    - we know the correct answer for each prompt in our mini dataset
    - score 1.0 if correct, else 0.0

    Real-world: replace this with a proper checker:
    - math parser + verifier
    - code execution sandbox + unit tests
    - symbolic solver, etc.
    """
    # Map known prompt -> correct integer answer
    gold = {
        prompts[0]: 5,          # 3x + 5 = 20 => x=5
        prompts[1]: 323,        # 17*19=323
        prompts[2]: 144,        # 12^2=144 (but required JSON format too)
    }

    rewards = []
    for p, c in zip(prompts, completions):
        pred = extract_first_int(c)
        rewards.append(1.0 if (pred is not None and pred == gold.get(p, None)) else 0.0)
    return rewards

def format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Reward outputs that match a strict format.
    For the JSON prompt, require: {"answer": <number>}
    For other prompts, just give a small neutral reward (0.0).

    Format rewards are important in GRPO/RL for “reasoning with constraints”.
    """
    rewards = []
    for p, c in zip(prompts, completions):
        if "Format exactly as JSON" in p:
            # Very strict pattern: {"answer": 144} with optional whitespace
            ok = re.fullmatch(r'\{\s*"answer"\s*:\s*-?\d+\s*\}', c.strip()) is not None
            rewards.append(1.0 if ok else 0.0)
        else:
            rewards.append(0.0)
    return rewards


reward_fns = [correctness_reward, format_reward]


# ------------------------------------
# 4) GRPO training configuration
# ------------------------------------
"""
Important GRPO knobs (interpretation):
- num_generations (K): group size; larger => better baseline estimate, more exploration,
  but higher compute. Typical 4–16.
- beta (KL coef): if >0, discourages drifting too far from reference policy.
  Some setups use beta=0; TRL docs mention beta=0.0 default motivated by recent studies. :contentReference[oaicite:5]{index=5}
- cliprange: PPO-style clipping epsilon.
- max_(prompt|completion)_length: controls rollout length, memory.

The effective "samples per step" is:
  batch_size_prompts * num_generations
So keep per_device_train_batch_size small if K is large.
"""

training_args = GRPOConfig(
    output_dir=args.output_dir,

    # Sampling / rollout
    num_generations=args.num_generations,
    max_prompt_length=args.max_prompt_length,
    max_completion_length=args.max_completion_length,

    # Optimization
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,

    # PPO-ish stabilization
    cliprange=args.clip_eps,      # name in TRL config is typically cliprange
    beta=args.beta_kl,            # KL coefficient toward ref policy (if ref exists)

    # Training length
    max_steps=args.max_steps,

    # Logging / saving
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    report_to=[],                 # set ["wandb"] or ["tensorboard"] if you want

    # Precision
    bf16=args.bf16,
    fp16=args.fp16,

    # Generation controls (optional; defaults are fine but you can tune)
    # temperature=0.7,
    # top_p=0.95,
)


# ------------------------------------
# 5) Build GRPOTrainer
# ------------------------------------
"""
GRPOTrainer will:
1) Take a batch of prompts (size B)
2) Generate K completions per prompt => total B*K samples
3) Compute reward for each completion using your reward_fns
4) For each prompt-group:
     baseline = mean(rewards in group)
     advantages = rewards - baseline (optionally normalized)
5) Compute old and new logprobs and apply PPO-style clipped policy gradient update
   (critic-free; no value head)
"""

trainer = GRPOTrainer(
    model=policy,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer,
    reward_funcs=reward_fns,
    # If you want KL regularization, supply ref_model and set beta>0
    ref_model=ref_model,
)


# ------------------------------------
# 6) Train
# ------------------------------------
"""
During training, watch for:
- reward mean increasing
- group advantage distribution (should not collapse to all zeros)
- KL (if beta>0): if it explodes, lower LR or increase beta
- mode collapse: outputs become identical; fix with higher temperature / entropy-like incentives

A known GRPO corner case:
- if ALL K samples are equally bad (all rewards identical), advantages become ~0
  => little/no gradient signal for that prompt (people call this "all-negative group").
  This is why verifiable rewards + good sampling diversity matter.
"""
trainer.train()

# Save final model
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print(f"[Done] Saved GRPO-trained model to: {args.output_dir}")
