"""
RLHF (PPO) training skeleton for a causal LLM — FULLY COMMENTED.

What this script demonstrates (end-to-end pipeline):
1) Load an SFT (reference) policy model (the "actor") and a frozen reference model (pi_ref)
2) Load a Reward Model (RM) that scores (prompt, response) pairs with a scalar reward
3) Run PPO to optimize the policy to maximize reward while staying close to pi_ref via KL penalty

IMPORTANT NOTES
- Real RLHF requires: (a) good prompts distribution, (b) well-trained reward model, (c) safety filters,
  (d) careful hyperparameters, (e) logging/evaluation, (f) robust data pipeline.
- This script is meant to be a *clear, correct template* you can adapt.
- It uses HuggingFace Transformers + TRL (PPOTrainer).
- You must install dependencies and provide a reward model checkpoint.

Dependencies (example):
  pip install torch transformers datasets accelerate trl peft

Typical usage:
  python rlhf_ppo_train.py


- Reward model training code (pairwise preference loss)
- DPO training code (no reward model, simpler and often preferred)
- GRPO-style group normalization variant
"""

import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

# TRL provides PPOTrainer which implements:
# - rollouts: generate responses
# - reward: score responses via reward model
# - PPO update: policy gradient with clipping + KL control
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead


# -----------------------------
# 0) Reproducibility + device
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Info] Using device: {device}")


# -----------------------------------------
# 1) Minimal prompts dataset (replace this)
# -----------------------------------------
class PromptOnlyDataset(Dataset):
    """
    RLHF training typically starts from a "prompt dataset":
    - user instructions/questions
    - could be multi-turn chat context
    - ideally matches the distribution your model will face in production

    In PPO RLHF, each sample provides a prompt; the policy generates a response.
    """

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


# Example prompts. Replace with real instruction data.
prompts = [
    "Explain what RLHF is in simple terms.",
    "Give a concise step-by-step plan to learn transformers.",
    "Summarize the difference between BatchNorm and LayerNorm.",
    "Write a safe refusal to a request for illegal hacking.",
    "Provide an intuitive explanation of KL divergence.",
]

dataset = PromptOnlyDataset(prompts)


# -------------------------------------------------------
# 2) Config: model IDs and RLHF hyperparameters
# -------------------------------------------------------
@dataclass
class TrainArgs:
    # Actor / policy model (should be SFT-tuned ideally)
    policy_model_name: str = "gpt2"  # Replace with your SFT checkpoint

    # Reward model checkpoint: must be a sequence classification model
    # trained to output a scalar reward (1 logit).
    reward_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    # ^ This is NOT a real RLHF reward model; it's a sentiment classifier used as a placeholder.
    # Replace with your actual RM trained on human preferences.

    # Token limits for generation
    max_prompt_length: int = 256
    max_new_tokens: int = 80

    # PPO training params
    batch_size: int = 2             # number of prompts per PPO update step
    mini_batch_size: int = 1        # PPO minibatch
    ppo_epochs: int = 4             # PPO epochs per batch
    learning_rate: float = 1e-5
    target_kl: float = 0.1          # KL control target (approx)
    init_kl_coef: float = 0.2       # KL penalty coefficient (beta)
    seed: int = 42

    # Logging / steps
    total_ppo_steps: int = 10       # demonstration steps; increase for real training
    log_every: int = 1

    # Mixed precision options
    fp16: bool = torch.cuda.is_available()


args = TrainArgs()


# -------------------------------------------------------
# 3) Load tokenizers and models
# -------------------------------------------------------
# Policy tokenizer (actor)
policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model_name)

# For GPT2-like models, pad token is often undefined; PPOTrainer needs padding.
if policy_tokenizer.pad_token is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

# Policy model with a value head:
# PPO requires both:
# - policy logits (to sample actions/tokens)
# - value function V(s) (baseline for advantage estimation)
#
# TRL wraps the base LM with a small ValueHead.
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.policy_model_name
).to(device)

# Reference model: frozen snapshot of the policy used to compute KL(pi || pi_ref)
# In RLHF, this is usually the SFT model checkpoint kept frozen.
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.policy_model_name
).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

# Reward model:
# In classical RLHF, RM is trained from pairwise preferences and outputs scalar reward r(x, y).
# Here we load a sequence classification model as a placeholder.
#
# For a *real* RM:
# - checkpoint should be AutoModelForSequenceClassification with num_labels=1 (one logit)
# - input should represent the (prompt, response) pair, typically concatenated
rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    args.reward_model_name
).to(device)
reward_model.eval()
for p in reward_model.parameters():
    p.requires_grad_(False)


# -------------------------------------------------------
# 4) PPO configuration
# -------------------------------------------------------
ppo_config = PPOConfig(
    model_name=args.policy_model_name,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    mini_batch_size=args.mini_batch_size,
    ppo_epochs=args.ppo_epochs,
    target_kl=args.target_kl,
    init_kl_coef=args.init_kl_coef,
    seed=args.seed,
    # You can also tune:
    # - cliprange (policy clip)
    # - cliprange_value (value function clip)
    # - vf_coef (value loss coefficient)
    # - gamma/lam (GAE parameters) if needed
)


# -------------------------------------------------------
# 5) Helper: tokenize prompts for the policy model
# -------------------------------------------------------
def tokenize_prompts(prompts: List[str]) -> Dict[str, torch.Tensor]:
    """
    Tokenize prompts for generation by the policy.
    We keep prompts to a maximum length to control memory and speed.
    """
    enc = policy_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


# -------------------------------------------------------
# 6) Helper: policy generation for rollouts
# -------------------------------------------------------
@torch.no_grad()
def generate_responses(prompts: List[str]) -> List[str]:
    """
    Generate responses from the current policy.

    In PPO RLHF, this is the rollout step:
      (prompt) -> (response sampled from policy)

    Generation settings matter a LOT.
    - Too deterministic can reduce exploration
    - Too random can destabilize RM scoring and training
    """
    inputs = tokenize_prompts(prompts)

    # Basic decoding. Adjust:
    # - do_sample=True for exploration
    # - top_p, temperature to control diversity
    generated = policy_model.pretrained_model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=policy_tokenizer.pad_token_id,
        eos_token_id=policy_tokenizer.eos_token_id,
    )

    # We only want the "new tokens" as response text.
    # Easiest: decode full text then strip the prompt prefix.
    full_texts = policy_tokenizer.batch_decode(generated, skip_special_tokens=True)

    responses = []
    for prompt, full in zip(prompts, full_texts):
        # naive prompt stripping; for chat templates you'd do structured parsing
        if full.startswith(prompt):
            responses.append(full[len(prompt):].strip())
        else:
            responses.append(full.strip())
    return responses


# -------------------------------------------------------
# 7) Helper: reward model scoring r(prompt, response)
# -------------------------------------------------------
@torch.no_grad()
def score_with_reward_model(prompts: List[str], responses: List[str]) -> torch.Tensor:
    """
    Compute scalar rewards using the reward model.

    In true RLHF:
      - RM is trained on preference comparisons
      - Outputs a scalar logit r(x, y) (higher = better)
    Here:
      - We use a sentiment classifier placeholder (NOT correct RLHF reward semantics)
      - The purpose is to show the mechanics of (x,y)->reward.

    Return:
      rewards: shape [batch]
    """
    # Common approach: concatenate prompt + response with separator
    # RM format is task-specific; match your RM training format.
    joined = [f"Prompt: {p}\nResponse: {r}" for p, r in zip(prompts, responses)]

    enc = rm_tokenizer(
        joined,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = reward_model(**enc)

    # For a real RM (num_labels=1): reward = out.logits.squeeze(-1)
    # For SST-2 placeholder (num_labels=2):
    # - logits correspond to [negative, positive]
    # We'll map reward to "positive logit - negative logit"
    logits = out.logits
    if logits.size(-1) == 1:
        rewards = logits.squeeze(-1)
    else:
        # difference between positive and negative as a scalar "preference-like" score
        rewards = logits[:, 1] - logits[:, 0]

    return rewards.detach()


# -------------------------------------------------------
# 8) Build PPOTrainer
# -------------------------------------------------------
# PPOTrainer expects:
# - policy model (with value head)
# - reference model (frozen)
# - tokenizer (policy tokenizer)
# - dataset (prompts)
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=policy_tokenizer,
    dataset=dataset,
)


# -------------------------------------------------------
# 9) Training loop: RLHF with PPO
# -------------------------------------------------------
"""
High-level PPO RLHF loop:

for each step:
  1) sample a batch of prompts
  2) policy generates responses (rollout)
  3) reward model scores each (prompt,response) -> reward
  4) PPO update step:
       maximize reward - beta * KL(pi || pi_ref)
       (plus value function loss for advantage estimation)

The TRL PPOTrainer handles:
- logprobs collection
- KL computation vs ref_model
- advantage estimation via value head
- clipped PPO objective optimization
"""

for step, batch in enumerate(ppo_trainer.dataloader):
    if step >= args.total_ppo_steps:
        break

    prompts_batch: List[str] = batch["prompt"]

    # 1) Rollout: generate responses
    responses_batch = generate_responses(prompts_batch)

    # 2) Score with Reward Model -> scalar rewards
    rewards = score_with_reward_model(prompts_batch, responses_batch)

    # TRL expects rewards as list of floats (or tensors per sample)
    rewards_list = [r.item() for r in rewards]

    # 3) PPO update
    # PPOTrainer.step takes:
    # - queries: tokenized prompt tensors
    # - responses: tokenized response tensors
    # - rewards: per-sample reward list
    #
    # We must provide token tensors (not raw strings) for queries/responses.
    query_tensors = policy_tokenizer(
        prompts_batch, return_tensors="pt", padding=True, truncation=True,
        max_length=args.max_prompt_length
    )["input_ids"].to(device)

    response_tensors = policy_tokenizer(
        responses_batch, return_tensors="pt", padding=True, truncation=True,
        max_length=args.max_new_tokens  # response length cap
    )["input_ids"].to(device)

    stats = ppo_trainer.step(
        query_tensors=query_tensors,
        response_tensors=response_tensors,
        rewards=rewards_list,
    )

    # 4) Logging (inspect what’s happening)
    if (step + 1) % args.log_every == 0:
        print("\n" + "=" * 80)
        print(f"[Step {step+1}]")
        for i in range(min(len(prompts_batch), 2)):
            print(f"\nPrompt:   {prompts_batch[i]}")
            print(f"Response: {responses_batch[i]}")
            print(f"Reward:   {rewards_list[i]:.4f}")

        # Key PPO stats often include:
        # - objective/kl: KL divergence vs reference model
        # - objective/kl_coef: current KL coefficient
        # - policy/entropy: entropy of the policy distribution
        # - ppo/returns/mean: mean returns
        # - ppo/policy/advantages_mean: mean advantages
        # Exact keys vary by TRL version.
        interesting_keys = [k for k in stats.keys() if any(
            s in k for s in ["kl", "entropy", "returns", "advantages", "loss"]
        )]
        print("\n[Stats]")
        for k in sorted(interesting_keys)[:30]:
            v = stats[k]
            if isinstance(v, (float, int)):
                print(f"  {k:30s}: {v:.6f}")
        print("=" * 80)


# -------------------------------------------------------
# 10) Save the aligned policy model
# -------------------------------------------------------
"""
After PPO, you typically save ONLY the policy model (actor with value head optionally).
For inference you usually save the base causal LM weights (without value head).
TRL provides .save_pretrained; you can also save policy_model.pretrained_model.
"""

save_dir = "./rlhf_ppo_aligned_model"
os.makedirs(save_dir, exist_ok=True)

# Save base LM (recommended for inference)
policy_model.pretrained_model.save_pretrained(save_dir)
policy_tokenizer.save_pretrained(save_dir)

print(f"\n[Done] Saved aligned policy model to: {save_dir}")


"""
NEXT STEPS (Production-grade RLHF checklist)
1) Replace prompts with a real instruction/chat dataset (and proper chat templates).
2) Train a real Reward Model:
   - Collect preference pairs (y+ vs y-) for the same prompt
   - Train RM with pairwise logistic loss
3) Add safety filters:
   - Block disallowed content
   - Penalize policy outputs that violate constraints
4) Tune PPO hyperparameters:
   - KL coefficient scheduling
   - batch sizes, LR, generation temperature/top_p
5) Evaluate:
   - Reward score distributions
   - Helpfulness/hallucination benchmarks
   - Safety benchmarks (jailbreak robustness)
6) Consider DPO (often simpler + more stable than PPO RLHF)

If you tell me what model you’re training (e.g., Qwen2.5, Llama 3, Yi, GPT-oss)
and your GPU constraints, I can rewrite this into a practical script
with PEFT/LoRA + accelerate + proper chat templates.
"""
