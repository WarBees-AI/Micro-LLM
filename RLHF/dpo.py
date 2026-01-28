"""
DPO (Direct Preference Optimization) — fully-commented, practical PyTorch + HuggingFace.

This script shows *real* DPO training mechanics:
- No reward model
- No PPO / rollouts
- Just preference pairs (chosen vs rejected) and a frozen reference policy

Core idea (high-level)
----------------------
Given a prompt x, and two responses:
  y+ (chosen, preferred by humans)
  y- (rejected, less preferred)

DPO trains the policy πθ to increase probability of y+ relative to y-,
*while anchoring to a fixed reference model* πref (usually the SFT model).

It can be derived from the RLHF objective with a KL regularizer, and yields
a simple supervised-style loss with strong alignment performance.

The DPO objective (most common form)
------------------------------------
Let:
  Δlogπθ = log πθ(y+|x) - log πθ(y-|x)
  Δlogπref = log πref(y+|x) - log πref(y-|x)

Then DPO loss per example:
  L_DPO = - log σ( β * ( Δlogπθ - Δlogπref ) )

where:
- σ is sigmoid
- β > 0 is an inverse temperature / strength parameter
  Larger β pushes stronger separation between chosen and rejected.
- Δlogπref acts like a baseline: we learn *relative* preference improvements
  over the reference, which stabilizes training and prevents drifting.

Intuition
---------
- If policy already prefers chosen over rejected more than reference does,
  then (Δlogπθ - Δlogπref) is positive -> σ(.) is high -> loss is small.
- If policy prefers rejected too much, loss increases -> gradient pushes
  to increase chosen probability and/or decrease rejected probability.

What you need
-------------
1) A base policy model to train (often initialized from SFT checkpoint)
2) A frozen reference model (same architecture/tokenizer)
3) Preference dataset with fields: prompt, chosen, rejected

Dependencies
-----------
pip install torch transformers datasets accelerate peft

Notes for production
--------------------
- Usually you train with LoRA/QLoRA to save VRAM.
- Use proper chat templates for chat models (Qwen/Llama).
- Use "prompt masking" so the loss only applies to response tokens, not prompt tokens.
  (This script does that.)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup


# -----------------------------
# 0) Reproducibility / device
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] device={device}")


# -----------------------------
# 1) DPO hyperparameters
# -----------------------------
@dataclass
class DPOArgs:
    # In real work: use your SFT checkpoint (e.g., Qwen2.5, Llama-3-Instruct, Yi, etc.)
    policy_model_name: str = "gpt2"
    ref_model_name: str = "gpt2"

    # DPO strength (inverse temperature)
    beta: float = 0.1

    # sequence limits
    max_prompt_len: int = 256
    max_response_len: int = 256
    max_total_len: int = 512

    # training
    lr: float = 5e-6
    batch_size: int = 2
    num_epochs: int = 1
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0

    # logging
    log_every: int = 10


args = DPOArgs()


# ---------------------------------------------------
# 2) Preference dataset (prompt, chosen, rejected)
# ---------------------------------------------------
class PreferenceDataset(Dataset):
    """
    Real DPO training needs preference pairs.

    Each item:
      - prompt: instruction/context
      - chosen: preferred completion
      - rejected: less preferred completion

    In practice, you’ll load from:
      - Anthropic HH (helpful/harmless)
      - OpenAI preference data (if available)
      - Your internal annotation pipeline
      - Synth preferences (careful: can bake model biases)
    """

    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# A tiny toy dataset for demonstration.
# Replace with real preference pairs.
toy_data = [
    {
        "prompt": "Explain RLHF in one paragraph.",
        "chosen": "RLHF aligns a language model by learning human preferences and optimizing the model to produce outputs humans rate higher.",
        "rejected": "RLHF is when you just train on more data until it behaves.",
    },
    {
        "prompt": "Give a safe refusal to a request for illegal hacking.",
        "chosen": "I can’t help with hacking or illegal access. If you’re securing your own systems, I can suggest defensive best practices.",
        "rejected": "Sure—first scan ports and try default passwords.",
    },
]

dataset = PreferenceDataset(toy_data)


# -----------------------------
# 3) Tokenizer + models
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.policy_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Policy model: trainable
policy = AutoModelForCausalLM.from_pretrained(args.policy_model_name).to(device)
policy.train()

# Reference model: frozen (no gradients)
ref = AutoModelForCausalLM.from_pretrained(args.ref_model_name).to(device)
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)


# ---------------------------------------------------------
# 4) Utility: build input IDs and response-only loss mask
# ---------------------------------------------------------
def build_prompt_response_tokens(prompt: str, response: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    We construct a single concatenated sequence:
      [prompt_tokens] + [response_tokens]

    DPO needs log-prob of the *response tokens conditioned on the prompt*.
    Therefore:
      - We compute logprobs for all tokens
      - But we ONLY sum logprobs over response tokens (mask prompt tokens out)

    Returns:
      input_ids: shape [seq]
      response_mask: shape [seq] where 1 for response tokens, 0 for prompt tokens

    Important detail:
    - For causal LM, probability of token t is predicted at position t-1.
      When computing logprob of a token, we align logits[:, :-1] with labels[:, 1:].
    - We will create a mask aligned to labels positions accordingly later.
    """
    # Tokenize prompt and response separately to know boundary.
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False, truncation=True, max_length=args.max_prompt_len
    )["input_ids"]

    resp_ids = tokenizer(
        response, add_special_tokens=False, truncation=True, max_length=args.max_response_len
    )["input_ids"]

    # Optionally add EOS to response (often beneficial)
    resp_ids = resp_ids + [tokenizer.eos_token_id]

    # Concatenate and truncate to max_total_len
    input_ids = (prompt_ids + resp_ids)[: args.max_total_len]

    # Response mask: 0 for prompt part, 1 for response part
    # But if truncation cuts into response, mask should reflect that.
    mask = [0] * min(len(prompt_ids), len(input_ids))
    remaining = len(input_ids) - len(mask)
    mask += [1] * max(0, remaining)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    """
    Create a padded batch with:
      - chosen_input_ids, chosen_attention_mask, chosen_response_mask
      - rejected_input_ids, rejected_attention_mask, rejected_response_mask

    Response mask is used to sum logprobs only over response tokens.
    """
    chosen_ids, chosen_rmask = [], []
    rejected_ids, rejected_rmask = [], []

    for ex in batch:
        c_ids, c_mask = build_prompt_response_tokens(ex["prompt"], ex["chosen"])
        r_ids, r_mask = build_prompt_response_tokens(ex["prompt"], ex["rejected"])
        chosen_ids.append(c_ids)
        chosen_rmask.append(c_mask)
        rejected_ids.append(r_ids)
        rejected_rmask.append(r_mask)

    # Pad to max length in this batch
    def pad(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
            attn[i, : s.size(0)] = 1
        return out, attn

    chosen_input_ids, chosen_attn = pad(chosen_ids, tokenizer.pad_token_id)
    rejected_input_ids, rejected_attn = pad(rejected_ids, tokenizer.pad_token_id)

    # Pad response masks with 0s (prompt mask=0, pad mask=0)
    def pad_mask(ms: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(m.size(0) for m in ms)
        out = torch.zeros((len(ms), max_len), dtype=torch.float32)
        for i, m in enumerate(ms):
            out[i, : m.size(0)] = m
        return out

    chosen_rmask = pad_mask(chosen_rmask)
    rejected_rmask = pad_mask(rejected_rmask)

    return {
        "chosen_input_ids": chosen_input_ids.to(device),
        "chosen_attention_mask": chosen_attn.to(device),
        "chosen_response_mask": chosen_rmask.to(device),
        "rejected_input_ids": rejected_input_ids.to(device),
        "rejected_attention_mask": rejected_attn.to(device),
        "rejected_response_mask": rejected_rmask.to(device),
    }


loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)


# ---------------------------------------------------------
# 5) Core: compute sequence log-prob for response tokens
# ---------------------------------------------------------
def response_logprob_sum(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log π(y|x) summed over response tokens only.

    Shapes:
      input_ids:      [B, T]
      attention_mask: [B, T]
      response_mask:  [B, T]  (1 for response tokens; 0 for prompt tokens & pad)

    For causal LM:
      - logits at position t predict token at t+1
      - so we align:
          logits[:, :-1, :] with labels = input_ids[:, 1:]
      - we must also align masks accordingly:
          response_mask[:, 1:] is the mask over label positions
          attention_mask[:, 1:] removes pad tokens from labels

    Return:
      logp_sum: [B]  (sum over response tokens for each sample)
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B, T, V]

    # Shift for next-token prediction
    logits = logits[:, :-1, :]             # predicts tokens 1..T-1
    labels = input_ids[:, 1:]              # actual tokens 1..T-1
    attn = attention_mask[:, 1:].float()
    rmask = response_mask[:, 1:]           # response tokens among labels

    # Combine masks:
    # - only positions that are response tokens
    # - and not padding
    mask = rmask * attn  # [B, T-1]

    # Log-softmax to get log-prob distribution over vocab
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, T-1, V]

    # Gather log-prob for the actual labels
    # labels: [B, T-1] -> gather -> [B, T-1]
    token_logp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Sum only over masked positions (response tokens)
    logp_sum = (token_logp * mask).sum(dim=-1)  # [B]
    return logp_sum


# ---------------------------------------------------------
# 6) DPO loss: -log σ( β * (Δlogπθ - Δlogπref) )
# ---------------------------------------------------------
def dpo_loss(
    policy_logp_chosen: torch.Tensor,
    policy_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Compute DPO loss per batch.

    Definitions:
      Δlogπθ   = logπθ(y+|x) - logπθ(y-|x)
      Δlogπref = logπref(y+|x) - logπref(y-|x)

    Then:
      z = β * (Δlogπθ - Δlogπref)
      L = -log σ(z) = softplus(-z)

    Using softplus is numerically stable.

    Return:
      scalar loss (mean over batch)
    """
    delta_policy = policy_logp_chosen - policy_logp_rejected
    delta_ref = ref_logp_chosen - ref_logp_rejected
    z = beta * (delta_policy - delta_ref)

    # -log(sigmoid(z)) = softplus(-z)
    loss = torch.nn.functional.softplus(-z).mean()
    return loss


# -----------------------------
# 7) Optimizer / scheduler
# -----------------------------
optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

total_steps = len(loader) * args.num_epochs
warmup_steps = int(args.warmup_ratio * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


# -----------------------------
# 8) Training loop
# -----------------------------
"""
Training loop details:
- For each batch of preference pairs:
  1) Compute policy logp sums for chosen and rejected
  2) Compute reference logp sums for chosen and rejected (no grad)
  3) Compute DPO loss
  4) Backprop through policy only
  5) Step optimizer + scheduler

Key differences vs PPO-RLHF:
- No environment rollouts
- No advantage estimation / value head
- No reward model
- Directly optimizes preference ordering relative to reference

Practical tips:
- Use gradient accumulation for larger effective batch size.
- Use LoRA/QLoRA for large models.
- Monitor: reward margin, chosen-vs-rejected accuracy, KL drift.
"""

step = 0
for epoch in range(args.num_epochs):
    for batch in loader:
        step += 1

        # --- Policy logprobs (with gradients) ---
        pol_c = response_logprob_sum(
            policy,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_response_mask"],
        )
        pol_r = response_logprob_sum(
            policy,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_response_mask"],
        )

        # --- Reference logprobs (no gradients) ---
        with torch.no_grad():
            ref_c = response_logprob_sum(
                ref,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_response_mask"],
            )
            ref_r = response_logprob_sum(
                ref,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_response_mask"],
            )

        # --- DPO loss ---
        loss = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=args.beta)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping stabilizes training (especially with long sequences)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            # Useful diagnostics:
            # - how often policy prefers chosen over rejected (in log-prob space)
            # - margin improvements relative to reference
            with torch.no_grad():
                delta_pol = (pol_c - pol_r)
                delta_ref = (ref_c - ref_r)
                acc = (delta_pol > 0).float().mean().item()
                margin = (delta_pol - delta_ref).mean().item()

            print(
                f"[epoch {epoch+1} step {step}] "
                f"loss={loss.item():.4f} "
                f"pref_acc={acc:.3f} "
                f"margin_vs_ref={margin:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

print("\n[Done] DPO training finished.")


"""
How to use this with a real chat model (Qwen/Llama etc.)
-------------------------------------------------------
1) Apply the model's chat template to prompt+response.
   Example (pseudo):
     prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
     chosen_text = prompt_text + chosen_answer
     rejected_text = prompt_text + rejected_answer

2) Keep "response_mask" = 1 only for answer tokens (not system/user tokens).

3) Use PEFT LoRA:
   from peft import LoraConfig, get_peft_model
   policy = get_peft_model(policy, LoraConfig(...))

4) Batch size and max lengths:
   - DPO is memory heavy because you do 2 forwards (chosen/rejected), plus ref forwards.
   - Use gradient checkpointing, bf16/fp16, and LoRA.

Common pitfalls
--------------
- If you don't mask prompt tokens, the model can "cheat" by optimizing prompt likelihood.
- If chosen/rejected lengths differ a lot, log-prob sums can be biased by length.
  Solutions:
   - Use average log-prob per token (length normalization)
   - Or keep dataset responses similar lengths
- β too high can overfit and reduce diversity; too low may under-align.

If you tell me your target model (Qwen2.5 / Llama-3 / Yi / PanGu) and
your preference data format (JSON fields, chat turns), I can adapt this
into a drop-in training script with:
- proper chat templates
- LoRA/QLoRA
- bf16 mixed precision
- accelerate / deepspeed configuration
- evaluation metrics + saving checkpoints
"""
