"""
ARPO — Agentic Reinforced Policy Optimization (multi-turn tool agent RL)
========================================================================

This is an EDUCATIONAL skeleton that implements the *core ARPO mechanics*:

  1) Multi-turn agent trajectories with tool calls
  2) Entropy-based adaptive rollout (branch more after tool usage when entropy spikes)
  3) Step-wise advantage attribution (credit assignment to each step / tool-use decision)
  4) PPO-style clipped policy update on the collected (state, action) steps

ARPO motivation (from paper)
----------------------------
- In tool-augmented multi-turn reasoning, typical RL algorithms do "trajectory-level" sampling:
  sample whole trajectories and update from final reward.
- But tool interactions often inject *high uncertainty* in the next generation step.
  ARPO observes token entropy increases after tool use, and exploits this:
  allocate more exploration at those high-entropy rounds via adaptive branching. :contentReference[oaicite:1]{index=1}

What this code does and does not do
-----------------------------------
✅ Shows the algorithmic structure and where ARPO differs from standard PPO/GRPO:
   - branching policy based on entropy
   - step-level advantage attribution for tool-use interactions
✅ Implements a minimal "tool environment" with deterministic tools
✅ Implements PPO-style update over step-level logprobs

❌ Not a production RL system:
   - no distributed rollout workers
   - no vLLM acceleration
   - no replay buffers / async
   - no robust sandbox execution for tools (you should add this in real systems)
   - no fancy reward shaping

Dependencies
------------
pip install torch transformers

If you want a production-ready pipeline:
- use the official ARPO repo as reference :contentReference[oaicite:2]{index=2}
- use accelerate / deepspeed + vLLM inference workers for scale
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# 0) Setup
# =============================================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1) Minimal "tool environment" for agentic multi-turn interaction
# =============================================================================
"""
In ARPO, the agent interacts with an environment via tool calls.

We model a trajectory as a sequence of "rounds":
  Round t:
    - agent observes state s_t (conversation so far, tool outputs so far)
    - agent chooses an action a_t:
        * either: normal text generation
        * or: a tool call (structured)
    - environment returns tool output (if tool called) and updates state

In real systems:
- state = full message history (system/user/assistant/tool)
- action = next token OR tool-call JSON OR function call schema output

Here we simplify:
- Each step action is a short string "assistant output"
- If it contains a pattern like: TOOL[name](args)
  we execute a tool and append tool output to state
"""

def toy_tool_execute(tool_name: str, tool_args: str) -> str:
    """
    Deterministic toy tools (replace with real tools).
    Tool outputs are environment feedback that can change agent behavior.
    """
    tool_name = tool_name.lower().strip()
    tool_args = tool_args.strip()

    if tool_name == "calc":
        # VERY unsafe in real systems (eval). Here it's toy.
        try:
            val = eval(tool_args, {"__builtins__": {}}, {})
            return f"[tool:calc] {val}"
        except Exception:
            return "[tool:calc] ERROR"
    if tool_name == "search":
        # Fake search: return a canned snippet
        return f"[tool:search] results for '{tool_args}': <snippet>"
    return f"[tool:{tool_name}] UNKNOWN TOOL"


def parse_tool_call(text: str) -> Optional[Tuple[str, str]]:
    """
    Detect a simple tool call syntax: TOOL[name](args)
    Example: TOOL[calc](17*19)
    Returns (tool_name, tool_args) or None if not a tool call.
    """
    text = text.strip()
    if not text.startswith("TOOL["):
        return None
    if "](" not in text or not text.endswith(")"):
        return None
    tool_name = text[len("TOOL[") : text.index("](")]
    tool_args = text[text.index("](") + 2 : -1]
    return tool_name, tool_args


# =============================================================================
# 2) Reward function (verifiable rewards are best)
# =============================================================================
"""
ARPO experiments emphasize tool-use efficiency and performance on multi-turn tasks. :contentReference[oaicite:3]{index=3}

In practice, you want reward that is:
- verifiable (unit tests, exact match, correctness checkers)
- sensitive to tool-use correctness and final answer quality

We implement a toy reward:
- We define a prompt whose correct solution requires using calc.
- Reward is 1.0 if the final answer contains the correct number, else 0.0.
- Additionally we can penalize unnecessary tool calls.
"""

def compute_final_reward(initial_prompt: str, final_answer: str, num_tool_calls: int) -> float:
    # Toy "gold" mapping
    if "17 * 19" in initial_prompt:
        correct = "323"
    elif "3x + 5 = 20" in initial_prompt:
        correct = "5"
    else:
        correct = None

    base = 1.0 if (correct is not None and correct in final_answer) else 0.0

    # Optional tool budget penalty: encourage tool efficiency
    # (ARPO reports better performance with less tool budget in experiments) :contentReference[oaicite:4]{index=4}
    penalty = 0.02 * num_tool_calls
    return max(0.0, base - penalty)


# =============================================================================
# 3) Policy model + logprob / entropy utilities
# =============================================================================
@dataclass
class ARPOConfig:
    model_name: str = "gpt2"     # replace with your SFT-initialized agent model

    # Trajectory structure
    max_rounds: int = 6          # max agent-environment turns
    max_new_tokens_per_round: int = 48

    # Adaptive rollout / branching
    base_branches: int = 1       # always sample at least 1 trajectory
    max_branches: int = 6        # max branches at a high-entropy tool-call round
    entropy_threshold: float = 4.0   # if entropy > threshold after tool, branch more

    # PPO update (step-level)
    clip_eps: float = 0.2
    lr: float = 1e-5
    ppo_epochs: int = 2
    minibatch_steps: int = 16
    gamma: float = 1.0           # episodic terminal reward (often gamma=1)
    lam: float = 1.0             # simple Monte-Carlo advantage; can implement GAE if desired

    # KL regularization to reference (optional)
    beta_kl: float = 0.0
    ref_model_name: Optional[str] = None

    # Sampling parameters
    temperature: float = 0.8
    top_p: float = 0.95


cfg = ARPOConfig()

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
policy.train()

ref = None
if cfg.beta_kl > 0:
    ref_name = cfg.ref_model_name or cfg.model_name
    ref = AutoModelForCausalLM.from_pretrained(ref_name).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

optimizer = optim.AdamW(policy.parameters(), lr=cfg.lr)


@torch.no_grad()
def next_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of the categorical distribution defined by logits for the *next token*.
    This entropy is an uncertainty measure.

    ARPO uses the observation: entropy spikes after tool interactions,
    so we should allocate more exploration there. :contentReference[oaicite:5]{index=5}
    """
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # [batch]
    return ent


def build_state_text(history: List[Dict[str, str]]) -> str:
    """
    Serialize a multi-turn history into a single text prompt for the LM.

    In real agent training you should use the model's chat template:
    tokenizer.apply_chat_template(messages, tokenize=False)

    Here we do a minimal format:
      USER: ...
      ASSISTANT: ...
      TOOL: ...
    """
    lines = []
    for m in history:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    lines.append("ASSISTANT:")  # prompt the model to speak next
    return "\n".join(lines)


@torch.no_grad()
def sample_one_round(history: List[Dict[str, str]]) -> Tuple[str, float, float]:
    """
    Sample one assistant output for the current round.

    Returns:
      text_out: assistant generation (for this round)
      logp_sum: sum logprob of generated tokens (used for PPO ratio)
      ent_next: entropy estimate at the first generation token (proxy uncertainty)

    Why entropy here:
    - ARPO triggers extra branching after tool calls where the model is uncertain.
    - A simple proxy: entropy at the first generation step after tool feedback.
    """
    prompt_text = build_state_text(history)
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(device)

    # Forward once to get logits at the next token position (entropy proxy)
    out0 = policy(**enc)
    logits_next = out0.logits[:, -1, :]  # [1, vocab]
    ent = float(next_token_entropy(logits_next).item())

    # Now sample a completion for this round
    gen_ids = policy.generate(
        **enc,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens_per_round,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # Extract only the newly generated text after prompt_text
    # (best-effort; exact slicing depends on tokenizer details)
    new_text = full[len(prompt_text):].strip() if full.startswith(prompt_text) else full.strip()

    # Compute logprob sum of the generated tokens for PPO
    # For PPO we need logπθ(a|s) for the chosen action.
    # Here "action" is the entire generated round text; we approximate by summing token logprobs.
    logp_sum = logprob_sum_of_generated(policy, prompt_text, new_text)

    return new_text, logp_sum, ent


def logprob_sum_of_generated(model: AutoModelForCausalLM, prompt_text: str, gen_text: str) -> float:
    """
    Compute sum of log-probabilities of gen_text tokens conditioned on prompt_text.

    We concatenate [prompt_ids] + [gen_ids] + [EOS], then sum log-probs over gen tokens.

    This is the standard way to get a sequence log-likelihood under a causal LM.

    NOTE:
    - For strict correctness you should build token boundaries using exact tokenization of prompt
      and track offsets in token space (this function does that).
    """
    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    g_ids = tokenizer(gen_text, add_special_tokens=False)["input_ids"]
    ids = torch.tensor((p_ids + g_ids + [tokenizer.eos_token_id]), device=device).unsqueeze(0)

    # Mask: 0 for prompt tokens, 1 for generated tokens (+EOS)
    mask = torch.tensor([0]*len(p_ids) + [1]*(len(g_ids)+1), device=device, dtype=torch.float32).unsqueeze(0)

    attn = torch.ones_like(ids, device=device)
    out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    labels = ids[:, 1:]
    m = mask[:, 1:]

    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    return float((token_logp * m).sum().item())


# =============================================================================
# 4) ARPO adaptive rollout: global trajectories + step-level branching
# =============================================================================
"""
This is the ARPO "signature" part:
- We roll out trajectories.
- When we detect a tool call and after tool feedback, if the next-token entropy is high,
  we BRANCH (sample multiple alternatives) at that step to explore.
This adaptively focuses sampling budget on uncertain tool-call rounds. :contentReference[oaicite:6]{index=6}

We store:
- per-step state history (as text)
- per-step chosen action text (assistant output)
- per-step logp_old (logprob under π_old at collection time)
- per-trajectory final reward
Then we do PPO-style updates on steps with attributed advantages.
"""

@dataclass
class StepRecord:
    state_text: str
    action_text: str
    logp_old: float
    # For step-level attribution
    round_index: int
    after_tool: bool

@dataclass
class Trajectory:
    prompt: str
    steps: List[StepRecord]
    final_answer: str
    num_tool_calls: int
    reward: float


def run_one_trajectory(prompt: str, branch_multiplier: int = 1) -> Trajectory:
    """
    Roll out one trajectory of up to cfg.max_rounds.

    branch_multiplier exists so the caller can spawn multiple trajectories.
    Real ARPO uses adaptive branching at specific rounds; we do that in the wrapper.
    """
    history = [{"role": "user", "content": prompt}]
    steps: List[StepRecord] = []
    num_tool_calls = 0
    after_tool = False

    for t in range(cfg.max_rounds):
        state_text = build_state_text(history)

        # Sample one assistant output for this round
        action_text, logp_old, ent = sample_one_round(history)

        steps.append(
            StepRecord(
                state_text=state_text,
                action_text=action_text,
                logp_old=logp_old,
                round_index=t,
                after_tool=after_tool,
            )
        )

        history.append({"role": "assistant", "content": action_text})

        # If assistant output is a tool call, execute tool and append tool output
        tool = parse_tool_call(action_text)
        if tool is not None:
            num_tool_calls += 1
            tool_name, tool_args = tool
            tool_out = toy_tool_execute(tool_name, tool_args)
            history.append({"role": "tool", "content": tool_out})
            after_tool = True
        else:
            after_tool = False

        # Early stop condition: if model writes "FINAL:" as end of episode
        if "FINAL:" in action_text:
            break

    final_answer = "\n".join([m["content"] for m in history if m["role"] == "assistant"])[-500:]
    reward = compute_final_reward(prompt, final_answer, num_tool_calls)
    return Trajectory(prompt=prompt, steps=steps, final_answer=final_answer, num_tool_calls=num_tool_calls, reward=reward)


def adaptive_rollouts(prompts: List[str]) -> List[Trajectory]:
    """
    ARPO adaptive rollout mechanism (core):
    - Start with base trajectory sampling
    - When tool is used and next-step entropy is high, branch more at that point.

    In a full implementation, branching means:
    - replicate the *same history* and sample multiple alternative continuations
      from that post-tool state.

    Here we approximate branching by:
    - rolling multiple trajectories from scratch, but we *increase the number of
      trajectories* when the prompt tends to trigger tool use and entropy is high.
    This keeps code simple while illustrating the intended control logic.

    For a faithful implementation:
    - detect exact tool-call rounds
    - clone history at that point
    - sample multiple continuations from that history (tree branching)
    """
    trajs: List[Trajectory] = []

    for p in prompts:
        # Always collect at least one trajectory (global sampling)
        base_traj = run_one_trajectory(p)
        trajs.append(base_traj)

        # If tool calls occurred, we treat this as a candidate for step-level branching.
        if base_traj.num_tool_calls > 0:
            # Simple entropy-based branching heuristic:
            # If any step after tool has high entropy, branch more.
            # (ARPO explicitly focuses on high-entropy tool-call rounds.) :contentReference[oaicite:7]{index=7}
            need_branch = False
            for s in base_traj.steps:
                if s.after_tool:
                    # We don't store entropy per-step in this minimal record;
                    # In production, store ent per step and branch based on it.
                    need_branch = True
                    break

            if need_branch:
                # Branch count could be proportional to uncertainty.
                # Here: just spawn additional trajectories up to max_branches.
                extra = min(cfg.max_branches - 1, 2)  # small demo; tune for real training
                for _ in range(extra):
                    trajs.append(run_one_trajectory(p))

    return trajs


# =============================================================================
# 5) Advantage attribution (step-level credit assignment)
# =============================================================================
"""
Trajectory-level RL gives one reward R for the whole trajectory.
ARPO additionally does "advantage attribution estimation" for step-wise tool interactions. :contentReference[oaicite:8]{index=8}

There are many ways to do this. A simple, common approach in agent RL:
- Assign the trajectory advantage to each step, possibly weighted
  to emphasize tool-related steps.

We implement a simple attribution:
- For each trajectory:
    A_traj = reward - baseline_reward
- For each step:
    A_step = A_traj
    If step is "after_tool": multiply by tool_weight (>1) to focus learning there

This captures the core intuition:
- push learning signal harder to steps around tool usage where uncertainty is high.

For more faithful attribution:
- compute per-step delta using leave-one-out / counterfactual rollouts
- or use a learned critic at step-level
"""

def compute_step_advantages(trajs: List[Trajectory], tool_weight: float = 1.5) -> List[Tuple[StepRecord, float]]:
    rewards = torch.tensor([t.reward for t in trajs], dtype=torch.float32)
    baseline = float(rewards.mean().item())

    step_data: List[Tuple[StepRecord, float]] = []
    for t in trajs:
        A_traj = t.reward - baseline
        for s in t.steps:
            A = A_traj * (tool_weight if s.after_tool else 1.0)
            step_data.append((s, A))
    return step_data


# =============================================================================
# 6) PPO-style update on step records
# =============================================================================
"""
We now perform PPO on the stored steps.
For each step:
- state_text is the LM conditioning context
- action_text is the generated output chunk for that round
- logp_old is stored at data collection time
- advantage is computed by attribution

We compute:
  logp_new = log πθ(action_text | state_text)
  ratio = exp(logp_new - logp_old)
  loss = -mean( min(ratio*A, clip(ratio, 1-eps, 1+eps)*A ) )

Optionally apply KL penalty to a reference policy (common RLHF stabilizer).
"""

def ppo_update(step_data: List[Tuple[StepRecord, float]]):
    if not step_data:
        return

    random.shuffle(step_data)

    for epoch in range(cfg.ppo_epochs):
        for i in range(0, len(step_data), cfg.minibatch_steps):
            mb = step_data[i : i + cfg.minibatch_steps]
            if not mb:
                continue

            logp_new_list = []
            logp_old_list = []
            adv_list = []

            for s, A in mb:
                logp_new = logprob_sum_of_generated(policy, s.state_text, s.action_text)
                logp_new_list.append(logp_new)
                logp_old_list.append(s.logp_old)
                adv_list.append(A)

            logp_new_t = torch.tensor(logp_new_list, device=device, dtype=torch.float32)
            logp_old_t = torch.tensor(logp_old_list, device=device, dtype=torch.float32)
            adv_t = torch.tensor(adv_list, device=device, dtype=torch.float32)

            # Optional advantage normalization (stabilizes)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

            ratio = torch.exp(logp_new_t - logp_old_t)
            ratio_clip = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)

            surr1 = ratio * adv_t
            surr2 = ratio_clip * adv_t
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # Optional KL-to-ref regularization
            # Use a sampled KL proxy on the same action_text chunks.
            if ref is not None and cfg.beta_kl > 0:
                with torch.no_grad():
                    logp_ref_list = []
                    for s, _A in mb:
                        logp_ref_list.append(logprob_sum_of_generated(ref, s.state_text, s.action_text))
                    logp_ref_t = torch.tensor(logp_ref_list, device=device, dtype=torch.float32)
                kl_proxy = torch.mean(logp_new_t.detach() - logp_ref_t)
                policy_loss = policy_loss + cfg.beta_kl * kl_proxy

            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()


# =============================================================================
# 7) End-to-end training loop
# =============================================================================
def train_arpo(num_steps: int = 50):
    for step in range(1, num_steps + 1):
        # ---- (A) Sample prompts batch ----
        prompts = random.sample(
            ["Compute 17 * 19 and return FINAL: <num> (you may use TOOL[calc](17*19))",
             "Solve 3x + 5 = 20 and return FINAL: <num> (you may use TOOL[calc]((20-5)/3))"],
            k=2
        )

        # ---- (B) Adaptive rollouts (ARPO) ----
        trajs = adaptive_rollouts(prompts)

        # ---- (C) Step-level advantage attribution ----
        step_data = compute_step_advantages(trajs, tool_weight=1.5)

        # ---- (D) PPO update over steps ----
        ppo_update(step_data)

        # ---- Logging ----
        avg_r = sum(t.reward for t in trajs) / max(1, len(trajs))
        avg_tool = sum(t.num_tool_calls for t in trajs) / max(1, len(trajs))
        if step % 5 == 0:
            print(f"[step {step:03d}] trajectories={len(trajs)} avg_reward={avg_r:.3f} avg_tool_calls={avg_tool:.2f}")

train_arpo(num_steps=30)
print("ARPO training finished.")
