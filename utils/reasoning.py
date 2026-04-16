"""Reasoning & Test-Time Compute utilities for Notebook 19.

Implements:
- ReasoningConfig: configuration dataclass for reasoning pipelines
- CoTPromptPipeline: Chain-of-Thought prompting with self-consistency (majority voting)
- ReasoningNode: tree node for MCTS reasoning search
- MCTSReasoner: Monte Carlo Tree Search over reasoning trajectories
- ProcessRewardModel: scores individual reasoning steps in [0, 1] via sigmoid

All implementations use MLX exclusively on Apple Silicon.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ReasoningConfig:
    """Configuration for reasoning and test-time compute pipelines.

    Attributes:
        max_reasoning_steps: Maximum number of CoT reasoning steps to generate.
        num_samples: Number of samples for self-consistency voting.
        branching_factor: Number of child nodes to expand in tree search.
        temperature: Sampling temperature for generation diversity.
    """

    max_reasoning_steps: int = 5
    num_samples: int = 5
    branching_factor: int = 3
    temperature: float = 0.7

    def __post_init__(self) -> None:
        assert self.max_reasoning_steps >= 1, "Need at least 1 reasoning step"
        assert self.num_samples >= 1, "Need at least 1 sample"
        assert self.branching_factor >= 1, "Branching factor must be >= 1"
        assert self.temperature > 0, "Temperature must be positive"


# ============================================================================
# Chain-of-Thought Prompting Pipeline
# ============================================================================


class CoTPromptPipeline:
    """Chain-of-Thought prompting with self-consistency (majority voting).

    Since we don't have a real language model in this educational setting,
    we simulate CoT generation using a simple arithmetic reasoning engine.
    The *algorithms* (CoT decomposition, self-consistency voting) are real —
    only the "model" is simulated for demonstration purposes.

    **Validates: Requirements 5.1, 5.2**
    """

    def __init__(self, config: Optional[ReasoningConfig] = None) -> None:
        self.config = config or ReasoningConfig()

    # ------------------------------------------------------------------
    # Simulated reasoning engine (stands in for a real LM)
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_arithmetic_reasoning(
        a: int, b: int, op: str, temperature: float = 0.7
    ) -> tuple[list[str], int]:
        """Simulate a CoT reasoning chain for simple arithmetic.

        With higher temperature, the simulated "model" is more likely to
        make errors — mimicking how real LMs behave with temperature.

        Returns:
            (reasoning_steps, final_answer)
        """
        # Error probability increases with temperature
        error_prob = min(0.4, temperature * 0.3)

        # Choose a random decomposition strategy
        strategies = ["direct", "decompose", "rearrange"]
        strategy = random.choice(strategies)

        steps: list[str] = []

        if op == "+":
            correct_answer = a + b
            if strategy == "direct":
                steps.append(f"Step 1: Compute {a} + {b} directly.")
                steps.append(f"Step 2: {a} + {b} = {correct_answer}.")
            elif strategy == "decompose":
                # Decompose into tens and units
                a_tens, a_units = a // 10 * 10, a % 10
                b_tens, b_units = b // 10 * 10, b % 10
                partial = a_tens + b_tens
                steps.append(
                    f"Step 1: Break into tens: {a} = {a_tens} + {a_units}, "
                    f"{b} = {b_tens} + {b_units}."
                )
                steps.append(f"Step 2: Add tens: {a_tens} + {b_tens} = {partial}.")
                steps.append(
                    f"Step 3: Add units: {a_units} + {b_units} = {a_units + b_units}."
                )
                steps.append(
                    f"Step 4: Combine: {partial} + {a_units + b_units} = {correct_answer}."
                )
            else:
                steps.append(f"Step 1: Rearrange: {a} + {b} = {b} + {a}.")
                steps.append(f"Step 2: Compute: {b} + {a} = {correct_answer}.")

        elif op == "*":
            correct_answer = a * b
            if strategy == "direct":
                steps.append(f"Step 1: Compute {a} × {b} directly.")
                steps.append(f"Step 2: {a} × {b} = {correct_answer}.")
            elif strategy == "decompose":
                a1 = a // 10 * 10
                a2 = a % 10
                p1 = a1 * b
                p2 = a2 * b
                steps.append(f"Step 1: Decompose {a} = {a1} + {a2}.")
                steps.append(f"Step 2: {a1} × {b} = {p1}.")
                steps.append(f"Step 3: {a2} × {b} = {p2}.")
                steps.append(f"Step 4: {p1} + {p2} = {correct_answer}.")
            else:
                steps.append(f"Step 1: {a} × {b} = {b} × {a}.")
                steps.append(f"Step 2: Compute: {b} × {a} = {correct_answer}.")

        elif op == "-":
            correct_answer = a - b
            steps.append(f"Step 1: Compute {a} - {b}.")
            steps.append(f"Step 2: {a} - {b} = {correct_answer}.")
        else:
            correct_answer = a + b
            steps.append(f"Step 1: Default to addition: {a} + {b} = {correct_answer}.")

        # Simulate errors with probability proportional to temperature
        if random.random() < error_prob:
            # Introduce a small error in the final answer
            error = random.choice([-2, -1, 1, 2])
            final_answer = correct_answer + error
            steps[-1] = steps[-1].replace(str(correct_answer), str(final_answer))
        else:
            final_answer = correct_answer

        return steps, final_answer

    # ------------------------------------------------------------------
    # Core CoT methods
    # ------------------------------------------------------------------

    def generate_with_cot(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> tuple[list[str], int]:
        """Generate a Chain-of-Thought reasoning chain and extract the answer.

        In a real system, this would call model.generate() with a CoT prompt.
        Here we parse the arithmetic problem and simulate reasoning.

        Args:
            prompt: An arithmetic problem string like "What is 17 * 24?"
            temperature: Override config temperature for this call.

        Returns:
            (reasoning_steps, final_answer)
        """
        temp = temperature if temperature is not None else self.config.temperature

        # Parse the arithmetic prompt
        a, b, op = self._parse_arithmetic_prompt(prompt)

        # Generate reasoning chain (simulated)
        steps, answer = self._simulate_arithmetic_reasoning(a, b, op, temp)

        return steps, answer

    def self_consistency(
        self,
        prompt: str,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> tuple[int, dict[int, int], list[list[str]]]:
        """Apply self-consistency: sample N reasoning chains, majority vote.

        This is the real algorithm from Wang et al. (2022):
        1. Sample N independent CoT solutions
        2. Extract the final answer from each
        3. Return the majority-vote answer

        Args:
            prompt: The problem to solve.
            n: Number of samples (default: config.num_samples, minimum 3).
            temperature: Sampling temperature.

        Returns:
            (majority_answer, vote_counts, all_reasoning_chains)
        """
        n = max(3, n or self.config.num_samples)
        temp = temperature if temperature is not None else self.config.temperature

        all_chains: list[list[str]] = []
        all_answers: list[int] = []

        for _ in range(n):
            steps, answer = self.generate_with_cot(prompt, temperature=temp)
            all_chains.append(steps)
            all_answers.append(answer)

        # Majority voting
        vote_counts = dict(Counter(all_answers))
        majority_answer = max(vote_counts, key=lambda k: vote_counts[k])

        return majority_answer, vote_counts, all_chains

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_arithmetic_prompt(prompt: str) -> tuple[int, int, str]:
        """Parse a simple arithmetic prompt into (a, b, operator).

        Handles formats like:
          "What is 17 * 24?"
          "Compute 123 + 456"
          "17 * 24"
        """
        import re

        # Find pattern: number operator number
        pattern = r"(\d+)\s*([+\-*/×x])\s*(\d+)"
        match = re.search(pattern, prompt)
        if match:
            a = int(match.group(1))
            op_raw = match.group(2)
            b = int(match.group(3))
            # Normalize operator
            op_map = {"×": "*", "x": "*", "/": "/", "+": "+", "-": "-", "*": "*"}
            op = op_map.get(op_raw, "+")
            return a, b, op

        # Fallback: return a simple default
        return 17, 24, "*"


# ============================================================================
# MCTS Reasoning
# ============================================================================


@dataclass
class ReasoningNode:
    """A node in the MCTS reasoning tree.

    Each node represents a partial reasoning state. The tree is built
    incrementally by the MCTSReasoner through selection, expansion,
    evaluation, and backpropagation.

    Attributes:
        state: The reasoning text accumulated so far.
        parent: Parent node (None for root).
        children: List of child nodes.
        value: Average reward estimate for this node.
        visits: Number of times this node has been visited.
    """

    state: str
    parent: Optional["ReasoningNode"] = None
    children: list["ReasoningNode"] = field(default_factory=list)
    value: float = 0.0
    visits: int = 0

    def is_leaf(self) -> bool:
        """A leaf node has no children (not yet expanded)."""
        return len(self.children) == 0

    def ucb1(self, exploration_weight: float = 1.414) -> float:
        """Compute UCB1 score for this node.

        UCB1(v) = X̄_v + c × √(ln(N_parent) / N_v)

        where:
          X̄_v = average reward (exploitation)
          c = exploration weight (√2 by default)
          N_parent = parent visit count
          N_v = this node's visit count

        Unvisited nodes return infinity (always explore first).
        """
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return self.value

        exploitation = self.value
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration


class MCTSReasoner:
    """Monte Carlo Tree Search over reasoning trajectories.

    Implements the four MCTS phases:
    1. **Selection** — traverse tree using UCB1 to find a leaf
    2. **Expansion** — generate new reasoning steps from the leaf
    3. **Evaluation** — score new nodes using a reward function
    4. **Backpropagation** — update ancestor statistics

    For educational purposes, we use a simple heuristic evaluator.
    In production (o1, DeepSeek-R1), a trained Process Reward Model
    would replace the evaluator.

    **Validates: Requirements 5.3, 5.4**
    """

    def __init__(
        self,
        config: Optional[ReasoningConfig] = None,
        evaluator: Optional[object] = None,
    ) -> None:
        self.config = config or ReasoningConfig()
        self.evaluator = evaluator  # Optional ProcessRewardModel
        self._step_templates = [
            "Let me break this into parts.",
            "First, I'll identify the key quantities.",
            "I can simplify by factoring.",
            "Applying the distributive property.",
            "Checking my intermediate result.",
            "Combining the partial results.",
            "Verifying the final answer.",
            "Using an alternative approach to confirm.",
        ]

    # ------------------------------------------------------------------
    # Phase 1: Selection — traverse tree using UCB1
    # ------------------------------------------------------------------

    def select(self, root: ReasoningNode) -> ReasoningNode:
        """Select a leaf node by traversing the tree using UCB1.

        At each internal node, pick the child with the highest UCB1 score.
        This balances exploitation (high-value nodes) with exploration
        (rarely-visited nodes).
        """
        node = root
        while not node.is_leaf():
            # Pick child with highest UCB1
            node = max(node.children, key=lambda c: c.ucb1())
        return node

    # ------------------------------------------------------------------
    # Phase 2: Expansion — generate new reasoning steps
    # ------------------------------------------------------------------

    def expand(self, node: ReasoningNode) -> list[ReasoningNode]:
        """Expand a leaf node by generating candidate next reasoning steps.

        In a real system, this calls the LM to generate continuations.
        Here we simulate with template-based reasoning steps.

        Returns:
            List of newly created child nodes.
        """
        new_children: list[ReasoningNode] = []
        bf = self.config.branching_factor

        for i in range(bf):
            # Pick a reasoning step (with some randomness)
            step_idx = (hash(node.state) + i) % len(self._step_templates)
            step_text = self._step_templates[step_idx]

            # Create child with extended state
            child_state = f"{node.state} → {step_text}"
            child = ReasoningNode(
                state=child_state,
                parent=node,
            )
            new_children.append(child)
            node.children.append(child)

        return new_children

    # ------------------------------------------------------------------
    # Phase 3: Evaluation — score nodes
    # ------------------------------------------------------------------

    def evaluate(self, node: ReasoningNode) -> float:
        """Evaluate a reasoning node, returning a score in [0, 1].

        If a ProcessRewardModel is provided, use it. Otherwise, use a
        simple heuristic based on reasoning chain length and diversity.
        """
        if self.evaluator is not None and hasattr(self.evaluator, "score_step"):
            # Use the PRM to score
            context = node.parent.state if node.parent else ""
            step = node.state
            # Extract just the last step
            if " → " in step:
                step = step.split(" → ")[-1]
            score = self.evaluator.score_step(context, step)
            if isinstance(score, mx.array):
                score = score.item()
            return float(score)

        # Heuristic evaluator for educational demos
        # Longer, more diverse reasoning chains score higher
        steps = node.state.split(" → ")
        num_steps = len(steps)

        # Reward for having multiple steps (diminishing returns)
        step_score = 1.0 - math.exp(-0.3 * num_steps)

        # Reward for diversity (unique steps)
        unique_ratio = len(set(steps)) / max(len(steps), 1)

        # Combine with some noise for exploration
        base_score = 0.5 * step_score + 0.5 * unique_ratio
        noise = random.gauss(0, 0.05)
        score = max(0.0, min(1.0, base_score + noise))

        return score

    # ------------------------------------------------------------------
    # Phase 4: Backpropagation — update ancestor statistics
    # ------------------------------------------------------------------

    @staticmethod
    def backpropagate(node: ReasoningNode, value: float) -> None:
        """Backpropagate the evaluation result up to the root.

        Updates visit counts and running average values for all
        ancestors of the evaluated node.
        """
        current: Optional[ReasoningNode] = node
        while current is not None:
            current.visits += 1
            # Incremental mean update
            current.value += (value - current.value) / current.visits
            current = current.parent

    # ------------------------------------------------------------------
    # Full search
    # ------------------------------------------------------------------

    def search(
        self,
        prompt: str,
        budget: Optional[int] = None,
    ) -> tuple[str, ReasoningNode]:
        """Run MCTS search over reasoning trajectories.

        Args:
            prompt: The problem to reason about.
            budget: Number of MCTS iterations (default: num_samples from config).

        Returns:
            (best_trajectory_text, root_node)
        """
        budget = budget or self.config.num_samples
        assert budget > 0, "Search budget must be positive"

        # Initialize root
        root = ReasoningNode(state=prompt)

        for _ in range(budget):
            # 1. Select a leaf
            leaf = self.select(root)

            # 2. Expand the leaf
            children = self.expand(leaf)

            # 3. Evaluate each new child
            for child in children:
                value = self.evaluate(child)

                # 4. Backpropagate
                self.backpropagate(child, value)

        # Extract best trajectory: follow most-visited children
        trajectory_parts = [root.state]
        node = root
        while not node.is_leaf():
            node = max(node.children, key=lambda c: c.visits)
            # Extract just the last step from the state
            if " → " in node.state:
                last_step = node.state.split(" → ")[-1]
                trajectory_parts.append(last_step)

        best_trajectory = " → ".join(trajectory_parts)
        return best_trajectory, root

    # ------------------------------------------------------------------
    # Visualization helper
    # ------------------------------------------------------------------

    @staticmethod
    def tree_summary(root: ReasoningNode, max_depth: int = 3) -> str:
        """Return a text summary of the MCTS tree for display."""
        lines: list[str] = []

        def _walk(node: ReasoningNode, depth: int) -> None:
            if depth > max_depth:
                return
            indent = "  " * depth
            # Show just the last step for readability
            label = node.state.split(" → ")[-1] if " → " in node.state else node.state
            if len(label) > 50:
                label = label[:47] + "..."
            lines.append(
                f"{indent}├─ [{label}] "
                f"visits={node.visits}, value={node.value:.3f}"
            )
            for child in node.children:
                _walk(child, depth + 1)

        _walk(root, 0)
        return "\n".join(lines)


# ============================================================================
# Process Reward Model
# ============================================================================


class ProcessRewardModel(nn.Module):
    """Scores individual reasoning steps in [0, 1] via sigmoid.

    A Process Reward Model (PRM) evaluates the quality of each intermediate
    reasoning step, not just the final answer. This enables much more
    effective search — you can prune bad reasoning paths early.

    Architecture:
        context_tokens → Embedding → Linear → ReLU → Linear → score
        step_tokens    → Embedding → Linear → ReLU → Linear → score
        combined = context_repr + step_repr → Linear → sigmoid → [0, 1]

    The sigmoid output guarantees scores are always in [0, 1].

    **Process vs Outcome Reward Models:**
    - **Outcome RM (ORM):** Scores only the final answer. Binary signal.
      Cheap to train (just need answer correctness labels) but provides
      sparse feedback for search.
    - **Process RM (PRM):** Scores each intermediate step. Dense signal.
      More expensive to train (need step-level labels) but enables
      targeted backtracking and much more efficient search.

    OpenAI's "Let's Verify Step by Step" (2023) showed PRMs significantly
    outperform ORMs for math reasoning. DeepSeek-R1 (2025) uses RL with
    process-level rewards to train reasoning directly into the model.

    **Validates: Requirements 5.5, 5.6**
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        d_hidden: int = 128,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # Shared embedding (simple byte-level for educational purposes)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )

        # Step encoder
        self.step_encoder = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )

        # Scoring head: combined representation → scalar score
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def _encode_text(self, text: str) -> mx.array:
        """Convert text to token IDs (simple byte-level encoding)."""
        # Byte-level tokenization: each character → its ordinal mod vocab_size
        ids = [ord(c) % self.vocab_size for c in text[: self.max_len]]
        if len(ids) == 0:
            ids = [0]  # Ensure non-empty
        return mx.array(ids)

    def _embed_and_pool(self, token_ids: mx.array) -> mx.array:
        """Embed tokens and mean-pool to get a fixed-size representation."""
        embedded = self.embedding(token_ids)  # [seq_len, d_model]
        # Mean pooling over sequence dimension
        pooled = mx.mean(embedded, axis=0)  # [d_model]
        return pooled

    def score_step(self, context: str, step: str) -> mx.array:
        """Score a single reasoning step given its context.

        Args:
            context: The reasoning context so far (previous steps).
            step: The new reasoning step to evaluate.

        Returns:
            Score in [0, 1] as an mx.array scalar.
        """
        # Encode context and step
        ctx_ids = self._encode_text(context if context else " ")
        step_ids = self._encode_text(step if step else " ")

        # Embed and pool
        ctx_repr = self._embed_and_pool(ctx_ids)   # [d_model]
        step_repr = self._embed_and_pool(step_ids)  # [d_model]

        # Encode through respective encoders
        ctx_encoded = self.context_encoder(ctx_repr)    # [d_model]
        step_encoded = self.step_encoder(step_repr)     # [d_model]

        # Combine: element-wise addition
        combined = ctx_encoded + step_encoded  # [d_model]

        # Score via sigmoid → guaranteed [0, 1]
        logit = self.score_head(combined)  # [1]
        score = mx.sigmoid(logit)

        # Force evaluation to get concrete value
        mx.eval(score)

        return score.reshape(())  # scalar

    def score_trajectory(self, steps: list[str]) -> list[float]:
        """Score an entire reasoning trajectory step by step.

        Each step is scored with all previous steps as context.

        Args:
            steps: List of reasoning step strings.

        Returns:
            List of scores in [0, 1], one per step.
        """
        scores: list[float] = []
        context_parts: list[str] = []

        for step in steps:
            context = " ".join(context_parts) if context_parts else ""
            score = self.score_step(context, step)
            scores.append(score.item())
            context_parts.append(step)

        return scores

    def __call__(self, context: str, step: str) -> mx.array:
        """Forward pass: score a reasoning step."""
        return self.score_step(context, step)
