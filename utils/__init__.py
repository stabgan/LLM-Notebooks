"""Shared utilities for the LLM Learning Notebook series."""

from utils.checks import validate_environment
from utils.viz import (
    plot_attention_heatmap,
    plot_loss_curve,
    plot_token_probabilities,
    plot_embeddings_2d,
)
from utils.benchmark import time_function, memory_snapshot, estimate_model_memory
from utils.data import TextDataset
from utils.moe import (
    MoEConfig,
    MoERouter,
    ExpertChoiceRouter,
    HashRouter,
    ExpertFFN,
    MoEBlock,
    top_k,
)
from utils.ssm import SSMConfig, SimpleSSM, SelectiveSSM, MambaBlock
from utils.alignment import RewardModelConfig, RewardModel, DPOConfig, DPOTrainer, SimpleLM, GRPOConfig, GRPOTrainer
from utils.scaling import ScalingLawParams, ComputeBudget, ScalingLawPredictor
from utils.reasoning import (
    ReasoningConfig,
    CoTPromptPipeline,
    ReasoningNode,
    MCTSReasoner,
    ProcessRewardModel,
)
from utils.transformer_analysis import (
    TransformerConfig,
    ActivationComparison,
    NormalizationComparison,
    gradient_flow_analysis,
    ParameterCounter,
    DTYPE_BYTES,
    xavier_init,
    kaiming_init,
    plot_init_distributions,
    backprop_walkthrough,
)
from utils.attention_optimization import (
    online_softmax,
    online_softmax_blocked,
    standard_attention,
    tiled_attention,
    flash_memory_analysis,
    PagedAttentionBlockManager,
    simulate_ring,
    benchmark_attention,
    benchmark_standard_vs_flash,
    plot_benchmark,
)
