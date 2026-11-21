"""Small presets sized for Kaggle's free GPUs (1-2x NVIDIA Tesla T4, 15 GiB each).

The architecture is the same DeepSeek V3 design (MLA attention, MoE with shared experts and
auxiliary-loss-free load balancing); only the sizes shrink, to about 60M parameters, most of which
is the 102,400-token vocabulary embedding and output projection.

T4s (Turing, compute capability 7.5) have no native bfloat16, so these presets train fully in
float32 and disable torch.compile. The MoE layer automatically uses its per-expert fallback
instead of `torch._grouped_mm` on these GPUs.

Kept in a separate file from `deepseek_v3_presets.py` so `submit_slurm_jobs.sh <nodes> all`
does not schedule them on the cluster.
"""

from torchfeather.deepseek_v3.deepseek_v3_model_args import DeepSeekV3ModelArgs
from torchfeather.deepseek_v3.mixture_of_experts.moe_layer import MoEArgs
from torchfeather.job_configs.job_config_dataclasses import JobConfig


def get_kaggle_t4_model_args() -> DeepSeekV3ModelArgs:
    return DeepSeekV3ModelArgs(
        vocab_size=102400,  # must cover the deepseek-moe-16b-base tokenizer
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=4,
        n_dense_layers=1,
        n_heads=4,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=1,
            top_k=2,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
        ),
        q_lora_rank=0,
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        mscale=0.70,
    )


def get_kaggle_t4_base_config() -> JobConfig:
    config = JobConfig()

    config.job.dump_folder = "./outputs"

    config.profiling.enable_profiling = False
    config.profiling.enable_memory_snapshot = False

    config.metrics.log_freq = 10
    config.metrics.save_folder = "metrics"
    config.metrics.enable_wandb = False

    config.model.hf_assets_path = "./assets/hf/deepseek-moe-16b-base"
    config.model.args = get_kaggle_t4_model_args()

    config.optimizer.name = "AdamW"
    config.optimizer.lr = 1e-3
    config.optimizer.eps = 1e-8

    config.lr_scheduler.warmup_steps = 30
    config.lr_scheduler.decay_ratio = 0.8
    config.lr_scheduler.decay_type = "cosine"
    config.lr_scheduler.min_lr_factor = 0.1

    config.training.local_batch_size = 8
    config.training.seq_len = 512
    config.training.max_norm = 1.0
    config.training.steps = 300
    config.training.dataset = "fineweb"
    config.training.dtype = "float32"
    config.training.mixed_precision_param = "float32"
    config.training.seed = 42

    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1

    config.checkpoint.enable = True
    config.checkpoint.folder = "checkpoint"
    config.checkpoint.interval = 150
    config.checkpoint.async_mode = "disabled"
    config.checkpoint.keep_latest_k = 2

    config.activation_checkpoint.mode = "none"

    config.compile.enable = False

    config.validation.enable = True
    config.validation.freq = 100
    config.validation.steps = 4
    config.validation.local_batch_size = 8

    config.generation.enable = True

    return config


def get_kaggle_t4_single_config() -> JobConfig:
    """One T4: plain single-device training."""
    return get_kaggle_t4_base_config()


def get_kaggle_t4_ddp_config() -> JobConfig:
    """Two T4s: replicated data parallelism (DDP)."""
    config = get_kaggle_t4_base_config()
    config.parallelism.data_parallel_replicate_degree = 2
    return config


def get_kaggle_t4_fsdp_config() -> JobConfig:
    """Two T4s: fully sharded data parallelism (FSDP2)."""
    config = get_kaggle_t4_base_config()
    config.parallelism.data_parallel_shard_degree = 2
    return config


def get_kaggle_t4_tp_config() -> JobConfig:
    """Two T4s: 2-way tensor parallelism (attention, dense FFN and experts sharded, loss parallel)."""
    config = get_kaggle_t4_base_config()
    config.parallelism.tensor_parallel_degree = 2
    return config


def get_kaggle_t4_ddp_2k_config() -> JobConfig:
    """Two T4s, DDP, 2,000 steps (about 16M tokens): the longer run with held-out validation every 250 steps."""
    config = get_kaggle_t4_ddp_config()
    config.training.steps = 2000
    config.lr_scheduler.warmup_steps = 100
    config.metrics.log_freq = 20
    config.checkpoint.interval = 1000
    config.validation.freq = 250
    config.validation.steps = 8
    return config


kaggle_t4_config_map = {
    "kaggle_t4_single": get_kaggle_t4_single_config,
    "kaggle_t4_ddp": get_kaggle_t4_ddp_config,
    "kaggle_t4_ddp_2k": get_kaggle_t4_ddp_2k_config,
    "kaggle_t4_fsdp": get_kaggle_t4_fsdp_config,
    "kaggle_t4_tp": get_kaggle_t4_tp_config,
}
