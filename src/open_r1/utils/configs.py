import argparse
import inspect
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

class QRLParser:
    def __init__(self, config_classes: Tuple):

        self.config_classes = config_classes

    def parse_args_and_config(self):
        parser = argparse.ArgumentParser()

        # 遍历所有配置类，动态添加命令行参数
        for config_class in self.config_classes:
            self._add_class_arguments(parser, config_class)

        # 解析命令行参数
        args = vars(parser.parse_args())

        # 将解析结果分配回配置类实例
        result = [self._populate_config_with_args(config_class, args) for config_class in self.config_classes]

        return tuple(result)

    def _add_class_arguments(self, parser: argparse.ArgumentParser, config_class):
        config_instance = config_class()
        for attr, default_value in vars(config_instance).items():
            if not attr.startswith("__"):  # 忽略私有属性
                arg_name = f"--{attr.replace('_', '-')}"  # 将下划线替换为命令行格式
                arg_type = self._infer_type(default_value)
                parser.add_argument(
                    arg_name,
                    type=arg_type,
                    default=default_value,
                    help=f"{attr} (default: {default_value})"
                )

    def _infer_type(self, default_value):
        if isinstance(default_value, bool):
            # 布尔值允许命令行使用 "true"/"false"
            return lambda x: x.lower() in ("true", "1", "yes")
        elif default_value is None:
            # 如果默认值为 None，解析为字符串类型
            return str
        return type(default_value)

    def _populate_config_with_args(self, config_class, args: dict):
        config_instance = config_class()
        for attr in vars(config_instance):
            if attr in args:
                setattr(config_instance, attr, args[attr])
        return config_instance


# class DataConfig:
#     def __init__(self):
#         self.dataset = "dapomath" # Dataset to use, options: "gsm8k", "dapomath", "codeforces"


# class TrainingConfig:
#     def __init__(self):
#         self.use_vllm = True  # Whether to enable vLLM for fast inference
#         self.learning_rate = 5e-6  # Learning rate
#         self.adam_beta1 = 0.9  # Beta1 parameter for the Adam optimizer
#         self.adam_beta2 = 0.99  # Beta2 parameter for the Adam optimizer
#         self.weight_decay = 0.1  # Weight decay for regularization
#         self.warmup_ratio = 0.1  # Warm-up ratio for the learning rate scheduler
#         self.lr_scheduler_type = "cosine"  # Type of learning rate scheduler
#         self.logging_steps = 1  # Number of steps between log outputs
#         self.per_device_train_batch_size = 1  # Training batch size per device **** when less than num_generations, it will be 1 ****
#         self.gradient_accumulation_steps = 4  # Number of steps to accumulate gradients
#         self.num_generations = 8  # Number of generations during training
#         self.max_prompt_length = 1024  # Maximum length of the prompt
#         self.max_completion_length = 8192  # Maximum length of the completion
#         self.num_train_epochs = 1  # Number of training epochs
#         self.save_steps = 35  # Number of steps between saving model checkpoints
#         self.save_total_limit = 3  # Maximum number of saved checkpoints
#         self.max_grad_norm = 0.1  # Maximum norm for gradient clipping
#         self.loss_type = "bnpo"  # Loss function type
#         self.report_to = "wandb"  # Tool to report logs (e.g., Weights & Biases)
#         self.output_dir = "model/Qwen3-3B_dapo_lora_dapomath_b32"  # Directory to save the model and checkpoints
#         self.optim = "adamw"  # Optimizer type


# class ModelConfig:
#     def __init__(self):
#         self.model_name = "/lustre/fsw/portfolios/nvr/users/weihua/model/Qwen3-8B"
#         self.max_seq_length = 10000
#         self.lora_rank = 32
#         self.lora_alpha = 32
#         self.load_in_4bit = False
#         self.load_in_8bit = False
#         self.fast_inference = True
#         self.gpu_memory_utilization = 0.5
#         self.use_gradient_checkpointing = "unsloth"
#         self.random_state = 3407
#         self.target_modules = [
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj",
#         ], # Remove QKVO if out of memory

@dataclass
class DataConfig:
    dataset: str = field(
        default="dapomath",
        metadata={"help": "Dataset to use, options: 'gsm8k', 'dapomath', 'codeforces'"},
    )


@dataclass
class TrainingConfig:
    use_vllm: bool = field(
        default=True,
        metadata={"help": "Whether to enable vLLM for fast inference"},
    )
    learning_rate: float = field(
        default=5e-6,
        metadata={"help": "Learning rate for training"},
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 parameter for the Adam optimizer"},
    )
    adam_beta2: float = field(
        default=0.99,
        metadata={"help": "Beta2 parameter for the Adam optimizer"},
    )
    weight_decay: float = field(
        default=0.1,
        metadata={"help": "Weight decay for regularization"},
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warm-up ratio for the learning rate scheduler"},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Type of learning rate scheduler (e.g., 'cosine')"},
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Number of steps between log outputs"},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Training batch size per device"},
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of steps to accumulate gradients"},
    )
    num_generations: int = field(
        default=8,
        metadata={"help": "Number of generations during training"},
    )
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Maximum length of the input prompt"},
    )
    max_completion_length: int = field(
        default=8192,
        metadata={"help": "Maximum length of the generated completion"},
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"},
    )
    save_steps: int = field(
        default=35,
        metadata={"help": "Number of steps between saving model checkpoints"},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Strategy for saving checkpoints (e.g., 'steps')"},
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of saved checkpoints"},
    )
    max_grad_norm: float = field(
        default=0.1,
        metadata={"help": "Maximum norm for gradient clipping"},
    )
    loss_type: str = field(
        default="bnpo",
        metadata={"help": "Loss function type (e.g., 'bnpo')"},
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "Beta parameter for the loss function"},
    )   
    report_to: str = field(
        default="wandb",
        metadata={"help": "Tool to report logs (e.g., 'wandb')"},
    )
    output_dir: str = field(
        default="model/Qwen3-3B_dapo_lora_dapomath_b32",
        metadata={"help": "Directory to save the model and checkpoints"},
    )
    optim: str = field(
        default="adamw",
        metadata={"help": "Optimizer type (e.g., 'adamw')"},
    )
    mask_truncated_completions: bool = field(
        default=True,
        metadata={"help": "Whether to mask truncated completions"},
    )
    ddp_find_unused_parameters: bool = field(
        default=True,
        metadata={"help": "Whether to find unused parameters in DDP"},
    )
    epsilon_high: float = field(
        default=0.28,
        metadata={"help": "High epsilon value for exploration"},
    )
    num_iterations: int = field(
        default=4,
        metadata={"help": "Number of iterations for training"},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of steps to accumulate gradients"},
    )
    # max_steps: int = field(
    #     default=500,
    #     metadata={"help": "Maximum number of training steps"},
    # )

@dataclass
class ModelConfig:
    model_name: str = field(
        default="/lustre/fsw/portfolios/nvr/users/weihua/model/Qwen3-8B",
        metadata={"help": "Path to the pretrained model"},
    )
    max_seq_length: int = field(
        default=10000,
        metadata={"help": "Maximum sequence length supported by the model"},
    )
    lora_rank: int = field(
        default=32,
        metadata={"help": "Rank for LoRA (low-rank adaptation)"},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Alpha for LoRA (low-rank adaptation)"},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 4-bit precision"},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8-bit precision"},
    )
    full_finetuning: bool = field(
        default=False,
        metadata={"help": "Whether to use full fine-tuning"},
    )
    fast_inference: bool = field(
        default=True,
        metadata={"help": "Enable faster inference"},
    )
    gpu_memory_utilization: float = field(
        default=0.5,
        metadata={"help": "Fraction of GPU memory to utilize"},
    )
    use_gradient_checkpointing: str = field(
        default="unsloth",
        metadata={"help": "Gradient checkpointing strategy"},
    )
    random_state: int = field(
        default=3407,
        metadata={"help": "Random seed for reproducibility"},
    )
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        metadata={"help": "Modules to apply LoRA on. Remove QKVO if out of memory."},
    )
    modules_to_save: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Modules to save during training."},
    )
    ln: bool = field(
        default=False,
        metadata={"help": "Whether to use LayerNorm (LN)"},
    )