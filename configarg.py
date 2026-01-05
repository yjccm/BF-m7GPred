from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import TrainingArguments

@dataclass
class ModelArgs:
    bert_name_or_path: str = field(default="dnabert2/DNABERT-2-117M")
    lstm1_hidden: int = field(default=384, metadata={"help": "hidden size for lstm1 "})
    lstm2_hidden: int = field(default=128, metadata={"help": "hidden size for lstm2"})
    lstm_dropout: float = field(default=0.5)
    dropout: float = field(default=0.2)

    bert_pick_idx: int = field(default=49, metadata={"help": "token index for bert branch"})
    base_pick_idx: int = field(default=250, metadata={"help": "base index for phys branch"})

    use_lora: bool = field(default=True, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.2, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="Wqkv", metadata={"help": "where to perform LoRA"})



@dataclass
class TrainingArgs(transformers.TrainingArguments):
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Keep extra columns for custom models."},
    )
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=120, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=16)
    num_train_epochs: int = field(default=30)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=120)
    weight_decay: float = field(default=0.05)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    metric_for_best_model: str = field(default="matthews_correlation")
    greater_is_better: bool = field(default=True)
