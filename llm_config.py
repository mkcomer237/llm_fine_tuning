from dataclasses import dataclass, field
from typing import Optional, List
from unsloth import is_bfloat16_supported

# List of supported 4-bit models
SUPPORTED_4BIT_MODELS = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
    "unsloth/Llama-3.2-1B-bnb-4bit",           # Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"  # Llama 3.3 70B
]

@dataclass
class ModelConfig:
    # Model parameters
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    hf_token: Optional[str] = None

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    target_modules: List[str] = None
    
    # Training parameters
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 5
    num_train_epochs: int = 1
    max_steps: int = 100
    learning_rate: float = 2e-4
    output_dir: str = "outputs"

    # Dataset parameters
    test_size: float = 0.02
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        
        # Validate model name if using 4-bit quantization
        if self.load_in_4bit and self.model_name not in SUPPORTED_4BIT_MODELS:
            raise ValueError(
                f"Model {self.model_name} is not in the list of supported 4-bit models. "
                f"Please choose from: {', '.join(SUPPORTED_4BIT_MODELS)}"
            )

        # Validate if bfloat16 is supported
        if not is_bfloat16_supported():
            raise ValueError("Bfloat16 is not supported on this system. Please use a system with bfloat16 support.")
