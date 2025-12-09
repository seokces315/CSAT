from .pooling import CLSPooling, LastTokenPooling, MeanPooling, AttentionPooling
from .head import LinearHead, MLPHead
from .loss import HuberLoss, FocalLoss

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model

import torch.nn as nn


# Custom Embedding moel for fine-tuning on CSAT task
class EmbTaskModel(nn.Module):
    # Initializer
    def __init__(self, task_type, model):
        super(EmbTaskModel, self).__init__()

        self.task_type = task_type
        self.backbone = model
        hidden_size = self.backbone.config.hidden_size

        # TODO : 2-stage learning (Freeze or Not)

        pooling_map = {}


# Custom LLM model for fine-tuning on CSAT task
class LLMTaskModel(nn.Module):
    # Initializer
    def __init__(
        self,
        task_type,
        model,
        pooling_type,
        head_type,
        r,
        dropout,
        criterion_type,
        delta,
        gamma,
    ):
        super(LLMTaskModel, self).__init__()

        self.task_type = task_type
        self.backbone = model
        hidden_size = self.backbone.config.hidden_size

        # TODO : 2-stage learning (Freeze or Not)

        # Select the pooler based on the specified type
        pooling_map = {
            "CLS": CLSPooling(),
            "LAST": LastTokenPooling(),
            "MEAN": MeanPooling(),
            "ATTN": AttentionPooling(hidden_dim=hidden_size),
        }
        self.pooler = pooling_map[pooling_type]

        # Select head type based on the specified type
        num_targets = 1 if self.task_type == "REG" else 3
        head_map = {
            "LINEAR": LinearHead(hidden_dim=hidden_size, out_dim=num_targets),
            "MLP": MLPHead(
                hidden_dim=hidden_size, out_dim=num_targets, dropout=dropout, r=r
            ),
        }
        self.head = head_map[head_type]

        # Select criterion function based on the specified type
        criterion_map = {
            "MSE": nn.MSELoss(),
            "HUBER": HuberLoss(delta=delta),
            "CE": nn.CrossEntropyLoss(),
            "FOCAL": FocalLoss(gamma=gamma),
        }
        self.criterion = criterion_map[criterion_type]

    # Forward pass
    def forward(self, **input_dicts):
        # Backbone inferencing
        input_ids = input_dicts["input_ids"]
        attention_mask = input_dicts["attention_mask"]
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Apply pooling over the last hidden state
        last_hidden_state = outputs.hidden_states[-1]
        pooled = self.pooler(last_hidden_state, attention_mask)

        # Pass pooled features through the task head
        logits = (
            self.head(pooled).reshape(-1)
            if self.task_type == "REG"
            else self.head(pooled)
        )

        # Convert labels to correct type and shape
        labels = input_dicts["labels"]
        labels = (
            labels.to(logits.dtype).reshape(-1)
            if self.task_type == "REG"
            else labels.long().reshape(-1)
        )

        # Compute the task-specific loss
        loss = self.criterion(logits, labels)

        return {"logits": logits, "loss": loss}


# Function for quantization
def get_quantization_config(dtype):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


# Function for LoRA configuration
def get_lora_config(r, lora_alpha, lora_dropout):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )


# Load a 4-bit LLM with LoRA and a task head
def load_llm(
    dtype,
    task_type,
    model_id,
    token,
    lora_r,
    lora_alpha,
    lora_dropout,
    pooling_type,
    head_type,
    r,
    dropout,
    criterion_type,
    delta,
    gamma,
):
    # Initialize tokenizer and set configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare 4-bit quantization configuration for the model
    quantization_config = get_quantization_config(dtype=dtype)

    # Load the backbone model with 4-bit quantization
    backbone_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": 0},
        quantization_config=quantization_config,
        token=token,
        trust_remote_code=True,
    )

    # Optimize the model for memory-efficient training
    backbone_model.config.use_cache = False
    backbone_model.gradient_checkpointing_enable()

    # Wrap the backbone model with LoRA for efficient fine-tuning
    lora_config = get_lora_config(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )
    peft_model = get_peft_model(backbone_model, lora_config)

    # Build the task-specific model on top of the LoRA-adapted LLM
    llm_task_model = LLMTaskModel(
        task_type=task_type,
        model=peft_model,
        pooling_type=pooling_type,
        head_type=head_type,
        r=r,
        dropout=dropout,
        criterion_type=criterion_type,
        delta=delta,
        gamma=gamma,
    )

    return tokenizer, llm_task_model
