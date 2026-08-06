"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # TODO: call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True 
    )

    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # TODO: sum p.numel() over every parameter tensor in the module
    param_num = 0
    for iterator_tensor in model.parameters():
        param_num += iterator_tensor.numel()
    return param_num

# Step 3 - is_model_4bit_quantized
import bitsandbytes as bb
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    for module in model.modules():
        if isinstance(module, bb.nn.Linear4bit):
            return True
    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer is None or tokenizer.pad_token is not None:
        return tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
    return target_modules

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    if target_modules is None:
        target_modules = get_lora_target_modules()
        
    peft_model = FastLanguageModel.get_peft_model(
        model=model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,

    )
    return peft_model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    trainable_num = 0
    for param in model.parameters():
        if param.requires_grad == True:
            trainable_num += param.numel()
    
    return trainable_num

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # TODO: return the fraction of parameters that are trainable.
    return trainable_count/total_count

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # TODO: return a tiny hand-written list of instruction/response example dicts.
    return [
        {
            "instruction": "What is machine learning?",
            "response": (
                "Machine learning is a method that allows computers "
                "to learn patterns from data."
            ),
        },
        {
            "instruction": "Translate 'good morning' into Vietnamese.",
            "response": "Chào buổi sáng.",
        },
        {
            "instruction": "Calculate 8 multiplied by 7.",
            "response": "56",
        },
        {
            "instruction": "Name one benefit of regular exercise.",
            "response": "Regular exercise can improve physical health.",
        },
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # TODO: combine example['instruction'] and example['response'] into one string
    instruction = example.get("instruction", "")
    response = example.get("response", "")
    
    return f'### Instruction:\n{instruction}\n\n### Response:\n{response}'

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # TODO: apply format_instruction_example to every example and return the list
    texts = []
    for example in examples:
        formatted = format_instruction_example(example)
        texts.append(formatted)

    return texts

# Step 12 - build_text_dataset
from datasets import Dataset 

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # TODO: return a datasets.Dataset with one 'text' column holding the given strings
    data = Dataset.from_dict({"text": texts})

    return data

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # TODO: call the tokenizer on text and return its input_ids as a plain list
    ans = tokenizer.encode(text)

    return ans

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # TODO: return the length of the input_ids sequence
    return len(input_ids)

# Step 15 - build_training_arguments
from transformers import TrainingArguments
import torch

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    # TODO: build TrainingArguments with batch size 1, given max_steps, given lr, bf16 or fp16.
    if torch.cuda.is_bf16_supported():
        is_bf16 = True
    else:
        is_bf16 = False

    args = TrainingArguments(
        output_dir = output_dir,
        max_steps = max_steps,
        learning_rate = learning_rate,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        bf16 = is_bf16,
        fp16 = not is_bf16,
        logging_steps=1,
        optim="adamw_8bit",
    )

    return args

# Step 16 - build_sft_trainer
from trl import SFTTrainer 

def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    # TODO: wire model, tokenizer, dataset, and training_args into an SFTTrainer
    sfttrainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = training_args,
        max_seq_length = max_seq_length,
        packing = False,
        dataset_text_field = "text",
    )
    return sfttrainer

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

