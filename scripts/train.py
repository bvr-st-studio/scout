import argparse
import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to config env file")
args = parser.parse_args()
load_dotenv(args.config)

login(token=os.getenv("HF_TOKEN"))

MODEL = os.getenv("MODEL", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
DATA_FILE = os.getenv("DATA_FILE", "data/football/instruct/scout-1-football.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/scout-1-football-12b")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # type: ignore
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files=DATA_FILE, split="train")


def format_example(example):
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


dataset = dataset.map(format_example)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        dataset_text_field="text",
        gradient_checkpointing=True,
    ),
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
print("Done. Scout saved.")
