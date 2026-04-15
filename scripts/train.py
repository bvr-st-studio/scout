import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

MODEL = os.getenv("MODEL", "nvidia/Minitron-4B-Base")
DATA_FILE = os.getenv("DATA_FILE", "data/processed/scout_v0.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/scout-v0")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float32,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files=DATA_FILE, split="train")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
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
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
	bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        dataset_text_field="text",
    ),
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
print("Done. Scout saved.")
