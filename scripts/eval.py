import argparse
import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to config env file")
args = parser.parse_args()
load_dotenv(args.config)

login(token=os.getenv("HF_TOKEN"))
MODEL = os.getenv("MODEL", "nvidia/Minitron-4B-Base")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/scout-1-football-instruct-4b") + "/final"

PROMPTS = [
    "What adjustments would you make at halftime against a team running heavy zone defense?",
    "Explain the difference between a 4-3 and 3-4 defensive front.",
    "A running back is averaging 2.1 YPC in the first half. What does that tell you?",
    "What is a mesh concept in passing offense?",
]


def generate(model, tokenizer, prompt):
    inputs = tokenizer(
        f"### Instruction: {prompt}\n### Response:", return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("=== BASE MODEL ===")
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map="auto"
)
for p in PROMPTS:
    print(f"\nQ: {p}")
    print(f"A: {generate(base, tokenizer, p)}")

print("\n=== SCOUT-1-FOOTBALL-INSTRUCT-4B ===")
scout = PeftModel.from_pretrained(base, OUTPUT_DIR)
for p in PROMPTS:
    print(f"\nQ: {p}")
    print(f"A: {generate(scout, tokenizer, p)}")
