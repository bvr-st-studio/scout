import argparse
import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to config env file")
parser.add_argument(
    "--base", action="store_true", help="Run base model instead of Scout"
)
args = parser.parse_args()
load_dotenv(args.config)

login(token=os.getenv("HF_TOKEN"))
MODEL = os.getenv("MODEL", "nvidia/Minitron-4B-Base")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/scout-1-football-instruct-4b") + "/final"


def generate(model, tokenizer, prompt):
    inputs = tokenizer(
        f"### Instruction: {prompt}\n### Response:", return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Strip the instruction prefix, return just the response
    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    return decoded.strip()


tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map="auto"
)

if args.base:
    model = base
    label = "BASE"
else:
    model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    label = "SCOUT"

print(f"\n🏈 Scout CLI [{label}] — type 'exit' to quit\n")

while True:
    try:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        print(f"\nA: {generate(model, tokenizer, question)}\n")
    except KeyboardInterrupt:
        break

print("\nDone.")
