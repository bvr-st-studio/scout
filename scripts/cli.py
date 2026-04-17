import argparse
import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument(
    "--base", action="store_true", help="Run base model instead of Scout"
)
parser.add_argument("--reason", action="store_true", help="Enable reasoning mode")
args = parser.parse_args()
load_dotenv(args.config)

login(token=os.getenv("HF_TOKEN"))
MODEL = os.getenv("MODEL", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/scout-1-football-12b") + "/final"

SYSTEM_PROMPT = "/think" if args.reason else "/no_think"


def generate(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    return decoded.strip()


tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # type: ignore
    device_map="auto",
)

if args.base:
    model = base
    label = "BASE"
else:
    model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    label = "SCOUT"

mode = "REASON" if args.reason else "INSTRUCT"
print(f"\n🏈 Scout CLI [{label} | {mode}] — type 'exit' to quit\n")

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
