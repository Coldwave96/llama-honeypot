import os
import torch
import argparse

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)

from fastapi import FastAPI, Request
import uvicorn, json, datetime

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", default=None, type=str, required=True, help="Base model path")
parser.add_argument("--lora_weights", default="./lora-alpaca", type=str, help="LoRA weights patch")
parser.add_argument("--gpus", default="0", type=str, help="Use cuda:0 as default. Inference using multi-cards: --gpus=0,1,...")
parser.add_argument("--port", default=8000, type=int, help="Port of API service")
parser.add_argument("--max_length", default=256, type=int, help='Maximum input prompt length, counted from the end of prompt')
parser.add_argument("--load_in_8bit", action="store_true", help="Quantify model in INT8")
parser.add_argument("--cpu_only", action="store_true", help="Inference on CPU only")
args = parser.parse_args()

if args.cpu_only is True:
    args.gpus = ""

# Set CUDA devices if available
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


# Peft library can only import after setting CUDA devices
from peft import PeftModel

# Set up model and tokenizer
def setup():
    global model, tokenizer, device, port, max_length
    max_length = args.max_length
    port = args.port
    load_in_8bit = args.load_in_8bit
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit = load_in_8bit,
        torch_dtype = load_type,
        low_cpu_mem_usage = True,
        device_map = "auto"
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_weights,
        torch_dtype = load_type,
        device_map = "auto"
    )

    if device == torch.device("cpu"):
        model.float()
    
    model.eval()
