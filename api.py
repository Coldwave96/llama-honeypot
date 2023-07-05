import os
import torch
import argparse

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig
)

from fastapi import FastAPI, Request
import uvicorn, json, datetime

from utils.prompter import Prompter

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", default=None, type=str, required=True, help="Base model path")
parser.add_argument("--lora_weights", default="./lora-honeypot", type=str, help="LoRA weights patch")
parser.add_argument("--gpus", default="0", type=str, help="Use cuda:0 as default. Inference using multi-cards: --gpus=0,1,...")
parser.add_argument("--max_length", default=256, type=int, help='Maximum input prompt length, counted from the end of prompt')
parser.add_argument("--load_in_8bit", action="store_true", help="Quantify model in INT8")
parser.add_argument("--cpu_only", action="store_true", help="Inference on CPU only")
parser.add_argument("--prompt_template", default="honeypot", help="The prompt template to use, using honeypot as default")
args = parser.parse_args()

if args.cpu_only is True:
    args.gpus = ""

# Set CUDA devices if available
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


# Peft library can only import after setting CUDA devices
from peft import PeftModel

# Set up model and tokenizer
def setup():
    global prompter, model, tokenizer, device, max_length
    prompter = Prompter(args.prompt_template)
    max_length = args.max_length
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

def honeypot(
    history,
    command,
    temperature,
    top_p,
    top_k,
    num_beams,
    max_new_tokens,
):
    prompt = prompter.generate_prompt(command=command, history=history)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        num_beams = num_beams,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids = input_ids,
            generation_config = generation_config,
            return_dict_in_generate = True,
            output_score = True,
            max_new_tokens = max_new_tokens
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    response = prompter.get_response(output)
    history.append([command, response])
    return response, history

app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    jsno_post = json.dumps(json_post_raw)
    json_post_list = json.loads(jsno_post)
    command = json_post_list.get("command")
    history = json_post_list.get("history")
    temperature = json_post_list.get("temperature")
    top_p = json_post_list.get("top_p")
    top_k = json_post_list.get("top_k")
    num_beams = json_post_list.get("num_beams")
    max_new_tokens = json_post_list.get("max_new_tokens")
    response, history = honeypot(
        history = history,
        command = command,
        temperature = temperature if temperature else 0.4,
        top_p = top_p if top_p else 0.75,
        top_k = top_k if top_k else 40,
        num_beams = num_beams if num_beams else 4,
        max_new_tokens = max_new_tokens if max_new_tokens else 128
    )
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", command:"' + command + '", response:"' + repr(response) + '"'
    print(log)
    return answer


if __name__ == '__main__':
    setup()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
