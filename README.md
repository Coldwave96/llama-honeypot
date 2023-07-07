# llama-honeypot
A honeypot backend API based on fine-tuning LLaMA via LoRA.

This repo contains code for fine-tuning LLaMA-7B vai LoRA, in order to fit the requirements for honeypot backend API. We gathered thousands of response paired with commands, try to adjust the model into a Linux console. In addition, we designed a prompt template which supports history, thus the model could better support multi-round interaction.

## Setup
1.Install dependencies.
```Bash
pip install -r requirements.txt
```

2.Prepare LLaMA model with HuggingFace type. Either from HuggingFace repo or converted locally from original LLaMA model through scripts provided by `transformers`. 

## Fine-tune Data Example
```Json
[
    {
        "command": "pwd",
        "history": [],
        "output": "/home/user"
    },
    {
        "command": "pwd",
        "history": [["cd Document", ""]],
        "output": "/home/user/Document"
    }
]
```

## Training(`finetune.py`)
* **Easy Example**
```Bash
python finetune.py \
    --base_model /path/to/base/model \
    --data_path /path/to/data/file \
    --output_dir ./lora-honeypot
```

* **Hyper-parameters Example**
```Bash
python finetune.py \
    --base_model /path/to/base/model \
    --data_path /path/to/data/file \
    --output_dir ./lora-honeypot \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```

## Inference
## Launch API
```Bash
python api.py \
	--base_model /path/to/base/model \
	--lora_weights ./lora-honeypot
```
## API Usage
* **Easy Example**
```Bash
curl -X POST "http://127.0.0.1:8000" \
     -H 'Content-Type: application/json' \
     -d '{"command": "pwd", "history": []}'
```

* **Hyper-parameters Example**
```Bash
curl -X POST "http://127.0.0.1:8000" \
     -H 'Content-Type: application/json' \
     -d '{"command": "pwd", "history": [], "temperature": 0.4, "top_p": 0.75, "top_k": 40, "num_beams": 4, "max_new_tokens": 128}'
```
