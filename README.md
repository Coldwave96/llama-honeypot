# llama-honeypot
A honeypot backend API based on fine-tuning LLaMA via LoRA.

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

## Usage
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
