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