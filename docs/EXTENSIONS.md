# Extending loCAII

The desktop composes four independent services: `Store`, `Engine`, `ModelManager`, and `MediaService`. Core modules do not import Qt. The UI starts background workers and receives structured events, so replacing inference or adding tools does not require rewriting the interface.

## Language models

Use Models > Add GGUF. Choose a GGUF file, optional matching multimodal projector, context length, and GPU layer count. The registry is stored in the private application directory, normally `.local/models.json`. It is not committed. A model family requires support in the installed llama.cpp runtime and a correct embedded chat template. Importing arbitrary weights does not guarantee that they support native tool calls or vision.

On the installed 8 GB GPU, use the tested 32K profiles and one prediction slot. The 27B profile uses partial offload and system RAM. `gpu_layers=0` and `device="none"` enable CPU-only operation. Model-manager methods are documented directly in `src/locaii/models.py`; another backend can implement the same methods.

## Tools

Install reviewed Python extensions in `.local/extensions/`. These execute as your user; only install code you trust. They are loaded on the next task turn. Tool names begin with `ext_`. A module exports `register_tools()` returning schema/implementation pairs:

```python
def register_tools():
    """Expose a small deterministic utility without filesystem side effects."""
    def count_words(arguments, context):
        """Count words in explicitly supplied text."""
        return {"words": len(arguments["text"].split())}

    return [{
        "schema": {
            "type": "function",
            "function": {
                "name": "ext_count_words",
                "description": "Count words in supplied text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        "execute": count_words,
    }]
```

An implementation receives `context['workspace']` and `context['cancel']`. It should return structured observed results, check cancellation, use bounded I/O, and avoid claiming success without evidence. External code must enforce its own access boundaries; the core cannot sandbox arbitrary Python.

## Media

Import a ComfyUI API-format workflow from Create. The graph is inspected for text, sampler, latent/image/video parameters; unsupported controls are hidden rather than pretending to alter output. See [MEDIA.md](MEDIA.md) for mappings, engine paths and supported built-in workflows.

## Events

The engine emits `status`, `delta`, `tool`, `memory`, `plan`, `review`, `usage`, `compaction`, `done`, and `error`. The media adapter also emits measured `progress` and `completed`. Database events retain meaningful receipts while streaming text is persisted as completed messages. UI extensions should update widgets only on the Qt main thread.
