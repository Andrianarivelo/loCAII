# Independent model runtime

loCAII launches its own official `llama.cpp` server. LM Studio is not used for inference, model discovery, token counting, or tool execution. Existing GGUF files are referenced in place and never copied into Git.

## Install and provenance

Run from the project directory using the loCAII Python environment:

```bash
python scripts/install_runtime.py
```

The installer downloads the pinned official Linux x86-64 Vulkan release [llama.cpp b10893](https://github.com/ggml-org/llama.cpp/releases/tag/b10893), verifies its SHA-256, safely extracts the archive, and writes ignored `runtime/runtime.json` with the exact binary path and source. The package is approximately 30 MB. Its published SHA-256 is:

```text
af00feecc9f417d89dc0649f9820d77d4a08b3257f206bca64766058d6046d56
```

The official release currently provides Linux Vulkan binaries rather than Linux CUDA binaries. Vulkan detects this computer's NVIDIA RTX 4070 Laptop GPU. loCAII explicitly selects NVIDIA because the Intel integrated GPU is also exposed as a Vulkan device. This is GPU inference without an LM Studio runtime dependency.

For a compatible independently installed CUDA build, set `LOCAII_LLAMA_SERVER` to its absolute executable path before launching. The backend discovers the NVIDIA device from `--list-devices`. A custom runtime must support the server flags and endpoints used in `src/locaii/models.py`; it is not automatically considered tested.

To update the pinned release, update `RELEASE`, `ASSET`, and `SHA256` together in the installer using the corresponding official release metadata, run the installer, then rerun the regression suite and synthetic live model checks. Close loCAII before replacing its runtime. Updating does not replace weights or private task history.

## Offline use and CPU fallback

After the runtime, Python dependencies, and model weights are installed, local chat, tools, and existing media workflows can operate offline. Optional web tools and cloud supervisors require network access. Re-running the installer needs internet access; ordinary launches do not.

Profiles live in ignored `.local/models.json`, or the selected `LOCAII_HOME` directory. Profiles can be registered and edited from the Models page. Programmatic edits use `ModelManager.update(model_id, **fields)`.

The installed 9B and 4B profiles default to 32,768 context tokens, one slot, quantized `q8_0` KV cache, and six or fewer CPU threads. Vision uses the matching projector on CPU, with a 1,024-token image ceiling. Images receive a conservative 2,048-token context reservation in addition to exact rendered text tokens. These are model-specific settings, not a guarantee that every future model fits an 8 GB GPU. The 27B profile uses partial GPU offload and is expected to be slower.

For CPU-only operation, set these fields in a profile:

```json
{
  "gpu_layers": 0,
  "device": "none",
  "context": 8192,
  "threads": 2,
  "vision": false
}
```

Context, GPU layers, thread count, vision, thinking, projector, cache type, and device can be updated without changing the GGUF. Changed settings apply on the next load. A model process owned by another app is never killed to free memory. If another GPU workload is running, wait for it, use CPU mode, or explicitly release it in its own application.

The loopback service defaults to port 8787; `LOCAII_MODEL_PORT` overrides it. An occupied port produces a visible error. Each model process gets a random authentication key held only in memory and its process environment. Browser origins are restricted. The application does not expose native tools through llama.cpp itself; its Python tool layer executes them under the selected application settings.

The inference idle timeout defaults to ten minutes, so a long CPU prompt can finish its initial processing before the first token arrives. The profile field `idle_timeout` can be set between 30 and 7,200 seconds. This is a silence timeout, not a limit on total task duration.

A Stop request interrupts the owned server if needed, including blocked prompt processing. A later request reloads it. Closing or crashing the app releases its owned Linux server through process-parent handling. Separate durable terminal jobs have their own controls and receipts.

## Verified CPU smoke test on this machine

The bounded test on 2026-09-10 used the actual installed Huihui Qwen3.5 4B abliterated Q4_K_M weights, llama.cpp b10893, CPU device `none`, zero GPU layers, 8,192 context, and two threads. It did not interrupt the user's active LM Studio task.

| Check | Observed result | Seconds |
|---|---|---:|
| Load independent CPU server | Ready on private test port 18787 | 3.43 |
| Visible answer | Exact `LOCAII_READY` | 3.54 |
| Native tool request | `read_text_file` with `{"path":"note.txt"}` | 25.33 |
| Tool receipt follow-up | Exact `SYNTHETIC_RECEIPT_742` | 7.57 |
| Unauthenticated request | HTTP 401 | Included in total |
| Complete smoke run | All four checks passed | 40.11 |

The text token counter reported 37 tokens for the first prompt, exactly matching the server's measured `prompt_tokens`. The tool follow-up supplied a deliberately synthetic receipt to test protocol continuation; it is not a claim that a real file tool was executed in this smoke run. Native file writes and receipts are independently covered by engine and tool regression tests.

NVIDIA compute-process snapshots before, during, and after showed only the existing LM Studio process at 6,582 MiB. The CPU test added no GPU allocation. Machine-local full results and raw backend logs are in `.local/verification/cpu-model/`, excluded from Git.

This CPU test does not establish GPU performance, vision correctness, or 27B usability. The separate later GPU and vision measurements are documented below. The 27B model remains unverified. Performance varies with concurrent work, model size, context, and the selected backend.

## Full CPU agent workflow baseline

A separate real engine test used the full normal tool schema, a synthetic `input.txt`, and instructions to write and reread a separate result. The first run failed after 171.93 seconds: the 4B model repeatedly mistyped the long absolute workspace path, then attempted a write without having read the verification code. The filesystem boundary rejected all four requests and the engine stopped visibly after its bounded failure threshold. The input remained unchanged and no result was written. This is a model/tool-use failure, not a successful end-to-end test. The reproducible harness is `scripts/check_cpu_workflow.py`; the run has a five-minute deadline and checks actual file hashes, tool receipts, and completion markers. A relative-path guidance improvement and fresh live verification are required before claiming this workflow passes.

After adding explicit relative-path guidance and actionable boundary errors, the second 4B CPU run completed all three required file operations correctly: reading the source, writing its observed `BIRCH-629` code into a separate result, and reading the result back. Source bytes remained unchanged. The model then returned `WORKFLOW_READY` without the requested code, so the strict end-to-end check still failed. The optional memory summary hit its 60-second budget; the new fallback preserved the extractive checkpoint and returned the model's visible answer instead of failing the task. Total time was 224.05 seconds. This validates path recovery and nonfatal memory fallback, while exposing a remaining 4B instruction-following limitation. It must not be presented as a complete workflow pass.

## Verified 9B workflow and resume

The actual Huihui Qwen3.5 9B abliterated model passed the strict CPU file workflow with four threads and an 8K context. It read the source, wrote the observed code to `result.md`, reread the result, and returned both `WORKFLOW_READY` and `BIRCH-629`. The original input was unchanged. The visible answer was produced at 195.10 seconds; the complete turn returned at 255.40 seconds after the optional memory budget expired and preserved the extractive checkpoint.

That run exposed a cancellation race: a terminated inference server could remain alive while draining requests, and a following turn could reuse it. The backend now kills and reaps only its own cancelled inference worker before returning and refuses to reuse a stopping worker. Model weights are read-only; task history and file receipts remain in the Python application's durable store.

A fresh engine then reopened the same saved task and passed a strict read-only resume check in 254.75 seconds. It read `result.md`, returned `RESUME_READY` with `BIRCH-629`, and preserved both file contents and the result's modification time. It performed no write, edit, or command. After another optional memory timeout, an immediate new request loaded a fresh worker and returned `RELOAD_READY`, verifying the cancellation race was fixed. All test-owned model processes were then stopped.

The model's 4B sibling still missed the requested final output format in its otherwise correct file run. The 9B result supports using 9B for this tested task; it is not a general accuracy guarantee for arbitrary long work.

## GPU contention guard

Before a fresh NVIDIA model load, loCAII checks free VRAM when `nvidia-smi` is available. Obvious contention produces an actionable message before model or KV allocation. On this computer, the read-only check correctly found 1,195 MiB free while the existing LM Studio model was active, versus approximately 6,646 MiB estimated for the default 9B profile. No competing model was loaded or stopped.

The check is an estimate and does not guarantee every architecture fits. It skips CPU-only profiles, explicitly selected integrated GPUs, unavailable NVIDIA utilities, and ambiguous multi-GPU mappings. Vulkan does not require CUDA tooling. Setting GPU layers to zero in the Models page explicitly selects device `none`, so a CPU fallback does not require editing hidden settings. Profiles are not changed automatically to work around contention.

## Verified image understanding

The actual 4B model with its matching Q8 projector correctly identified a synthetic red square on the left and a blue circle on the right. Its response was exactly `Left: red square. Right: blue circle.` The CPU-only run used four threads and an 8K context, loaded in 3.49 seconds, and finished in 15.04 seconds. NVIDIA snapshots were unchanged before, during, and after. The conservative image budget was 2,119 tokens; the server measured 174 actual prompt tokens. This validates image transport, projector loading, and this simple recognition case, not general visual accuracy. The reproducible fixture and checker are in `scripts/check_vision.py`.

## Reproduce validation

Run in the loCAII environment from the repository directory. Reports, synthetic workspaces, and screenshots remain under ignored `.local/verification/`.

```bash
# CPU defaults. The 9B file workflow needs several minutes on CPU.
python scripts/check_cpu_workflow.py --model qwen35-9b-abliterated --threads 4 --deadline 480
python scripts/check_vision.py

# Reopen the exact saved task, verify read-only resume, and exercise the next request.
python scripts/check_cpu_workflow.py --resume-report /absolute/path/to/report.json --deadline 480

# Explicit GPU checks: run only after other GPU tasks have finished and memory is free.
python scripts/check_cpu_workflow.py --gpu --model qwen35-9b-abliterated --threads 6 --deadline 480
python scripts/check_vision.py --gpu --model qwen35-9b-abliterated --threads 6
```

`--gpu` uses full GPU offload, a 32K context, and the matching vision projector on CPU. CPU mode remains the default and retains the 8K context and explicit zero offload. The GPU checks refuse obvious VRAM contention and never release another application's model. The 9B GPU workflow, read-only resume and vision checks passed on 2026-09-11; see [verification results](VERIFICATION.md). Every report records actual settings and strict outcomes.

## Measured GPU result

On 2026-09-11 the 9B abliterated model passed the same synthetic file workflow with full NVIDIA Vulkan offload, six threads, 32K context, q8 KV cache and the CPU vision projector. Workflow including memory took 38.63 seconds; read-only continuation took 11.54 seconds. A fresh-process resume after the memory evidence fix took 25.89 seconds including load and a subsequent inference request. The earlier CPU trial used four threads and 8K context and took 255.40 seconds for the workflow, so these single-run timings include different context, caching and memory-summary behavior. They are not a controlled throughput benchmark.

GPU vision also passed the synthetic two-shape check in 8.33 seconds. The owned server allocated 5,446 MiB for that run. No LM Studio inference server was used. Optional summaries remain fallible; exact copied evidence and the archived transcript are retained separately.

## Stop and model release verification, 2026-09-14

The chat Stop action now unloads the owned model even after a task error or
completion. A separate lifecycle interrupt bypasses blocked prediction and token
counting; process cleanup runs outside the Qt thread. New work waits for cleanup.
A concurrent Stop invalidates pending startup before it can allocate another model.

`python scripts/check_stop_release.py` tested a private Qwen3.5 9B GPU worker:
GPU usage was 27 MiB before loading, 5,489 MiB loaded, and 27 MiB after Stop.
Cancellation and worker exit took 0.142 seconds in this single run. Both the
inference thread and owned process exited; the outcome was `ModelCancelled`.
This is an observed functional check, not a latency guarantee or population estimate.
Other GPU applications were not stopped. The report is private under
`.local/verification/stop-release/10e96f7f441f4de7a9d51a9074e4f6de/report.json`.
