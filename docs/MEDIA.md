# Local image and video generation

loCAII starts ComfyUI directly with its own Python environment. The desktop interface and media adapter do not use LM Studio, its API, its launcher or its model manager. Existing model weights are reused without copying them.

The bundled image workflow uses RealVisXL V5.0 Lightning. The video workflow uses Wan 2.2 TI2V 5B Q5_K_M, UMT5 FP8 on CPU, and the Wan VAE. The installed ComfyUI GGUF loader handles the quantized video weights. These are external dependencies, not Python packages distributed with loCAII.

## Configuration

The main application JSON configuration accepts an optional `media` object. Existing installation defaults are:

```json
{
  "media": {
    "comfy_dir": "/home/andry/.local/opt/ComfyUI",
    "python": "/home/andry/miniconda3/envs/local-media/bin/python",
    "url": "http://127.0.0.1:8188",
    "output_dir": "/run/media/andry/andryData/AI_Generated",
    "startup_timeout": 180,
    "generation_timeout": 3600,
    "poll_seconds": 1
  }
}
```

ComfyUI listens on loopback only. The adapter enables memory mapping, reserves 1 GB of GPU memory, disables API nodes, and permits only the installed GGUF loader and local CPU text-encoder custom node. The chat model is released through loCAII's model manager before media generation. Chat loads lazily again when needed. An active chat operation must finish before the app begins media work.

If an existing ComfyUI process is available, loCAII reuses it only when its queue is idle. It never sends an unscoped interrupt. Cancellation uses the installed engine's atomic `/api/jobs/{id}/cancel` route. Older engines without that route can delete a queued job by ID but cannot safely cancel an unowned active job; the interface reports this limitation explicitly.

The app stops a ComfyUI child process it started when the job finishes. If another client queued work on that process, it leaves the server running to preserve that work. On a shared idle server, it requests unloading models and freeing memory. The configured output directory must match an already running shared server's output directory.

## Controls and media inputs

`generate(kind, prompt, params, cancel, emit)` accepts image/video kind and the following controls where supported by the workflow:

- `workflow`: registered workflow name.
- `width`, `height`: multiples of 32, from 64 to 2048.
- `steps`: 1 to 100.
- `seed`: nonnegative 63-bit integer.
- `frames`: 8k + 1, from 9 to 129 for the bundled Wan workflow.
- `fps`: 1 to 60.
- `negative_prompt`: a separate negative description.
- `timeout`: total allowed seconds including startup.
- `image_path` or `input_image`: existing image up to 32 MiB for the Wan workflow.

Wan image-to-video uses ComfyUI's verified `Wan22ImageToVideoLatent.start_image` input. The source is validated with Pillow, copied into the ComfyUI input directory under a unique job ID, then loaded by `LoadImage`. No source image is modified. Image inputs are rejected clearly for workflows without a supported image connection.

Larger dimensions and longer videos can exhaust memory. The controls expose the model's workload rather than promising that every configuration fits the installed GPU. Begin with the bundled presets and increase one setting at a time.

## Adding workflows

Export the graph using ComfyUI's **API format**. Visual editor exports with `nodes` and `links` are not executable API graphs. Use the workflow import button or `register_workflow(name, path, kind)`.

Common positive prompt, seed, dimensions and sampling controls are discovered from the graph. For ambiguous graphs, wrap the API JSON with explicit mappings:

```json
{
  "graph": {
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "Your positive prompt", "clip": ["2", 0]}}
  },
  "mapping": {"prompt": ["3", "text"]}
}
```

The example illustrates the wrapper schema only. A runnable graph must also contain its loader, sampler, decoder and output nodes. Add entries such as `"steps": ["12", "steps"]` to expose more controls. A verified Wan graph can additionally set `"image_input_node": "5"` if node 5 is `Wan22ImageToVideoLatent`.

Imported workflows are copied into the app state directory. Node packages and referenced weights must already exist in ComfyUI. Custom nodes can execute code, so import graphs and install custom nodes only from sources you trust. The startup custom-node allowlist can be extended in the adapter when installing and validating a new engine integration.

## Receipts and progress

The state directory contains `media/jobs/<job_id>.json`, complete engine history alongside successful or failed completed jobs, and `media/comfy.log` for engine startup diagnostics. Receipts contain prompt, parameters, workflow, timestamps, status, actual output files and error details. Files are accepted only when nonempty and inside the configured output directory.

Progress reports actual queue state and elapsed seconds. A running engine may be loading weights, encoding text, sampling or decoding. The adapter does not invent a completion percentage. These receipts are execution evidence; content quality still requires looking at the generated image or video.

Run offline regression checks with:

```bash
python -m pytest tests/test_media.py -q
```

## Recovery after an application crash

`jobs()` reads local receipts only and is safe for the interface thread. Run `recover_jobs()` in a background worker at startup or when refreshing job history. It checks queue and recent history in two batched requests, each with a two-second timeout, rather than making one request per job. Completed jobs are recovered only after validating actual output files. Engine errors become failed receipts. Active prompt IDs remain running. A refused connection marks the previous run interrupted with an explicitly unconfirmed outcome; temporary timeouts produce an unknown state that can be checked again. Missing or expired history is never treated as proof that the job succeeded, and recovery never automatically repeats a generation.

When loCAII starts its own ComfyUI process, a lightweight detached watcher remembers both the GUI and engine process start identities. If the GUI disappears unexpectedly, the watcher allows the engine queue to finish, then requests idle model unloading. It leaves the engine history available for the next app session. It never sends a global interrupt or terminates another client's queued work. Process start identities prevent accidental action after a process ID is reused. Recovery also requests idle model unloading when a surviving engine has no queued work.

## Reproducible real smoke test

`python scripts/check_media.py` performs read-only installation checks: source-defined workflow nodes, actual weight paths and file headers, and the ffprobe executable. It does not import torch, load models, or start an engine.

`python scripts/check_media.py --run` additionally renders a 512 x 512 image and a 832 x 480, 33-frame video with fixed prompts and seed 42. It decodes the image with Pillow and counts actual video frames with ffprobe. Reports, event logs and engine logs are stored under `.local/verification/media/`, which is ignored by Git. Use `--kind image` or `--kind video` for one pipeline only. This verifies generation and media integrity; the short smoke settings are not a quality benchmark.

Both the smoke test and the application refuse to start media generation over another application's GPU compute workload. The application may reuse only a matching configured ComfyUI process with an idle queue. It does not stop foreign processes or unload their models. The default free-memory minimum is 6500 MiB and can be configured through `media.min_free_vram_mib`.

The smoke test locates ffprobe through PATH or alongside the active Python executable, so conda activation is unnecessary. Install `ffmpeg` from conda-forge into the loCAII environment if its binary is missing.

`python scripts/check_media.py --cpu-api` performs an intermediate runtime check without rendering. It requires 4 GiB available RAM, starts an owned CPU-only ComfyUI process on unused port 18188 with CUDA hidden, and checks both graphs against the live `/object_info` registry. The check verifies required inputs, actual weight-name choices, linked output indices, numeric bounds, CPU device selection and an empty queue. It never submits a prompt. The owned process is stopped before returning, and live node schemas plus a JSON report are preserved alongside its log. This validates runtime node registration while leaving generation and content quality to the separate GPU smoke test.

## Measured generation and recommended starting preset

On 2026-09-11 RealVisXL produced an inspected 512 × 512 red-teapot PNG in 16.08 seconds. Wan produced an inspected 832 × 480, 33-frame H.264 clip in 121.34 seconds, including startup and text encoding. The video uses 20 sampling steps, seed 42 and 24 FPS. Its 1.375-second duration and 33 decoded frames were verified independently; extracted first/middle/last frames show a coherent red-hulled sailboat moving across blue water. These are single fixture timings, not quality guarantees for arbitrary prompts.

Start with the bundled Wan preset. Tiny 512 × 288, 9-frame trials produced severe visual distortion at both 8 and 20 steps, even though the MP4s decoded successfully. Changing the decoder or disabling dynamic VRAM did not fix those tiny trials. Increasing resolution and length together to the preset above produced coherent frames. Full VAE and GGUF hashes matched published files; a separate VAE reconstruction succeeded. We did not establish independent minimum width, height or frame count.

The Wan template uses the standard `VAEDecode` node from the [official ComfyUI template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_5B_ti2v.json). ComfyUI can fall back to tiling on insufficient decoder memory. Detailed failures and visual assessments remain in [verification](VERIFICATION.md) and the benchmark CSV. Test controls `--video-width`, `--video-height`, `--video-frames`, and `--video-steps` make smaller and larger trials reproducible. `--legacy-memory` (or `media.disable_dynamic_vram: true`) is a troubleshooting option; it was tested and did not improve the distorted tiny trial. It stays off by default.
