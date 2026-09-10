# Verification on this computer

Tests use synthetic files in isolated workspaces. No historical research-data task was executed, and original input bytes were checked before and after the live file tests.

## Environment

RTX 4070 Laptop GPU with 8 GB VRAM, 32 GB system RAM, Python 3.12 in the dedicated `locaii` Conda environment, PySide6 6.11.2, official llama.cpp b10893 Vulkan runtime with verified release hash. During CPU tests, LM Studio continued its existing generation and held about 6.6 GB VRAM. The user explicitly asked to let that generation finish, so it was not stopped or unloaded.

## Automated regression coverage

The current suite covers native tool request/result serialization, exact rendered-text token counting, authenticated loopback inference, model lifecycle/cancellation, durable archives and resume, incomplete memory fallback, preserved original instructions, no fabricated empty completion, workspace boundaries and backups, protected measurement files, bounded attachments, terminal cancellation and recovery, supervision opt-in and structured reviews, media queue ownership and crash recovery, and Qt page/worker behavior. Deterministic fixtures exercise orchestration, not model intelligence.

**100 tests passed** in the integrated suite. Python compilation, shell syntax, dependency consistency and native desktop launch checks also passed.

## Actual model checks

- **4B protocol smoke: passed**, 40.11 seconds total on CPU with two threads and 8K context. Visible answer, native tool request, supplied receipt continuation, HTTP 401 without authentication, and token-counter agreement were observed.
- **4B full workflow baseline: failed**, 171.93 seconds. The model mistyped absolute workspace paths. Four rejected operations caused a visible bounded stop. No source alteration or fabricated result was accepted.
- **4B after path guidance: file work passed, strict final reply failed**, 224.05 seconds. It read the source, wrote the observed code into a separate output, and reread that output. It omitted the code in its final reply. The optional model-written memory timed out; extractive memory remained usable and the completed file work was preserved.
- **9B full workflow: passed**, including real read/write/reread, preserved input and the correct final verification code. Resume initially failed because a timed-out optional memory summary left its owned inference process draining. The backend now kills and reaps that process before allowing a reload. A regression test covers this lifecycle fault. **The same saved task then passed read-only resume in 254.75 seconds**: it reread the saved code, returned the correct final marker, preserved source/result hashes and modification time, and performed no writes. Its optional memory summary timed out safely; an immediate fresh model load/request returned `RELOAD_READY`, directly verifying the process-reuse fix. The earlier failed resume remains recorded separately.

**4B CPU vision passed in 15.04 seconds**, including model/projector load: a synthetic image containing a red square on the left and a blue circle on the right produced the exact correct description. The matching projector ran on CPU; NVIDIA allocation remained unchanged. This checks simple image ingestion and grounding, not general visual reasoning.

These are single trials under concurrent load, not a statistical performance ranking. The larger model and thread counts differ. The first failed run is retained rather than discarded to make the improvement visible. The live harness is `scripts/check_cpu_workflow.py`; raw reports and transcripts stay in ignored `.local/verification/`.

## Compaction and resumption

The reproducible stress check archives an oversized synthetic tool receipt, reduces the active prompt below its response reserve, reopens the saved checkpoint, and checks the complete archive against the original database using SHA-256. Every original user constraint remains in the prompt. It uses a deterministic JSON-character token proxy, explicitly not measured model tokens or inference speed. Production text budgeting uses the loaded model's rendered template and tokenizer.

See [machine-readable measurements](benchmarks/measurements.json), [workflow CSV](benchmarks/cpu-workflows.csv), and [PNG](benchmarks/verification.png), [PDF](benchmarks/verification.pdf), [SVG](benchmarks/verification.svg) figures. No inferential statistics are appropriate for these deterministic checks and single live trials.

## Supervision

A real authenticated Codex CLI invocation returned a structured `continue` verdict for an unsupported completion claim, requiring actual write/read receipts. It consumed 13,985 reported tokens for a 2,162-character packet. The initial estimated allowance was exceeded, exposing hidden CLI context overhead; subsequent code reserves additional CLI headroom and shows actual usage. The allowance remains an estimate, not a billing guarantee. See [supervision details](SUPERVISION.md).

The regression suite verifies that a supervisor plan reaches the local model and that a `continue` verdict causes another local iteration before completion. OpenAI/Anthropic API and Claude CLI adapters are implemented but have not been live-tested with credentials in this task.

## Desktop and media

The desktop launched natively on the user's display, and screenshots were inspected. Desktop/menu shortcuts and single-instance activation are installed. Native QMediaPlayer also played a synthetic one-second H.264 clip through EndOfMedia with no errors. This validates the player, not video generation. Image and video controls connect to real ComfyUI workflows, with 13 required source node classes and four required weight headers found locally. The media adapter has tests for job completion, cancellation, external queue ownership, missing outputs, batch crash recovery, refused connections and uncertain outcomes.

A separate **CPU-only ComfyUI readiness check passed in 7.65 seconds**: all 18 graph nodes matched the live `/object_info` schemas, CPU execution and an empty queue were confirmed, zero prompts were submitted, and the owned process was stopped. FFmpeg/ffprobe 9.0.1 is installed for actual video validation.

Real GPU image/video generation and 9B GPU/vision load remain pending while the user's existing LM Studio generation occupies the GPU. Installed files, offline tests and UI screenshots are not presented as proof that rendering has succeeded. The reproducible `scripts/check_media.py` refuses to start its real checks when a foreign GPU workload lacks sufficient headroom.

## Practical limits

The app preserves progress and exposes failures; it cannot make 4B reasoning equivalent to a larger model. CPU inference is usable but slow. Increase model size, add independent review, or narrow the task when small-model instruction following fails. Large contexts are not a substitute for compaction on an 8 GB GPU. File-tool restrictions do not sandbox optional shell commands or Python extensions.
