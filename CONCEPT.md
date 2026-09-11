# loCAII: local creative and work assistant

loCAII is a modular PySide6 desktop workbench for local language models, files, images, and video. It runs its own llama.cpp server and talks directly to ComfyUI. LM Studio is not a runtime dependency. Existing downloaded model weights are reused without copying many gigabytes into the repository.

## Experience

A calm desktop interface combines a task sidebar, a streaming conversation, attached files, a workspace browser, and an inspectable activity panel. Users can switch between Chat, Create, Models, and Supervision. Long operations run off the UI thread, have visible progress and Stop controls, and save durable checkpoints. A desktop shortcut and launch script start the app with a double click.

## Architecture

Python modules separate storage, inference, tools, task execution, media generation, supervision, and UI. Models and generation workflows have registries so new local weights or ComfyUI API graphs can be added. Third-party extensions expose tool schemas and implementations through an explicit local extension directory. The model server listens only on loopback. On this 8 GB GPU, model profiles start conservatively with one prediction slot and a 32K context. GPU-intensive chat and media operations are serialized; the app releases its language model before a render.

## Long tasks and correctness

An agent loop executes native model tool calls, persists completed receipts, maintains Markdown memory and a plan, and can resume after interruption. Failed or incomplete model summaries must not strand a task: original messages are archived before a measured, extractive fallback is adopted. The active prompt retains real user instructions and tool request/result consistency. Writes use backups and target the chosen workspace by default. Raw scientific data and metadata must not be changed automatically. File contents and tool outputs are evidence, not new permission. Empty responses and transient read errors receive bounded recovery; failed commands are not replayed indefinitely. Long commands expose durable jobs with polling and cancellation.

## Economical supervision

Local mode uses the small model for planning, execution, and self-evaluation. Optional OpenAI or Claude supervisors receive a bounded objective, plan, evidence receipts, changed-file summaries, and validation results at configured milestones, not every token of conversation. A supervisor returns structured evaluation and next steps; local iterations do most work. A token budget, maximum iterations, pause/stop, and explicit completion criteria constrain the loop. CLI or API supervisors require the user's configured authentication; the app does not extract credentials from other applications. Manual packet export/import supports review by an existing assistant task without unattended remote calls. Report actual token counts when provided and distinguish estimated tokens from billed usage. Do not promise a fixed financial saving without measurement.

## Delivery and evaluation

Build and test on this machine with Huihui Qwen3.5 abliterated GGUF. Validate a real multi-step task on synthetic files, interruption/resume, archive fallback, error recovery, supervisor-guided iteration, file import, and actual image/video output. Inspect the live Qt UI and its screenshots. Preserve measurements as CSV and figures. Publish source and documentation to a private Andrianarivelo/loCAII repository on main, excluding private history, credentials, weights, generated media, runtime binaries, and development memory.

## Initial boundaries

This is a complete extensible first desktop version, not a claim to reproduce every undocumented feature in Codex. The prior local fixes form the reliability baseline. Optional cloud providers are usable only with valid authentication and their availability is shown honestly. A stopped or unreachable backend produces a visible recoverable state, never an empty success.

## Simplified interactive supervision

Open & connect launches an existing Codex or Claude CLI in a terminal with a session-specific local-model bridge. The supervisor owns difficult reasoning and final evaluation, while bounded simple/moderate tasks run through the same durable local worker as chat. Five MCP tools provide model/scope discovery, idempotent delegation, efficient status waiting, saved-session continuation and cooperative cancellation. Existing automatic-review controls remain available under Advanced review settings. Scope is frozen per connection; subordinate remote review is disabled. Interactive CLI usage is separate from the automatic-review allowance.

## Durable live task steering

The chat composer remains available during local work. New text is stored in a bounded, per-task SQLite queue, then atomically inserted into the conversation at engine boundaries. The engine checks before each tool dispatch and after prediction, review and memory work. Undispatched calls receive explicit nonexecution receipts before new instructions enter the native tool-call history. Existing results stay intact. The UI shows queued and delivered states, preserves pending updates on Stop/restart, and closes the ordinary-chat completion race by continuing with frozen settings after worker exit. Text steering does not alter model, workspace or permissions.
