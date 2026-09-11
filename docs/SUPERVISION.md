# Supervision and token economy

## Open a supervisor and delegate local work

1. Select a local model and workspace. In Supervision, choose **Codex CLI** or **Claude CLI**.
2. Click **Open & connect supervisor**. A terminal opens using that CLI's existing sign-in and default model. The supervisor receives the selected local model as a subordinate and checks the connection.
3. Give the supervisor your objective in that terminal. It plans the work, delegates simple to moderate jobs to the local model, inspects observed receipts, and requests a focused correction or verification when needed. Delegated tasks appear in loCAII's task list.

Keep loCAII open. **Copy launch command** provides instructions for starting a new connected terminal manually. It does not attach tools to an already running Codex or Claude conversation. **Disconnect** revokes new requests and cancels queued jobs; the active local task can finish. Use loCAII's **Stop** to interrupt active work.

The primary flow requires no API key field, review interval, JSON import, or model-name entry. The CLI must already be installed and signed in. The connection indicator reports when the supervisor has contacted loCAII; it does not claim that a terminal remains alive after its last request. Missing CLI/configuration errors appear in the app, and terminal startup errors remain visible.

### Delegation and scope

The bridge exposes five tools: `local_model_info`, `delegate_task`, `task_status`, `continue_task`, and `cancel_task`. Instructions tell the supervisor to delegate file inspection, extraction, summaries, small edits, repetitive operations and concrete validation. Complex reasoning, decomposition and final evaluation remain the supervisor's responsibility. This guidance encourages delegation; it cannot guarantee every model will choose well.

Each connection freezes the selected workspace, model and permission preferences. Change them by disconnecting and reconnecting. The supervisor cannot widen those preferences through tool arguments. Only bounded task instructions and resulting evidence cross the bridge. Read/write permissions still apply to the local worker. An explicit delegation can return file content to the external supervisor, so select the workspace and read preferences accordingly.

Jobs share the same desktop worker as local chat and media, preventing concurrent GPU loads. Unsent drafts are preserved. A unique request ID makes network retries idempotent. Continuations reuse the saved task session. After an app restart, unfinished jobs are marked interrupted for inspection and are never silently replayed. The local subordinate's external review is disabled, preventing recursive supervisor calls.

The connection uses authenticated loopback HTTP behind a stdio MCP adapter. Credentials stay in a private local file. Generated CLI settings apply only to the launched session; global CLI configuration and authentication files are unchanged. Other configured Codex MCP servers and plugins are disabled for this launch; Claude uses a strict single-server MCP configuration. Built-in execution tools are disabled so the supervisor uses the scoped local worker.

### Cost and evidence

The supervisor polls efficiently with `task_status(wait_seconds=20)` and receives bounded progress, actual tool receipts, final output and local token usage. It checks observed evidence rather than accepting a completion claim. The app does not automatically upload the full conversation or original files.

**Interactive terminal usage belongs to the CLI provider account and is separate from loCAII's automatic-review allowance.** The app does not meter or cap the live terminal's provider tokens. Delegating work can reduce external execution tokens, but no fixed savings percentage is promised. Account limits, model choice, task complexity and retries affect cost.

## Advanced automatic reviews

Expand **Advanced review settings** for automatic review and manual packet workflows. **When to review** defaults to **Tool work only (recommended)**: ordinary chat responses do not request a supervisor or consume its allowance, even in a conversation that previously used tools. **Every message, including conversation** explicitly enables initial planning and reply reviews. Tool-based work retains milestone/final evaluation and correction iterations. Local work remains the default. Remote review only runs when explicitly enabled. A reviewer proposes a concise plan, evaluates milestones and final evidence, and returns structured `complete`, `continue`, or `blocked`. `continue` launches another local iteration, without granting additional filesystem permissions. These reviewers are separate from the interactive supervisor connection above.

### Review modes

- Off: ordinary local agent execution with automatic task memory.
- Local: the loaded model reviews a compact packet. This uses no external tokens, but does not provide independent higher-quality reasoning.
- Manual: Export packet creates a task-local JSON file. Ask an existing Codex or Claude conversation to review it. Import review validates the result; Continue applies it. For stale-state protection, wrap the review as `{"history_digest":"<packet hash>","review":{...}}`.
- Codex CLI: uses the installed, authenticated `codex exec`, structured output, ephemeral state and read-only execution policy. Shell, apps, browser, computer use and agent delegation are disabled for the reviewer. The work is executed by the local model.
- Claude CLI: uses `claude -p` with all built-in tools disabled, safe mode to disable user customizations, an explicitly empty strict MCP configuration, JSON schema output, no session persistence and a per-invocation USD allowance. The user's existing CLI authentication is used by the CLI itself.
- OpenAI/Anthropic API: configure the exact provider model ID and `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the launch environment. Keys are never extracted from other apps or saved into project settings. These adapters use the official Responses and Messages endpoints.

### Review packets

A packet includes bounded user requests, memory, recent observed tool receipts and the latest assistant response. Recognizable credential patterns are redacted, but automatic redaction is not a guarantee: inspect manual packets and choose an appropriate workspace before enabling remote reviews. Original files and full chat histories are not automatically uploaded. The packet reports that it is bounded so missing observations cannot be interpreted as success.

### Automatic review measurements

A real Codex CLI check correctly refused to accept an unsupported completion claim and requested actual write/read receipts. Its 2,162-character packet resulted in 13,743 input tokens and 242 output tokens, including CLI system context. This exposed the importance of measuring the entire provider invocation, not just the packet. The initial 12K estimate was exceeded; the implementation now reserves an extra 18K tokens per CLI review before dispatch and displays actual reported usage.

The allowance stops additional reviews when insufficient estimated capacity remains. This skips optional reviews for the rest of the current turn and records a neutral notice, while local work continues. A skipped review is never recorded as approval. The next turn rechecks the remaining allowance. Provider errors and explicit supervisor blockers still surface. It is not a guaranteed billing cap, especially for CLI agents with hidden context or reasoning. APIs bound output tokens; the app conservatively reserves input/output capacity and records actual usage when returned. Check your provider's own account limits for an independent financial cap.

For economy, use manual review in an existing assistant conversation or API review at meaningful milestones, and let the local model do intermediate file/tool work. Do not assume a universal percentage saving. Model quality, retries, packet size and review frequency all affect the result.

## Live delegation verification

A real Codex CLI session used the MCP adapter and Qt worker to delegate to Qwen3.5 9B abliterated at 32K context on the GPU. It read a synthetic verification code, wrote and read back the result, then requested a read-only continuation in the same saved session. Both jobs completed, original bytes and the result modification time were preserved, the unsent desktop draft survived, and Codex returned the observed code. Total elapsed time was 59.59 seconds. No recursive external reviews ran.

The CLI reported **127,567 input tokens, including 86,912 cached input tokens, and 499 output tokens** across its turn. Cached input is a subset of input, not an additional amount. The local jobs reported 13,794 tokens. This short test demonstrates working delegation, **not token savings**. CLI system context and repeated evidence evaluation can make supervision disproportionately expensive for tiny jobs. Prefer meaningful batches and milestone reviews for longer work.

The prior direct local file-workflow plus continuation check took 50.49 seconds. These are single trials with different prompts and memory timings, so the [comparison figure](benchmarks/delegation.png) and [CSV](benchmarks/delegation.csv) do not establish a speedup, overhead estimate, or statistical ranking. Full private receipts remain under `.local/verification`.

Reproduce the paid/GPU integration check explicitly with `python scripts/check_delegation.py --live`. Ordinary tests do not use a provider or load a model. Native terminal execution was also verified using a synthetic CLI, without a provider call. Claude command/configuration behavior is covered by tests; its live provider route has not been verified.

## Verification and sources

The interactive connection follows the [official Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli), including stdio transport and MCP server instructions. The Codex review adapter follows [official non-interactive execution documentation](https://developers.openai.com/codex/noninteractive/). The API adapters use [OpenAI Responses](https://platform.openai.com/docs/api-reference/responses) and [Claude Messages](https://platform.claude.com/docs/en/api/messages/create). CLI flags were checked against the installed versions. The Codex CLI route was live-tested; API and Claude CLI routes require the user's valid credentials and are covered by local validation tests, not a claimed live cloud run.

## Simple-chat allowance regression

Compared identical synthetic chat fixtures against commit b6ae8cb and the fix: automatic review calls decreased from two to zero under default settings. A reply with insufficient automatic-review allowance previously ended in error; it now completes locally. The test uses scripted answers and no provider or GPU calls. [CSV](benchmarks/review-scope.csv) and [figure](benchmarks/review-scope.png) record behavior, not model quality or timing. Reproduce with `python scripts/benchmark_review_scope.py --before b6ae8cbb0c55eb085a7e9fad6408d22f61cc10ca`.
