# Supervision and token economy

Local work is the default. In Supervision, choose a provider, review interval, iteration limit and allowance. No remote request occurs unless remote supervision is explicitly enabled. A supervisor first proposes a concise plan, evaluates milestones and final evidence, and returns structured `complete`, `continue`, or `blocked`. `continue` launches another local iteration. It does not grant new filesystem permissions.

## Modes

- Off: ordinary local agent execution with automatic task memory.
- Local: the loaded model reviews a compact packet. This uses no external tokens, but does not provide independent higher-quality reasoning.
- Manual: Export packet creates a task-local JSON file. Ask an existing Codex or Claude conversation to review it. Import review validates the result; Continue applies it. For stale-state protection, wrap the review as `{"history_digest":"<packet hash>","review":{...}}`.
- Codex CLI: uses the installed, authenticated `codex exec`, structured output, ephemeral state and read-only execution policy. Shell, apps, browser, computer use and agent delegation are disabled for the reviewer. The work is executed by the local model.
- Claude CLI: uses `claude -p` with all built-in tools disabled, safe mode to disable user customizations, an explicitly empty strict MCP configuration, JSON schema output, no session persistence and a per-invocation USD allowance. The user's existing CLI authentication is used by the CLI itself.
- OpenAI/Anthropic API: configure the exact provider model ID and `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the launch environment. Keys are never extracted from other apps or saved into project settings. These adapters use the official Responses and Messages endpoints.

## What is sent

A packet includes bounded user requests, memory, recent observed tool receipts and the latest assistant response. Recognizable credential patterns are redacted, but automatic redaction is not a guarantee: inspect manual packets and choose an appropriate workspace before enabling remote reviews. Original files and full chat histories are not automatically uploaded. The packet reports that it is bounded so missing observations cannot be interpreted as success.

## What was measured

A real Codex CLI check correctly refused to accept an unsupported completion claim and requested actual write/read receipts. Its 2,162-character packet resulted in 13,743 input tokens and 242 output tokens, including CLI system context. This exposed the importance of measuring the entire provider invocation, not just the packet. The initial 12K estimate was exceeded; the implementation now reserves an extra 18K tokens per CLI review before dispatch and displays actual reported usage.

The allowance stops additional reviews when insufficient estimated capacity remains. It is not a guaranteed billing cap, especially for CLI agents with hidden context or reasoning. APIs bound output tokens; the app conservatively reserves input/output capacity and records actual usage when returned. Check your provider's own account limits for an independent financial cap.

For economy, use manual review in an existing assistant conversation or API review at meaningful milestones, and let the local model do intermediate file/tool work. Do not assume a universal percentage saving. Model quality, retries, packet size and review frequency all affect the result.

## Sources

The Codex adapter follows [official non-interactive execution documentation](https://developers.openai.com/codex/noninteractive/). The API adapters use [OpenAI Responses](https://platform.openai.com/docs/api-reference/responses) and [Claude Messages](https://platform.claude.com/docs/en/api/messages/create). CLI flags were checked against the installed versions. The Codex CLI route was live-tested; API and Claude CLI routes require the user's valid credentials and are covered by local validation tests, not a claimed live cloud run.
