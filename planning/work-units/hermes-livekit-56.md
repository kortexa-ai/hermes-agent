# Gateway environment-probe startup warm-up

Owner: [hermes-livekit #56](https://github.com/kortexa-ai/hermes-livekit/issues/56),
under the [realtime latency program](https://github.com/kortexa-ai/hermes-livekit/issues/14).
Upstream: [hermes-agent #106064](https://github.com/NousResearch/hermes-agent/issues/106064).

The first agent prompt waits for the process-local Python toolchain probe. Run
that existing probe alongside the gateway's startup prerequisites so its cache
is available before the first user turn. Keep the provider metadata warm-up.

## Invariants

- Honor `agent.environment_probe` and the caller's profile terminal policy.
- Let the existing resolver omit host inspection for remote terminal backends.
- Reuse its single worker, bounded wait and process-local cache.
- Preserve the existing bounded inbound gate and lazy fallback on failure.
- Do not construct an agent or capture session prompts, memory or toolsets.
- Keep Mira's model, reasoning setting, profile files and tool capabilities unchanged.

## Validation and delivery contract

Run the canonical test runner for environment probing, startup gating and model
metadata on Linux and macOS. Check first-turn preparation with real subprocesses
and compare probe output before and after startup warm-up. Use a silent,
text-only Realtime canary after a profile-scoped gateway restart; this does not
prove physical audio playback. Keep code in the separate upstream contribution
branch and cherry-pick the fix into `kortexa-ai/main` for Mira. Record measured
results, deployment and rollback evidence in the owning issue, not this note.
