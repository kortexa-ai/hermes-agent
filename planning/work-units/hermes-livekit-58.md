# Gateway TTS tool-boundary delivery

Owner: [hermes-livekit #58](https://github.com/kortexa-ai/hermes-livekit/issues/58),
under the [realtime latency program](https://github.com/kortexa-ai/hermes-livekit/issues/14).
Upstream: [hermes-agent #106115](https://github.com/NousResearch/hermes-agent/issues/106115)
and [PR #106161](https://github.com/NousResearch/hermes-agent/pull/106161).

Flush buffered speech when a model response yields to a tool. Deliver visible
completed commentary without waiting for the tool result or duplicating text
already accepted through the delta callback. Keep one audio handle per turn.

## Invariants

- `None` is a non-finalizing gateway segment boundary, not text or turn completion.
- Honor the interim commentary opt-out, independently of transcript streaming.
- Keep separate visible segments separate in speech, including a pending delta
  followed by completed Codex commentary.
- Preserve stale, finished, inactive and cancelled consumer guards.
- Preserve fallback suppression after audible output and normal final response delivery.
- Keep Mira's profile, model, reasoning, tool capabilities and custom WPE unchanged.

## Validation and deployment contract

Exercise the real agent delivery mixin, gateway callbacks, chunker and async
consumer with an in-memory speech provider and PCM sink. Hold the tool result
until the expected acknowledgment reaches that sink. Run the canonical test
runner on Linux and macOS for gateway TTS, text/interim delivery and chunking.

Keep the upstream contribution branch separate from `kortexa-ai/main`. Before
deploying to Snappy, verify the exact integration diff, a clean production
checkout, current dependency health and a focused revert path. Restart only
Mira through its normal gateway lifecycle. Confirm both adapters reconnect and
run an isolated text-only Realtime canary with per-chat voice disabled. That
canary checks deployment health; the in-memory PCM test checks this fix's speech
ordering. Neither establishes physical speaker quality.

Record test results, upstream review, Project membership, deployment and rollback
evidence in the owning issue rather than this file.
