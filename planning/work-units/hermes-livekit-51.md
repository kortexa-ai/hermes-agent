# Responses first-event timing

Tracking: https://github.com/kortexa-ai/hermes-livekit/issues/51

This note uses the approved core-latency tracking exception. Implementation
belongs in Hermes Agent; hermes-livekit consumes the existing API timing hook.
Keep this fork-only execution note out of an upstream contribution.

Record the first accepted parsed Responses event once per API attempt. Lifecycle
events before text count. Retired requests cannot write the active attempt's
timestamp, and streams without accepted events remain unset. Preserve the
existing per-attempt reset, retry behavior, text/reasoning delivery and tool flow.
Do not describe this timestamp as socket receipt, HTTP TTFB or pure model prefill.

Reuse one existing upstream fix and preserve authorship. Verify its regression
on the runtime base, then run adjacent tests through the canonical hermetic
runner with real imports. Validate the exact delivered bytes and the opt-in
voice-metric hook. Do not change prompts, model/reasoning settings or WPE.

Deploy the focused runtime branch to Mira on snappy only. Establish an exact
service selector and known-good rollback before restart. Keep current candidate,
test results, deployment state and upstream coordination in the tracking issue.
