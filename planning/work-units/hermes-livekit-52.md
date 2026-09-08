# Gateway cancellation cleanup

Tracking: https://github.com/kortexa-ai/hermes-livekit/issues/52

This core-latency unit uses the approved cross-repository tracking exception.
Keep the upstream contribution separate from the kortexa-ai/main runtime branch
and keep this fork-only note out of the upstream change.

When a gateway turn is cancelled, cancel its text consumer before entering the
normal flush cleanup path. Distinguish cancellation of the outer turn from
cancellation of the child consumer. Normal completion retains its flush budget;
cancelled turns must not return an ordinary completed reply.

Preserve TTS abort/drain, generation-owned session release and background-task
settlement, including cancellation during the flush itself. Exercise the real
turn orchestrator with event-controlled worker/transport boundaries and run
adjacent streaming, interruption, generation and delivery contract tests through
the canonical hermetic runner. Do not change the model, prompts or WPE build.

Contribute a focused generic fix upstream, preserve its separate branch, then
backport to the runtime branch and deploy only Mira with a preflighted rollback.
Keep exact revisions, review evidence, service state and live qualification in
the issue/project rather than this note.
