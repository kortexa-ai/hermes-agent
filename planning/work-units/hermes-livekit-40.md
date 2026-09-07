# Streaming audio completion and fallback

Tracking issue: https://github.com/kortexa-ai/hermes-livekit/issues/40

Franci approved cross-repository tracking because this fork's issue tracker is
disabled. These generic consumer fixes accompany the LiveKit plugin's PCM sink.

The sink owns the audible flag: incomplete PCM buffered by a resampler must
not suppress whole-file fallback. Cancellation releases output and preserves
partial-audio suppression. A long spoken answer must be allowed to drain after
text generation finishes, rather than being cut off at ten seconds.

`tts.streaming.completion_timeout` sets the maximum completion wait in seconds.
Its default is 120; finite numeric values from 1 to 600 are accepted. This
budget does not delay playback. Provider and sink failures can terminate the
stream sooner. Explicit short waits remain available for cancellation cleanup.

Validate through `scripts/run_tests.sh tests/gateway/test_streaming_tts_consumer.py`
and the companion plugin's real-consumer and paced long-reply regressions.
The provider, model, voice, and prompt construction are unchanged.
