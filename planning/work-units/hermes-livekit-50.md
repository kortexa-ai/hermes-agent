# First-sentence streaming TTS and the Kortexa runtime branch

Tracking: [hermes-livekit#50](https://github.com/kortexa-ai/hermes-livekit/issues/50).
Franci approved cross-repository tracking because Issues are disabled on this fork.
Upstream proposal: [hermes-agent#105235](https://github.com/NousResearch/hermes-agent/issues/105235).

## Branch and runtime contract

- `kortexa-ai/main` is the Kortexa runtime branch. The production checkout on snappy
  must track `origin/kortexa-ai/main`; routine updates must retain this branch.
- Keep upstream contributions on separate `kortexa-ai/<topic>` branches based on
  upstream main. Cherry-pick the required changes onto `kortexa-ai/main`.
- Preserve the existing streaming completion/fallback ownership fix when integrating
  upstream. Do not reset the runtime branch to upstream or drop that fix.
- The first-sentence PR must not include this deployment note or unrelated fork changes.

## Behavior and configuration

`tts.streaming.first_sentence_min_chars` is a positive integer, default 20. Mira uses
1 so a complete short opener can reach synthesis before the next sentence arrives.
Only the first nonempty emission uses the override; later sentences still use 20.
The gateway, CLI speaker, and speak-stream WebSocket share the same parser.

Apply profile settings through `hermes --profile mira config set`, not by rewriting
profile files. Verify streaming completion and non-silent received RTP with an isolated
silent call. Received RTP is not proof of physical speaker onset quality.

No custom WPE rebuild, GPU-service change, camera upload, microphone activation, or
physical playback is required. Deployment receipts and current status belong in the
tracking issue, not this source document.
