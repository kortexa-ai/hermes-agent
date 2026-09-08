# Gateway model metadata warm-up

Tracking: https://github.com/kortexa-ai/hermes-livekit/issues/55

This core-latency unit uses the approved cross-repository tracking exception.
Keep this fork-only note out of the upstream contribution.

Warm the configured route's metadata through the existing bounded startup gate,
without constructing an agent or freezing session prompts, tools or memory.
Preserve the profile and secret context across the executor boundary. Keep lazy
initialization and current context limits as the fallback.

Do not run inference probes at startup. Bedrock context discovery can send a
large synthetic prompt; mixture-of-agents can resolve to Bedrock. Leave these
routes lazy rather than adding startup inference or caching a guessed limit.

Test the real config/runtime/metadata path with temporary profile homes and a
fake HTTP transport. Check cache reuse, profile credentials, explicit context
pins, nonfatal failure and the existing startup deadline. Keep the upstream
branch separate, then backport to kortexa-ai/main and deploy Mira only after
preserving and validating the production branch's existing changes.

Exact revisions, measurements, live coordination and deployment evidence belong
in the issue/project. Do not change Mira's model or rebuild WPE.
