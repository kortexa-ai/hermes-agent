# Hermes upstream reconciliation

Owner: https://github.com/kortexa-ai/hermes-livekit/issues/79
The Hermes fork has Issues disabled; this follows the existing cross-repository
runtime tracking arrangement.

`kortexa-ai/main` is the working runtime branch and must descend from both
`upstream/main` and `origin/main`. Keep upstream PR branches separate, rebase
those branches individually, then bring their reviewed changes into the runtime
branch. Preserve unpublished fixes and check semantics when a PR was closed as
superseded: closure alone does not prove its entire patch landed.

The reconciliation against `dce1e9b37581dd62e480a9064dc04a709c2940d3` retains:

- PR #86233's regression for plugin toolset discovery. Its production fix is
  already supplied by upstream `361dc5cf69fb9b2677e8b2c1e1ac95a92e40cfd1`.
- PR #105549's cancellation propagation and prompt text-stream settlement.
- PR #110527's profile/session-owned toolsets, with current registry, abandoned
  loader and gateway selection boundaries.
- The unpublished speech completion budget and sink-owned audible/fallback
  state, plus the independent first-sentence threshold and later batching.
  Upstream #115846 retained shared `min_len` from superseded #96933 but omitted
  the independent `first_sentence_min_chars` override.

Provider metadata warm-up, environment-probe warm-up, speech tool boundaries,
and Responses first-event timing are already present upstream. Keep their
historical contract notes without introducing duplicate implementations.

Before rewriting a remote branch, retain the original refs and a Git bundle,
validate its focused behavior and code-health checks, and use an exact
`--force-with-lease`. Keep a clean isolated worktree until the combined runtime
passes. Activate the completed runtime branch on Snappy and verify both existing
Hermes profiles through their normal gateway lifecycle; profile consolidation
is outside this reconciliation. Record branch heads, validation, deployment,
and rollback evidence in the issue.
