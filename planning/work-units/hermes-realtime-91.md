# Hermes Node launcher lifecycle scanning

Owner: https://github.com/kortexa-ai/hermes-realtime/issues/91

Rebase the Kortexa runtime onto upstream while preserving session-owned toolsets,
profile session scope, cancellation cleanup, and speech behavior. Upstream moved
background turns and secondary reconnects into mixins; retain Kortexa behavior at
their new production entry points.

Recognize Node interpreter shebangs before recursively walking source as shell.
Explicit shell invocation always owns execution semantics. Keep literal lifecycle
checks, bounded reads, and scan budgets. A prior interpreter-only visit must not
suppress a later shell scan of the same file. Unknown interpreters retain the
existing walk. Do not add a product-specific exception or another upstream PR;
the upstream design discussion remains in NousResearch/hermes-agent#105758.

Validate synthetic local and remote launchers, shell wrappers, revisit order, and
the real Factory launcher through scanner-only diagnostics. Check the affected
Kortexa contracts before activating the runtime branch. Retain recovery refs and
a bundle, deliver with an exact remote lease, then verify runtime revision and
both served profiles. Keep execution receipts in the owning issue.
