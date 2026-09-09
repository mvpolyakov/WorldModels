# Agent Rules

## Code Comments

- Do not add thinking, reasoning, or rationale annotations into code comments.
- Comments should describe **what** the code does or **why** a non-obvious decision was made — not the agent's internal reasoning process.
- Do not narrate your own actions in comments (e.g., "Here we capture X from the enclosing scope to avoid a global").

## Third-Party APIs & Libraries

- When working with volatile or less-documented libraries (like Genesis), **do not blindly trust web search snippets for API signatures**.
- Proactively use `python -c "import ...; import inspect; print(inspect.signature(...))"` (via `run_command` with `BypassSandbox: true` if needed) or `grep_search` to verify parameter names (e.g., `rho` vs `density`, `key_action` vs `action`) *before* writing code.

## User Code-Structure Preferences

- Make only changes that pertain to the requested task. Do not reformat existing code for cosmetic reasons, such as changing one line to multiple lines or vice versa; be parsimonious with edits.
- Prefer clear, semantic names that communicate an object's role and distinguish related concepts. For example, use names such as `TrajectoryRecorder`, `mobile`, and `mobile_config` instead of ambiguous names like `data` or `obj` when the distinction matters.
- Encapsulate behavior according to its domain meaning. For example: keep `World` focused on simulation orchestration and core world state; put trajectory capture, ghost lifecycle, capture indexing, and image export in a dedicated `TrajectoryRecorder`.
- Prefer explicit delegation over growing a central object with unrelated responsibilities. A domain service may be attached to `World` for access, but it should own its own state and behavior.
- When moving a function into a class as a method, put the implementation in that method rather than leaving a one-use forwarding wrapper, unless the separate function has an independent use.
- When a refactor definitively makes code obsolete, remove it as part of the change, including stale validation, special cases, helpers, imports, and comments.
- Represent scene components uniformly through the configuration model. If something is a scene component, include it as an ordinary entry in the appropriate list with the same representation as its peers; avoid special top-level config sections or special entity paths unless the semantics genuinely require them.
- Favor data-driven configuration over hidden behavior. Physical and visual properties such as friction, rolling friction, textures, and object poses should be visible and adjustable in the scene config.

## User preferences

- Preserve user-authored changes and existing implementation choices. Before editing, inspect the current saved file and treat it as authoritative. Do not revert, replace, restore, or reintroduce code the user has removed, even if an earlier version or prior conversation contained it; ask first if the requested change depends on that code.
- Apply the smallest change that fully implements the requested task. Avoid unrelated refactors, rewrites, formatting changes, or behavioral changes to existing code.
- Keep validation and error handling proportionate to realistic failure modes; avoid defensive checks for invalid states that the application contract already rules out.
- Use precise variable names that describe what a value contains; distinguish sizes, counts, and limits from the data itself (for example, `required_history_size` rather than `required_history`).
