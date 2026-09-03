# Agent Rules

## Code Comments

- Do not add thinking, reasoning, or rationale annotations into code comments.
- Comments should describe **what** the code does or **why** a non-obvious decision was made — not the agent's internal reasoning process.
- Do not narrate your own actions in comments (e.g., "Here we capture X from the enclosing scope to avoid a global").

## Third-Party APIs & Libraries

- When working with volatile or less-documented libraries (like Genesis), **do not blindly trust web search snippets for API signatures**.
- Proactively use `python -c "import ...; import inspect; print(inspect.signature(...))"` (via `run_command` with `BypassSandbox: true` if needed) or `grep_search` to verify parameter names (e.g., `rho` vs `density`, `key_action` vs `action`) *before* writing code.

## User Code-Structure Preferences

- Prefer clear, semantic names that communicate an object's role and distinguish related concepts. For example, use names such as `TrajectoryRecorder`, `mobile`, and `mobile_config` instead of ambiguous names like `data` or `obj` when the distinction matters.
- Encapsulate behavior according to its domain meaning. Keep `World` focused on simulation orchestration and core world state; put trajectory capture, ghost lifecycle, capture indexing, and image export in a dedicated `TrajectoryRecorder`.
- Prefer explicit delegation over growing a central object with unrelated responsibilities. A domain service may be attached to `World` for access, but it should own its own state and behavior.
- Represent scene components uniformly through the configuration model. If something is a scene component, include it as an ordinary entry in the appropriate list with the same representation as its peers; avoid special top-level config sections or special entity paths unless the semantics genuinely require them.
- Favor data-driven configuration over hidden behavior. Physical and visual properties such as friction, rolling friction, textures, and object poses should be visible and adjustable in the scene config.
- When runtime objects and their configuration records are both needed, pass and name them separately rather than using a configuration record where a Genesis entity is required.
