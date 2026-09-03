# Agent Rules

## Code Comments

- Do not add thinking, reasoning, or rationale annotations into code comments.
- Comments should describe **what** the code does or **why** a non-obvious decision was made — not the agent's internal reasoning process.
- Do not narrate your own actions in comments (e.g., "Here we capture X from the enclosing scope to avoid a global").

## Third-Party APIs & Libraries

- When working with volatile or less-documented libraries (like Genesis), **do not blindly trust web search snippets for API signatures**.
- Proactively use `python -c "import ...; import inspect; print(inspect.signature(...))"` (via `run_command` with `BypassSandbox: true` if needed) or `grep_search` to verify parameter names (e.g., `rho` vs `density`, `key_action` vs `action`) *before* writing code.
