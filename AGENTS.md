# Agent Rules

## Core Rules

- Whenever altering existing files, apply the minimal change necessary to the task: the smallest correct change, following existing patterns, verified before handoff. Ask before making any extra changes.
- Keep changes scoped to the request and the affected package or module. Do not refactor unrelated code or make cosmetic formatting changes.
- Preserve user-authored changes and existing implementation choices. Before editing, inspect the current saved file and treat it as authoritative. Do not revert, replace, restore, or reintroduce code the user has removed unless explicitly asked; if the requested change depends on that code, ask first.
- If a requested behavior is impractical or blocked by a technical limitation, say so before implementing it and present the available choices. Do not paper over the limitation with an unrequested workaround or claim the original request was implemented unchanged.

## Data Access

- When querying data, always use the Snowflake MCP server directly (`mcp__snowflake__run_snowflake_query`). Do not explore the local repository or filesystem for data unless explicitly asked.

### Snowflake Constraints

- Use `INSERT` or `MERGE` instead of `UPDATE` statements because `UPDATE` is blocked.
- Always qualify table names with their database and schema.
- Do not attempt OAuth configuration changes without asking first.
- When asked to construct or write a query, only print it. Execute a query only when explicitly asked to run or execute it.

## Voice and Response Style

- Be direct. State the point first, then support it. Use specific examples and names rather than vague claims.
- Prioritize correctness over agreement. State uncertainty instead of guessing, and push back on technically unsound requests with evidence.
- Be concise. Remove filler and do not use preambles such as "Great question!" or performative enthusiasm such as "exciting," "incredible," or "powerful" unless genuinely warranted.
- Keep plans proportional to the task. For small changes, state the approach, affected areas, and key verification without expanding into an exhaustive implementation checklist.
- Do not use sycophantic responses such as "You're absolutely right!" Respond directly.
- Do not use stock chatbot phrases such as "I hope this helps!", "Let me know if...", "Of course!", "Certainly!", or "Found the smoking gun!"
- Do not announce honesty or fairness with phrases such as "to be honest", "honestly", "the honest answer", "full transparency", "to be fair", or "I'll be candid". Honesty is the baseline.
- Do not pad the end of a response with unsolicited offers of further work. Do not save an important point for a closing "one more thing"; put it where it belongs.
- Use US English spelling, such as "behavior" rather than "behaviour".
- Do not over-format. Limit headers and bullet points to where they are genuinely useful.
- Never use highlighting as emphasis. Bold is acceptable when needed. In agent replies, avoid inline-code spans because they render with a highlighted background in the user's terminal. Write commands, identifiers, settings keys, and filenames as plain text, and use Markdown links for file paths. This restriction applies to replies, not to Markdown or code written into repository files.
- Before sending a response, scan the draft for these undesirable response patterns. Treat them as preferences, not defects to hunt mechanically.

## Problem Solving

- When proposing ML architectures or analytical approaches, validate that the approach handles the problem's specific constraints, including variable-length outputs and data characteristics, before presenting it. Do not propose generic solutions without checking applicability.

## Third-Party APIs and Libraries

- When working with volatile or less-documented libraries such as Genesis, do not trust web-search snippets blindly for API signatures.
- Before writing code, inspect the installed library directly or search its source to verify parameter names such as `rho` versus `density` and `key_action` versus `action`. For Python, use introspection such as `python -c "import ...; import inspect; print(inspect.signature(...))"`, requesting the required sandbox access if necessary.

## Tech Stack

- The primary language is Python.
- For analysis code, use pandas for data manipulation and matplotlib or seaborn for visualization unless instructed otherwise.
- Before running Python from the command line, activate the main Python environment with `pyenv activate ai`.

## Python Coding Standards

- Use Python 3 only. Never import from `__future__`.
- Use `time.monotonic()` for durations and `time.time()` only for wall-clock timestamps.
- Put imports at the top of the file. Valid exceptions are circular imports, lazy loading for worker isolation, `TYPE_CHECKING` blocks, and a heavy module needed only on some paths through a startup-sensitive entry point.
- Add type hints where reasonable and not overly clunky. Typing is a helper rather than a goal; be pragmatic about strict typing versus a targeted `cast` or `ignore`.
- Avoid `**kwargs` unless genuinely necessary.
- Give public functions, classes, and modules docstrings, subject to the maturity and obviousness guidance below.

## Code Comments and Docstrings

These rules apply primarily to maturing code. Keep comments and docstrings light in early prototyping or exploratory code, and always omit what is genuinely obvious.

### What Earns a Comment

- A comment should say something the code cannot: why an approach is used, what breaks otherwise, or which bug it guards against. Comment non-obvious constraints, surprising behavior, and complex data manipulation such as multidimensional tensor reorganization. Do not restate obvious assignments or control flow.
- Preserve non-obvious invariants, constants that should not be tuned blindly, and diagrams.
- Keep comments shorter than the code they explain. Narrative spanning files belongs in documentation such as `README.md`, `SPEC.md`, or a docs directory, with a short pointer from the code.
- Never add internal thinking or reasoning-process annotations to code comments.

### Scope of Comment Changes

- These rules apply to comments an agent writes in new code and on lines it is already changing.
- Do not sweep existing comments while fixing a bug or refactoring. Leave an existing comment alone unless the change makes it inaccurate, rewrites the lines it describes, or the user requested comment cleanup.

### Audience and Purpose

- Write for a human reading the code for the first time. Explain what the code actually does, not how its author already thinks about it.
- Explain why when the reason is non-obvious, such as using a running total so pairwise distance requires one subtraction rather than a per-pair scan.
- Be concise but complete. Do not narrate the edit that produced the code or data with phrases such as "consistent with X below", "now correctly matches Y", or "fixed to satisfy Z". State the underlying fact or invariant instead.
- Do not narrate an agent's actions in comments.

### Jargon

- Do not use unexplained internal jargon such as "warp" or "core set". Describe the actual operation, such as "re-express each time as non-meal minutes since midnight" or "use the weekday or weekend meal times".
- Prefer human terms: `t` is a time, not "the hour feature"; say "shorter way around the 24-hour clock", not "circular arc".

### Define Every Symbol

- Define every variable named in an explanation inline where it is used; do not leave symbols such as `w_k`, `s_k`, or `D_k` undefined.
- For a formula, list each symbol before the expression.
- For dense closed-form or vectorized math, explain the derivation. For example, explain why `w @ M @ w - sum(w_i^2)` over `(sum(w_i))^2 - sum(w_i^2)` is the off-diagonal weighted mean and what each `einsum` subscript does.

### Multidimensional Arrays

- State what each dimension means. Use named-dimension shape notation, such as `(B, O)` for customers in a batch by orders per customer.
- Define the dimension vocabulary once in a shape legend at the top of the module or section, then reuse those letters.
- State concrete shapes such as `(O,)`, `(B, O)`, and `(B, O, O)`, not vague descriptions such as "optional batch axis" or a bare ellipsis.
- For a matrix, explain what an element means, such as "entry `[..., i, j]` is the distance between order `i` and order `j`; the diagonal is zero."

### Docstring Format

- Use `@param name:` and `@return:` tags, except when the parameter or return value is obvious or the code is early-stage prototyping or exploration.
- Each `@param` should give the type and, where relevant, shape and units, followed by the meaning.
- `@return` should give the type, structure such as dictionary keys, DataFrame columns, or tuple contents, the meaning of each part, and relevant NaN or edge cases.
- Put a blank line before `@return`.

## Naming and Syntax Clarity

- Prefer clear, semantic names that communicate an object's role and distinguish related concepts. For example, use `TrajectoryRecorder`, `mobile`, and `mobile_config` rather than ambiguous names such as `data` or `obj` when the distinction matters.
- Avoid one- or two-letter names except where a variable is used once immediately after its definition.
- Use names faithful to the values they hold. Distinguish sizes, counts, and limits from the data itself, such as `required_history_size` rather than `required_history`.
- Name functions and methods with action verbs such as `get_`, `extract_`, `find_`, `compute_`, or `build_`. Avoid noun-only method names such as `_base_names`, which read as attributes. Predicates beginning with `is_` or `has_` are exceptions.
- Use plural names to distinguish arrays from scalars, such as `block_starts` and `order_counts`.
- Distinguish variables that index different things, such as `batch_customer_positions` for per-customer arrays and `order_row_indices` for per-order arrays, and explain the distinction in a comment when it is not obvious.
- Prefer the clearer NumPy idiom `np.expand_dims(a, -1)` over `a[..., :, None]`.

## Code Structure and Configuration

- Encapsulate behavior according to its domain meaning. For example, keep `World` focused on simulation orchestration and core world state; put trajectory capture, ghost lifecycle, capture indexing, and image export in a dedicated `TrajectoryRecorder`.
- Prefer explicit delegation over growing a central object with unrelated responsibilities. A domain service may be attached to `World` for access, but it should own its own state and behavior.
- When moving a function into a class as a method, put the implementation in that method rather than leaving a one-use forwarding wrapper, unless the separate function has an independent use.
- When a refactor definitively makes code obsolete, remove it as part of the change, including stale validation, special cases, helpers, imports, and comments.
- Represent scene components uniformly through the configuration model. Include a scene component as an ordinary entry in the appropriate list with the same representation as its peers; avoid special top-level configuration sections or entity paths unless the semantics genuinely require them.
- Favor data-driven configuration over hidden behavior. Make physical and visual properties such as friction, rolling friction, textures, and object poses visible and adjustable in scene configuration.
- Keep validation and error handling proportionate to realistic failure modes. Avoid defensive checks for invalid states that the application contract already rules out.

## Testing

- Test the actual implementation. Do not duplicate the implementation's logic in the test; a test that recomputes the expected value in the same way can pass when both are wrong.

## Engineering Practices

### Memory Vigilance in Maturing Code

- Watch for needless copies of known-large data, including whole-DataFrame copies, `sort_values`, and eager read paths.
- Pandas view-versus-copy behavior is often non-obvious. When memory matters, restructure to avoid the copy, such as using `argsort` plus indexing instead of `sort_values`, rather than guessing.
- Do not mutate a caller's large object to save memory; find a non-mutating approach.

### Temporary Reference Implementations

- When writing a fast or hard-to-read path, temporarily keep the slow, obvious implementation beside it as an oracle while verifying equivalence.
- After verification, remove the slow implementation and all oracle-comparison code and tests. Do not leave both implementations or their comparison scaffold in the codebase.

### Verify Refactors and Optimizations Empirically

- Do not rely on reasoning alone to conclude behavior is unchanged. Run the changed path against a reference or before-and-after implementation on representative and edge-case inputs with a tight tolerance.
- Check NaN, empty, and boundary patterns as well as the happy path.

### Check for Dangling References

- After renames or refactors, search for stale names, including undefined variables introduced by a rename and old function names remaining in docstrings, Markdown, or call sites.

### Surface Extra Changes

- When a task's natural consequences extend beyond what was requested, such as dropping a newly dead parameter, merging helpers, or fixing an adjacent bug, call them out explicitly rather than folding them in silently.
