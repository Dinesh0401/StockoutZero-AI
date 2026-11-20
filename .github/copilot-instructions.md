# StockoutZero AI — Copilot repository instructions

These instructions apply to all code, tests, and documentation in this repository. Keep the scientific baseline behavior reproducible and do not make unverified claims of research novelty.

## Engineering principles

- Prefer clear, small, cohesive modules with explicit responsibilities over clever abstractions.
- Follow SOLID where helpful, but do not invent unnecessary layers or patterns.
- Isolate **pure statistical/decision logic** from HTTP handlers, persistence, external services, and UI code.
- Keep FastAPI route handlers thin: request validation, service invocation, and response mapping. Never embed model fitting directly in a route.
- Name functions by intent; make invalid or unidentifiable states explicit rather than silently substituting zero forecasts.
- Never represent missing stock as confirmed zero stock. Preserve observed sales versus latent demand distinctions.
- Keep backwards compatibility unless a breaking change is documented and tested. Do not rewrite working parts without a reason.
- No hardcoded secrets, user data, or personally identifiable store data in tests, examples, logs, or Git commits.
- Use meaningful commits, and present any major refactor as a separate pull request.

## Python conventions

- Target modern Python (3.11+ when configuring the project); use type hints on all new or modified public functions, methods, and return values.
- Prefer `str | None`, `list[Observation]`, `TypedDict`, `Enum`, and frozen dataclasses for appropriate domain types. Use Pydantic at API boundaries; don't make everything a Pydantic model.
- Write **Google-style Python docstrings**, readable with `python -m pydoc`, for public modules, classes, and nontrivial functions. Include summary, `Args`, `Returns`, `Raises`, and `Notes` as relevant, especially statistical assumptions and units.
- Private functions should be documented when their behavior is non-obvious; do not add meaningless one-line boilerplate to obvious helpers.
- Avoid broad `except Exception`, mutable default arguments, hidden I/O, and magic constants. Make random seeds explicit in research.
- Prefer stable tested library implementations for probability tails and optimization to ad hoc numerical routines.
- Use `ruff check` and `ruff format`; add type checking with `mypy` or `pyright` incrementally and keep relevant Pytest tests green.

## Frontend conventions

- The **current dashboard is plain JavaScript in `backend/app/static/index.html`**, not TypeScript. For changed nontrivial JavaScript functions use **JSDoc** (`@param`, `@returns`, `@throws`) rather than pretending TSDoc already applies.
- If/when a React + TypeScript frontend is created: enable `strict` in `tsconfig.json`, use **TSDoc** (`/** ... */` plus `@param`, `@returns`, `@remarks`, `@example`) for exported functions, services, hooks and important types.
- Do not use `any` without an explicit justification. Use narrow domain types, discriminated unions, runtime validation for API responses, and accessible UI components.
- Add ESLint and Prettier to a TypeScript package when it exists; do not add an unused Node toolchain to the present Python-only starter.
- Prefer `textContent` to interpolating untrusted values into `innerHTML`, and handle API errors accessibly.

## Research-specific guardrails

- State the observation model and its assumptions explicitly (e.g., the distinction between observed sales and censored latent demand).
- Distinguish estimated demand distribution quantiles from uncertainty intervals for learned parameters.
- Treat non-identifiable or insufficient-data conditions as first-class outputs that block unsafe auto-reorder advice.
- Benchmarks must use deterministic seeds, leakage-free temporal splits, consistent evaluation samples, fair baselines, metrics with units, and confidence intervals.
- Keep experiments and reference outputs separate from the production API; never claim synthetic results establish real-world performance.
- Add regression tests for new behavior, including invalid inputs, zero availability, all-censored histories, numerical extremes, and missing data.
- Preserve provenance and human approval for any inventory recommendation.

## When making changes

1. Read the neighboring source, tests, and `docs/RESEARCH_PROTOCOL.md`.
2. Describe the smallest useful refactor; do not reorganize the repository speculatively.
3. Implement with type hints, docstrings, tests, and minimal coupling.
4. Run the existing test suite and relevant lint/type commands if configured; report exactly what ran.
5. Update public API and research docs for changed semantics.
6. Explain any compatibility breaks, remaining limitations, and measured benchmark changes.

See `docs/CODE_STYLE.md` for examples and module-boundary guidance.
