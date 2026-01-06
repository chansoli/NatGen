You are Copilot Agent designing a code-mutation task that injects a small snippet into an existing function without altering the original functionality. This task will ship as part of the SemanticPreservingTransformation used to create NatGen variants, and must plug in as an extension to the existing NatGen architecture.

INTEGRATION
- Register this transformation in the SemanticPreservingTransformation registry with a clear module name/config key, defaulting to enabled with an explicit probability knob read/validated from the NatGen config (fail fast on missing/invalid values rather than silently skipping).
- Follow existing NatGen config conventions for toggles and probabilities to avoid divergence from other transformations.

GOAL
- Insert a 3-4 line snippet into a random location within an existing function so that behavior stays unchanged.
- The snippet should: (1) add a benign function call (no side effects), (2) perform an inline refactor on the target function (e.g., inline a trivial temp var or move an expression inline), and (3) spread the inserted lines across the function body when feasible.

ASSUMPTIONS
- Target languages start with C/C++/Java/Python (same NatGen languages); adjust if we later extend.
- We control the injected call target; it can be a no-op helper we add alongside the function (or a standard library call proven side-effect-free like `std::ignore`/`(void)` casts/log stubs guarded off).
- Functions are non-empty and parsed via tree-sitter; we can pick safe insertion anchors (before returns, between statements) while preserving formatting.

INLINE FUNCTION SOURCE (INITIAL)
- Ultimately the inlined function will be randomly selected from a corpus; to start, use a single example while ensuring no name collisions on inline.
- Example seed function:
```cpp
int add(int a, int b)
{
	int sum = a;
	sum += b;
	return sum;
}
```

INLINE HELPER HYGIENE
- Enforce a renaming/namespace strategy to avoid collisions when the helper is inlined multiple times in the same translation unit (e.g., mangle helper name with a per-application suffix and keep it local to the enclosing scope where possible).
- Apply per-language scoping rules (C/C++: static/anonymous namespace or unique suffix; Java: local helper method with unique name; Python: nested def with unique name) to keep helper identifiers isolated.

DELIVERABLES
1) A transformation module that, given a parsed function, emits a modified function with the injected snippet meeting the three properties above.
2) Unit/property tests covering: no-behavior-change (syntactic check + optional differential test harness), random placement respects scope, and line-spreading logic when possible.
3) Config flags to enable/disable each sub-transformation (benign call, inline refactor, line spreading) for ablations, and a configurable probability of occurrence that honors the existing NatGen config file conventions.

GUARDRAILS
- Keep inserted call pure: no globals, no I/O, and compiled out if needed (e.g., inline `void noop(...) {}` or `(void)` expressions).
- Preserve original control flow; avoid inserting inside `for` headers or splitting multi-statement macros.
- Maintain ASCII output and original indentation style.

WORKPLAN
1) Define snippet schema: a benign call line plus required inline-refactor lines (e.g., inline temp assignment removal). Provide per-language templates.
2) Implement insertion point selection using AST: pick a random statement boundary inside the function body; avoid returns/throws as the first line after insertion.
3) Add line-spreading strategy: when the function has >=3 statements, interleave snippet lines across distinct safe boundaries; otherwise keep contiguous.
4) Implement inline-refactor helper: detect trivial temps (`auto t = expr; use(t);`) and replace with direct `use(expr);` while adding the benign call nearby.
5) Emit formatting-preserving edits (respect leading/trailing whitespace, braces) and ensure deterministic seeding for reproducibility.
6) Write tests per language with golden inputs/outputs plus a differential runner stub (compile/run optional) to ensure no semantic drift for safe cases.
7) Integrate as a selectable transformation in the NatGen pipeline (CLI flag + metadata reason codes for skips/failures).

OPEN QUESTIONS
- Which no-op call template per language is safest (e.g., `(void)sizeof(expr);`, `noop(expr);` with inline definition, or `assert(true);` if not stripped)?
- Should we allow insertion in single-line lambdas/expressions, or skip such functions entirely?
- How to balance randomness with reproducibility across datasets? Seed strategy?


# Example
Original
```cpp
int foo() {
	int x = 1;
	int y = 2;
	int z = x + y;
	return z;
}
```

Inline Function
```cpp
int inline_snippet_add_deadbeef(int a, int b)
{
    int sum = a;
    sum += b;
    return sum;
};
```

Original-inlined: step 1
```cpp
int foo() {
	int x = 1;
	int y = 2;
	int z = x + y;
	return z;
}
```

Original-inlined: step 2
```cpp
int foo() {
	int x = 1;
	int y = 2;
	int z = x + y;

    int a;
    int b;
    int result;

    int sum = a;
    sum += b;
    
    result = sum;

	return z;
}
```