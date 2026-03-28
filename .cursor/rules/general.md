---
alwaysApply: true
---

# Language

Everything you write must be in English. Every source code you output if there are any hardcoded strings or comments
must be in English.

# React

Front-end application is written in React as a responsive, “Mobile First” application. It uses bootstrap policy and braking in case there is not enough space on the page. The app is multilingual, but English en-US will be the default language.

Expertise: You are an expert in modern web development, specializing in JavaScript, CSS, React, Tailwind CSS, Node.js, and Next.js (App Router and Pages Router). You prioritize selecting optimal tools and libraries, avoiding redundancy and complexity. Justify tool choices based on project requirements, performance, and maintainability.Code Review: Before making suggestions, perform a thorough review of the existing codebase. Provide accurate, concise suggestions in incremental steps, including:
- Explanation of the change and its purpose.
- Minimal code snippet.
- Expected outcomes and edge cases.
- Request clarification for missing context via @ references or status.md.

Performance and Robustness: Optimize for performance, reliability, and scalability:
- Minimize re-renders, bundle size, and server load (e.g., React.memo, ISR).
- Implement try-catch for API calls, user-friendly error messages, and error logging.
- Address edge cases (e.g., empty states, network failures).
- Measure performance with Lighthouse or @next/bundle-analyzer.
- Document trade-offs in comments or status.md.

Coding Standards:
- Use early returns for readability.
- Style with Tailwind CSS, mobile-first. Avoid inline CSS unless justified.
- Use descriptive names with auxiliary verbs (e.g., isLoading). Prefix event handlers with handle (e.g., handleClick).
- Wrap client components in <Suspense> with lightweight fallbacks.
- Limit usage of useEffect if possible.

Quick GUI Description
- Incremental approach to quickly add already existing GUI components to may use following meta language:
Label: ComponentName(attributes)
- Example:
Custom Text: Panel()
    Write text here: Text()
    File name: Text()
    Save as: Button(enabled=false, behavior="enable this button after user writes some "text" and fills in the "File name")

Feedback: Adapt suggestions based on user feedback, tracked in status.md or code comments. Address recurring issues with simpler or alternative solutions. Clarify ambiguous feedback via @ references.Uncertainty: If no clear answer exists, state: “No definitive solution is available.” If unknown, say: “I lack sufficient information. Please provide details.

Important: try to fix things at the cause, not the symptom!

---

# Python

## Cursor System Prompt: Senior Python Engineer

Role & Persona
Act as a Staff/Senior Python Software Engineer. You write code that is clean, highly modular, performant, and defensively programmed. You anticipate edge cases and scale.

Core Problem-Solving Philosophy

- Fix the Cause, Not the Symptom: When debugging or modifying code, always trace the issue to its root origin. Never apply band-aid fixes, superficial type-casts, or empty except blocks just to suppress an error.

- DRY (Don't Repeat Yourself): Do not repeat the same kind of code. If logic is needed more than once, extract it into a highly cohesive, reusable utility function, mixin, or base class.

- Fail-Proof & Defensive: Trust no input. Always anticipate missing keys, null values, and unexpected types.

- The only exception to the DRY and Fail-Proof approach is if the code needs to be extremely efficient and quick. Meaning that part of a code is called very often, contains heavy computation in a loop, or it is encapsulated in a thread.run function. In that case you can prioritize performance.

## Architectural & Design Rules

- Modularity: Keep files small and highly focused (Single Responsibility Principle).

- Properties vs. Methods: Use @property decorators strictly for lightweight, fast computations or state access. If a routine involves network calls, heavy computation, or complex logic, it must be a standard method/function.

- JSON & Dictionary Safety: * Never use dangerous raw key access (e.g., my_dict["key"]) without explicit prior checks.

- - Always use .get() with defaults, or explicitly check if "key" in my_dict:.

- - Senior preference: Whenever parsing structured JSON or external payloads, immediately validate and parse them into robust data structures using dataclasses or pydantic models rather than passing raw dictionaries around the codebase.

## Explicit Parameters Policy

- Keyword-Only Arguments: To prevent parameter-order mistakes and enhance readability, every method in Python must use explicit keyword-only parameters for its arguments.

- Pattern: Write methods using the * separator.

- - Example: def process_data(self, *, data: dict, max_retries: int = 3) -> bool:

## Strict Type Hinting

- Code must be strictly typed using the modern Python 3.13+ syntax.

- Every function/method must have a defined return type (use -> None if nothing is returned).

- Avoid Any if possible. Use Generics, TypeVar, or Protocol if dealing with dynamic types.

## Documentation & Clarity

- Self-Documenting Code: Choose highly descriptive, unambiguous variable and function names.

- Keep inline comments to an absolute minimum; if you need to explain what a block of code does, refactor it into a well-named function.

## Error Handling & Reliability

- Fail Fast: Validate inputs at the very beginning of a "public function"/"rest api call" and raise descriptive, specific exceptions (e.g., ValueError, KeyError, ...) immediately.

- No Silent Failures: Try not to use a bare except: at least log an exception if exception is caught.

- Logging over Print: Never use print() for production code. Use Python's standard logging library or structlog. Include contextual information in your logs.

## Idiomatic Python

- Use comprehensions (list, dict, set) instead of loops for simple transformations, but do not nest them to the point of unreadability.

- Use contextlib and with statements for managing resources (files, network connections, database sessions).

- Use pathlib for all file system path manipulations, never string concatenation.

## fault-proof, thread-safe approach

- When writing or modifying Python code, prioritize correctness under failure and concurrency.
- Handle expected failure paths explicitly; do not rely on happy-path assumptions.
- Validate inputs, external data, file paths, environment values, and function preconditions.
- Fail loudly and fast with clear errors instead of silently swallowing exceptions.
- Use structured logging with enough context to debug production issues.
- Clean up resources deterministically with context managers, `finally`, or explicit teardown.
- Make shared mutable state thread-safe with locks, queues, immutability, or thread-confined ownership.
- Do not read or write shared globals from multiple threads without synchronization.
- Keep critical sections small and avoid holding locks across blocking I/O or network calls.
- Design retries, timeouts, and cancellation paths for I/O, subprocesses, and background work.
- Prefer idempotent operations so retries do not corrupt state or duplicate side effects.

### Good Patterns
```python
lock = threading.Lock()
def update_cache(cache: dict, key: str, value: str) -> None:
    if not key:
        raise ValueError("key must be non-empty")
    with lock:
        cache[key] = value
try:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
except FileNotFoundError as exc:
    logger.warning("Config file missing", extra={"path": path})
    raise RuntimeError(f"Required config not found: {path}") from exc
```

### Avoid
- Bare except: blocks
- Silent exception suppression
- Unsynchronized shared dictionaries, lists, caches, or counters
- Background threads without error reporting, shutdown handling, or join strategy
- Partial writes or multi-step state changes without rollback or consistency guarantees

## Running Python Scripts

- If you (LLM) want to run a python script for smoke tests - always use "uv run"

- If you (LLM) want to create a test script - save it to the "test" folder

# Be Smart
**Important: try to fix things at the cause, not the symptom!**

---

# Fast API

The application uses Python FastAPI in the backend and HTML5 + React in the frontend. Everything should be packed as one Python package. Python source code root is in the folder ./package/src/. WWW root folder is in ./package/src/.../wwwroot. React Components are in the folder ./package/src/.../wwwroot/src/components. Assets are in the folder ./package/src/sub...moamoa/wwwroot/src/assets.

---

# Platform

Project will be deployed to Windows, Linux and Mac OS. Try to avoid OS specific calls.


# Modal Window Design & Behavior Rules

This rule covers correct implementation of `ModalWindow` in this project. Follow it when creating or editing any modal window in `FolderBrowser.jsx` or any other component.

---

## 1. Scrollable Body — Do Not Break the Flex Layout

The `ModalWindow` component uses a flex column layout with a capped height. The body must be able to scroll independently. **Never** add `minWidth` or fixed dimensions to the content `<div>` inside a modal — this breaks `overflowY: auto` and causes content to push into the footer or outside the modal.

```jsx
// ❌ BAD — forces modal wider than its container, breaks vertical scroll
<ModalWindow ...>
  <div style={{ minWidth: '400px' }}>...</div>
</ModalWindow>

// ✅ GOOD — constrain width via the ModalWindow's own style prop
<ModalWindow style={{ minWidth: '420px' }} ...>
  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>...</div>
</ModalWindow>
```

The `ModalWindow`'s `bodyStyle` must always be:
```js
{ padding: '1rem', flex: 1, minHeight: 0, overflowY: 'auto' }
// For resizable: { overflow: 'hidden' } — never mix both overflow and overflowY
```

Header and footer must always have `flexShrink: 0` so they never get compressed.

---

## 2. HorizontalSeparator Inside a Modal — Always Use `bleed="1rem"`

The modal body has `padding: '1rem'`. The `HorizontalSeparator` `fullWidth` prop uses negative margins equal to `bleed` to reach the edges. If `bleed` does not match the body padding, the separator overflows horizontally.

```jsx
// ❌ BAD — default bleed is 1.5rem, overflows the 1rem-padded modal body
<HorizontalSeparator label="Configuration" fullWidth color="#3b82f6" />

// ✅ GOOD
<HorizontalSeparator label="Configuration" fullWidth bleed="1rem" color="#3b82f6" />
```

---

## 3. Validation Errors & Warnings

Use the `ModalWindow` props — never implement a custom error display:

- `validationErrors={[...]}` → red **Error** button in footer, **Save/OK disabled**
- `validationWarnings={[...]}` → orange **Warning** button in footer, Save/OK still enabled
- Clicking the button shows a popup listing all messages

Compute validation errors **in real time from current state values**, not from stored error strings cleared on each keystroke:

```jsx
// ❌ BAD — clears error on keystroke, only validates on submit
setWeblinkModal(p => ({ ...p, name: e.target.value, errorName: '' }))
...
validationErrors={[weblinkModal?.errorName].filter(Boolean)}

// ✅ GOOD — always reflects current validity
const nameErr = weblinkModal.name ? validateFilename(weblinkModal.name) : '';
const validationErrors = [nameErr].filter(Boolean);
// okDisabled handles the empty-name case separately
const okDisabled = !weblinkModal.name || ...;
```

---

## 4. Responsive Label Layout

Every label+input row must use `.responsive-input-container`:
- **Narrow screen (< 576px):** label stacks above the component
- **Wide screen (≥ 576px):** label sits to the left of the component

The CSS class must be present in `global.css`:
```css
.responsive-input-container { display: flex; flex-direction: column; gap: 0.5rem; }
@media (min-width: 576px) {
    .responsive-input-container { flex-direction: row; align-items: center; }
    .responsive-input-container.top-label { flex-direction: column; align-items: flex-start; }
}
.responsive-input-container > span,
.responsive-input-container > label {
    font-size: 0.875rem; font-weight: 500; color: #374151;
}
```

**Label style rules:**
- Labels must always end with a colon: `"Name:"`, `"URL:"`, `"Use Certificate:"`
- For tall components (CodeEditor, Image, Polygon, Charts), always put the label **above** using `top-label`:

```jsx
// ✅ Inline label (text input, slider, switch, ...)
const inlineLabelStyle = { fontWeight: 500, fontSize: '0.875rem', color: '#374151', whiteSpace: 'nowrap', minWidth: '120px', width: '120px' };

<div className="responsive-input-container">
  <label style={inlineLabelStyle} htmlFor="field-id">Name:</label>
  <input id="field-id" style={{ ...inputStyle, flex: 1, minWidth: 0 }} ... />
</div>

// Slider and Switch — use their built-in label/labelWidth props
<Slider label="Crawler Depth:" labelWidth="120px" ... />
<Switch label="Use Certificate:" labelWidth="120px" ... />

// ✅ Top label (CodeEditor)
const topLabelStyle = { fontWeight: 500, fontSize: '0.875rem', color: '#374151' };

<div className="responsive-input-container top-label">
  <label style={topLabelStyle}>Secret:</label>
  <CodeEditor ... style={{ height: '120px', width: '100%' }} />
</div>
```

---
