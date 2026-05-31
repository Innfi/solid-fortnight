---
name: reference-counter
description: >
  Code architecture analysis: counts and aggregates all imports across the codebase,
  then renders ASCII bar-chart graphs of (1) file/module reference counts and
  (2) named-entity (function, class, type, variable) reference counts, both ordered
  descending. Use when user says "reference counter", "count imports", "analyze imports",
  "show import graph", "which files are most imported", or invokes /reference-counter.
---

Analyze all imports in the codebase. Produce two ASCII bar-chart graphs:
1. **File graph** — each imported module path, ranked by how many files import it
2. **Entity graph** — each imported named symbol, ranked by how many times it appears in import lists

## Step-by-step procedure

### 1. Collect raw imports

Use Grep with these patterns to collect every import statement. Exclude `node_modules`, `dist`, `.git`, `build`, `.next`, `out` paths (use glob filters to target source files only: `**/*.ts`, `**/*.tsx`, `**/*.js`, `**/*.jsx`, `**/*.mts`, `**/*.mjs`).

Patterns to grep (run all, then merge results):

| Pattern | Captures |
|---------|---------|
| `from\s+['"][^'"]+['"]` | ES module `import ... from '...'` |
| `require\(['"][^'"]+['"]\)` | CommonJS `require('...')` |
| `import\s+['"][^'"]+['"]` | Side-effect import `import '...'` |

For each match, record:
- The **source file** containing the import
- The full **import statement text** (use `-C 0` or content mode)

### 2. Extract module paths

From each import statement, extract the module path string (the quoted part after `from` or inside `require()`).

Normalize paths minimally:
- Strip leading `./` or `../` chains — keep them as-is for relative modules (do NOT resolve to absolute; keep the raw string as the key)
- Collapse duplicate slashes
- Strip trailing `/index` if present (treat `./utils/index` same as `./utils`)

### 3. Extract named entities

From each import statement, extract named imports using this logic:
- `import { A, B as C, type D }` → entities: `A`, `B` (use original name, not alias), `D`
- `import DefaultName from '...'` → entity: `DefaultName` (mark as default)
- `import * as Ns from '...'` → entity: `Ns` (mark as namespace)
- `require(...)` with destructuring `const { X, Y } = require(...)` → entities `X`, `Y` if visible in the same line

Count each entity occurrence independently (if two files both import `cn`, that's count 2).

### 4. Aggregate counts

Build two maps:
- `fileRefs: Map<modulePath, count>` — increment for each file that imports this module (count unique importing files, not total lines)
- `entityRefs: Map<entityName, count>` — increment per occurrence across all import statements

### 5. Render graphs

**Format:**

```
## File References  (top N, sorted by import count ↓)

<module-path>              ██████████████████████ 45
<module-path>              █████████████ 30
<module-path>              ████████ 18
...

## Entity References  (top N, sorted by import count ↓)

<entity-name>              ████████████████████ 32
<entity-name>              █████████████████ 28
<entity-name>              ████████████ 20
...
```

**Bar scaling:**
- Max bar width = 40 `█` characters
- Bar width for entry = `round(count / maxCount * 40)` (minimum 1 for any nonzero count)
- Left column width = longest module/entity name + 2 spaces (pad with spaces)
- Right of bar: ` <count>` (space then integer)

**Cutoff:**
- Show top 40 entries per graph (or all if fewer than 40)
- After the bar chart, print a one-line summary: `Total unique modules: N | Total unique entities: M | Files scanned: K`

**Grouping (optional but preferred):**
If more than 5 entries in File References come from the same npm package (e.g., `react`, `react/...`), group them under a `[react]` sub-header with a combined total shown separately.

## Output structure

Always output both graphs in a single response:

```
# Reference Counter Analysis

## File References  (N unique modules, K files scanned)
...bar chart...

## Entity References  (M unique entities)
...bar chart...

---
Total unique modules: N | Total unique entities: M | Files scanned: K
```

## Error handling

- If no imports found: report "No import statements found in scanned files."
- If a file can't be read: skip silently, count it in a "skipped" tally at the bottom.
- If codebase has mixed TS/JS: scan all; note the mix in the summary line.

## Scope

Always exclude from scanning:
- `**/node_modules/**`
- `**/dist/**`
- `**/build/**`
- `**/.next/**`
- `**/out/**`
- `**/.git/**`
- `**/*.min.js`
- `**/*.d.ts` (type declaration files — they declare, not import for use)

Include:
- `**/*.ts`, `**/*.tsx`, `**/*.mts`
- `**/*.js`, `**/*.jsx`, `**/*.mjs`, `**/*.cjs`

## Caveman mode compatibility

Graphs are data — render them verbatim regardless of active caveman intensity level. Only the surrounding prose (labels, summary line) should compress under caveman rules.
