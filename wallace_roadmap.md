# Wallace Roadmap

**Version:** 1.0  
**Last Updated:** 2026-10-08
**Active Track:** A — Core Intelligence
**Current Step:** 2 — Cross-Language Code Operations

---

## How to Read This File

An agent interacting with this roadmap should follow these steps in order:

1. **Locate the file** — Find `wallace_roadmap.md` in the project root.
2. **Parse the metadata** — Extract `Version`, `Last Updated`, and `Active Track` from the header.
3. **Determine active track** — Read the `Active Track` line to know which track section to operate in.
4. **Find current step** — Within the active track, locate the `Current Step` heading and read its description.
5. **Execute the step** — Follow the implementation details under the current step.
6. **Update after completion** — Once a step is done, update the `Current Step` field before moving to the next step in the same track.
7. **Switch tracks** — When a track is fully completed, move to the next track and set `Active Track` accordingly.

---

## Current Status

| Field | Value |
|------|------|
| Active Track | A — Core Intelligence |
| Current Step | 2 — Cross-Language Code Operations |
| Progress in Track | 1 / 3 |

---

## Full Roadmap

### Track A — Core Intelligence

#### Step 1 — Project Understanding
**Goal:** Deepen project/tool awareness beyond simple file reads.

**Status:** Complete — 2026-10-08

| Capability | Implementation ideas |
|------------|----------------------|
| Multi-file semantic understanding | Summarize cross-file relationships, imports, and data flows |
| Dependency graph extraction | Parse `pyproject.toml`, `requirements.txt`, etc. |
| Change impact analysis | Given a Git diff, predict affected modules/tests |
| Code summarization | Generate concise explanations of modules/functions |

**Implementation:** `dependency_analysis.py` now builds a Python import graph,
summarizes modules, compares dependency versions, and analyzes changed files.
Wallace exposes these capabilities through `summarize_project_file`,
`analyze_project_dependencies`, and `analyze_git_change_impact`.

**Milestone:** Completed with a read-only Git change-impact tool and project
dependency-analysis tools.

---

#### Step 2 — Cross-Language Code Operations
**Goal:** Move beyond Python-only operations.

| Capability | Implementation ideas |
|------------|----------------------|
| Language detection | Auto-detect language from file extensions/content |
| Syntax checking | Leverage `ruff`, `mypy`, `black` checks across languages |
| Translation | Convert Python snippets to JS/Go/etc. using LLMs |
| Refactoring suggestions | Propose safe refactorings across languages |

**Milestone:** Successfully translate a Python snippet to JavaScript and validate syntax.

---

#### Step 3 — Tool Orchestration
**Goal:** Compose multiple tools reliably and handle errors gracefully.

| Capability | Implementation ideas |
|------------|----------------------|
| Tool composition patterns | Chains, branches, loops, conditionals |
| State management | Maintain conversation/state across tool invocations |
| Error handling | Retry policies, fallback strategies, graceful degradation |
| Logging & observability | Structured logs, metrics, tracing |

**Milestone:** A working orchestration framework that chains tools with proper error recovery.

---

### Track B — Cross-Language Code Operations

#### Step 1 — Language Detection
**Goal:** Automatically identify source code language from file contents.

| Capability | Implementation ideas |
|------------|----------------------|
| Extension-based heuristics | `.py` → Python, `.js` → JavaScript, etc. |
| Fuzzy matching | Analyze tokens/syntax patterns for ambiguous files |
| Confidence scoring | Return confidence level per detection |

**Milestone:** Accurately detect language for files without extension.

---

#### Step 2 — Syntax Checking
**Goal:** Validate code across different programming languages.

| Capability | Implementation ideas |
|------------|----------------------|
| Python | `ruff check`, `mypy run` |
| JavaScript/TypeScript | `eslint --no-eslintrc`, `tsc --noEmit` |
| Go | `go vet ./...` |
| Rust | `cargo clippy` |

**Milestone:** Batch syntax-check all project files and report issues by language.

---

#### Step 3 — Translation
**Goal:** Convert code between languages while preserving semantics.

| Capability | Implementation ideas |
|------------|----------------------|
| Python → JavaScript | Translate idiomatic constructs |
| Python → Go | Port standard library patterns |
| Type mapping | Preserve type signatures during translation |
| Docstring conversion | Adapt documentation format |

**Milestone:** Translate a small Python module to JavaScript with preserved functionality.

---

#### Step 4 — Refactoring Suggestions
**Goal:** Propose safe improvements across languages.

| Capability | Implementation ideas |
|------------|----------------------|
| Dead code removal | Identify unused imports, functions, variables |
| Complexity reduction | Long methods, high cyclomatic complexity |
| DRY violations | Duplicate code blocks |
| Security smells | Hardcoded secrets, unsafe eval |

**Milestone:** Analyze a codebase and generate prioritized refactoring recommendations.

---

### Track C — I/O & External Access

#### Step 1 — Voice I/O
**Goal:** Add speech recognition and synthesis.

| Capability | Implementation ideas |
|------------|----------------------|
| Speech-to-text | Integrate Whisper (`whisper-python` or `speech_recognition`) |
| Text-to-speech | Use `gTTS`, `pyttsx3`, or system TTS APIs |
| Voice-aware prompts | Adjust phrasing based on detected tone/context |
| Hands-free workflow | Start/stop actions via voice commands |

**Milestone:** A basic voice loop: speak → transcribe → process → respond aloud.

---

#### Step 2 — Web Research
**Goal:** Structured external data gathering.

| Capability | Implementation ideas |
|------------|----------------------|
| Authenticated API calls | OAuth2, service accounts, token caching |
| Rate limiting | Respectful crawling with exponential backoff |
| Content parsing | HTML/CSS/JS extraction and sanitization |
| Search indexing | Cache and index frequently queried results |

**Milestone:** A reliable web scraper that respects robots.txt and handles authentication.

---

#### Step 3 — Cloud Workspace Access
**Goal:** Authorized access to cloud-hosted data.

| Capability | Implementation ideas |
|------------|----------------------|
| Google Workspace | Docs, Sheets, Drive via official SDKs |
| Microsoft Graph | Teams, OneDrive, SharePoint |
| Slack/Teams | Channel messages, file uploads |
| Permission handling | Token acquisition, scope validation |

**Milestone:** Authenticate to Google Workspace and summarize a user's recent Docs.

---

### Track D — Execution & Quality

#### Step 1 — Workflow Templates
**Goal:** End-to-end task execution patterns.

| Template | Description |
|----------|-------------|
| Draft → Review → Publish | Write a blog post from a brief |
| Requirements → Design → Implement | Build a mini-app from a spec |
| Test generation | Create unit tests from docstrings |
| Email drafting | Generate personalized emails with tone adaptation |

**Milestone:** End-to-end "plan → draft → review" cycle for a real-world task.

---

#### Step 2 — Error Recovery
**Goal:** Robust failure handling.

| Capability | Implementation ideas |
|------------|----------------------|
| Retry with backoff | Exponential backoff with jitter |
| Circuit breakers | Prevent cascading failures |
| Fallback paths | Graceful degradation when primary fails |
| Dead letter queues | Store failed operations for manual review |

**Milestone:** A resilient pipeline that recovers from transient network/API failures.

---

#### Step 3 — Evaluation
**Goal:** Objective measurement of quality.

| Metric | Purpose |
|-------|--------|
| Task completion rate | % of tasks successfully executed |
| Response time | Latency for voice/text interactions |
| Accuracy | Hallucination rates, factual correctness |
| User satisfaction | Feedback collection and sentiment analysis |

**Milestone:** A dashboard showing key metrics over time.

---

## Quick Reference

| Track | Focus Area |
|------|-----------|
| A | Core Intelligence |
| B | Cross-Language Code Operations |
| C | I/O & External Access |
| D | Execution & Quality |

---

*End of file.*
