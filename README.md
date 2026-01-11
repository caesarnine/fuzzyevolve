# fuzzyevolve

> Inspired by AlphaEvolve - but can work on any text, evolving using fuzzy criteria like "creative", "funny", "interesting", etc.

`fuzzyevolve` is an experimental playground that *mutates* pieces of text (prompts, prose, code snippets – anything), scores each mutation with a multi‑metric LLM judge, and maintains diversity with a MAP‑Elites archive. The result is an **always‑improving population** of texts, automatically steered toward whatever creative goal you configure.

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Workflow & Architecture](#workflow--architecture)

   * [Sequence diagram](#sequence-diagram)
   * [Flow diagram](#flow-diagram)
5. [Repository Layout](#repository-layout)
6. [Extending the System](#extending-the-system)
7. [Development Setup](#development-setup)
8. [License](#license)

---

## Features

| Category              | Highlights                                                                                                                                         |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Evolutionary core** | *MAP‑Elites* archive with **top‑k** elites per cell, multi‑island architecture, periodic migration & global sparring.                              |
| **LLM ensemble**      | Probabilistic model picker (`pick_model`) lets you blend fast & slow models (e.g. Gemini Flash vs Pro).                                            |
| **Judge & scoring**   | One LLM ranks candidates across *N* metrics. Ratings are updated with **TrueSkill** (one environment per metric) – uncertainty aware and additive. |
| **Mutation grammar**  | Mutator LLM returns structured search/replace edits with exact substrings.                                                                         |
| **Rich UX**           | Colour console logging, animated progress bar, optional `MutationViewer` that live‑renders recent edits.                                           |
| **Config‑first**      | All knobs (axes, metrics, model weights, iterations…) live in a single \[TOML/JSON] config file.                                                   |
| **Pure Python ≥3.10** | No compiled extensions; runs anywhere you can `pip install`.                                                                                       |

---

## Quick Start

To get started with `fuzzyevolve`, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/caesarnine/fuzzyevolve.git
    cd fuzzyevolve
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    uv sync
    source .venv/bin/activate
    ```

3.  **Set up your LLM provider:**
    By default, `fuzzyevolve` uses Google Gemini via PydanticAI's `google-gla` provider. Set `GOOGLE_API_KEY` for authentication. You can also edit the `config.toml` to use other models supported by [PydanticAI](https://ai.pydantic.dev/).

4.  **Run the evolution:**
    You can provide the initial text as an argument, a file path, or via standard input.

    *   **From a string:**
        ```bash
        fuzzyevolve "This is my starting prompt."
        ```

    *   **From a file:**
        ```bash
        echo "This is my starting prompt." > seed.txt
        fuzzyevolve seed.txt
        ```

    *   **From stdin:**
        ```bash
        cat seed.txt | fuzzyevolve
        ```
    The best result will be saved to `best.txt` by default. You can specify a different output file with the `-o` or `--output` option.

---

## Configuration

`fuzzyevolve` can be configured via a TOML or JSON file (specified with the `--config` option) and command-line arguments. Command-line arguments will override any values set in the configuration file.

Here is an example `config.toml`:

```toml
iterations = 1000
island_count = 4
elites_per_cell = 5
migration_interval = 100
migration_size = 2
sparring_interval = 50
max_diffs = 1
inspiration_count = 3
judge_include_inspirations = false
log_interval = 10
random_seed = 42

[axes]
lang = ["txt"]
len = { bins = [0, 100, 500, 1000, 2000] }

metrics = ["creativity", "clarity", "impact"]

[[llm_ensemble]]
model = "google-gla:gemini-3-flash-preview"
p = 0.8
temperature = 1.0

[[llm_ensemble]]
model = "google-gla:gemini-3-pro-preview"
p = 0.2
temperature = 0.8

judge_model = "google-gla:gemini-3-pro-preview"

mutation_prompt_goal = "Evolve this text to be more persuasive."
mutation_prompt_instructions = "Propose a single search/replace edit to improve the text."
```

### Command-Line Options

*   `--config` / `-c`: Path to a TOML or JSON config file.
*   `--output` / `-o`: Path to save the best final result.
*   `--iterations` / `-i`: Number of evolution iterations.
*   `--goal` / `-g`: The high-level goal for the mutation prompt.
*   `--metric` / `-m`: A metric to evaluate against (can be specified multiple times).
*   `--judge-model`: The LLM to use for judging candidates.
*   `--log-level` / `-l`: Logging level (debug, info, warning, error, critical) or a number.
*   `--log-file`: Path to write detailed logs.
*   `--quiet` / `-q`: Suppress the progress bar and non-essential logging.

For a full list of configuration options, please refer to the `Config` class in `fuzzyevolve/config.py`.

---

## Workflow & Architecture

### Sequence diagram

```mermaid
sequenceDiagram
    actor Runner as run()
    participant Arc as MapElitesArchive
    participant MutLLM as LLM‑mutation
    participant Patch as Search/replace applier
    participant Judge as LLMJudge
    participant JLLM as LLM‑judge
    participant TS as TrueSkill

    Runner->>Arc: sample parent
    Runner->>MutLLM: mutation prompt
    MutLLM-->>Runner: edit proposals
    loop each edit
        Runner->>Patch: apply_search_replace()
        Patch-->>Runner: child text
    end
    Runner->>Judge: rank_and_rate(parent, children)
    Judge->>JLLM: ranking prompt
    JLLM-->>Judge: rankings
    Judge->>TS: update ratings
    TS-->>Judge: new mu/sigma
    Judge-->>Runner: updated mu/sigma
    loop each child
        Runner->>Arc: add child, resort bucket
    end
    alt migration
        Runner->>Arc: migrate elites
    end
    alt global sparring
        Runner->>Judge: rank pool across islands
        Judge->>TS: update
    end
```

### Flow diagram

```mermaid
graph TD
    subgraph Init
        A1[Load config] --> A2[setup_logging]
        A2 --> A3[Seed text -> Elite]
        A3 --> A4[Create islands]
    end
    subgraph Loop[Evolution Loop]
        B1[Pick island & parent]
        B2[Pick inspirations]
        B3[Ask mutator‑LLM]
        B4[Apply edits]
        B5[Judge.rank_and_rate]
        B6[Archive.add children]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6
    end
    A4 --> Loop
    Loop -->|every k| Mig[Migration]
    Loop -->|every m| Spar[Global sparring]
    Mig --> Loop
    Spar --> Loop
    Loop --> End[Write best.txt]
```

---

## Repository Layout

```
fuzzyevolve/
├── core/
│   ├── archive.py         # MAP-Elites archive implementation
│   ├── descriptors.py     # Descriptor space definitions
│   ├── engine.py          # Evolution loop orchestration
│   ├── judge.py           # LLM-based multi-metric judge
│   ├── models.py          # Core data models
│   └── scoring.py         # TrueSkill scoring implementation
├── llm/
│   ├── client.py          # Model ensemble selection
│   ├── models.py          # LLM model specs
│   └── prompts.py         # Prompt building functions
├── mutation/
│   └── mutator.py         # Mutation generation
├── console/
│   ├── logging.py         # Logging setup
│   └── mutation_viewer.py # Mutation edit viewer
├── cli.py                 # Command-line interface (Typer)
├── config.py              # Configuration loading
└── __init__.py

best.txt                   # Default output file
pyproject.toml             # Project metadata and dependencies
README.md                  # This file
```

---

## Extending the System

| Want to…                    |  How                                                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Add metrics**             | Append names to `metrics` array; TrueSkill envs & judge prompt auto‑expand.                                            |
| **Change descriptor space** | Edit `[axes]` in config and ensure your code sets those descriptor keys before `archive.add()`.                        |
| **Swap LLM back‑end**       | Change `model=` strings to any provider supported by PydanticAI.                                                       |
| **Evolve other artefacts**  | Feed binary‑friendly descriptors & patch logic (e.g., JSON merge, AST diffs).                                          |
| **Web UI**                  | The dependency list already includes Flask‑SocketIO – wire `MutationViewer` into a websocket for real‑time dashboards. |

---

## Development Setup

```bash
# Create venv and install dev extras
uv venv && source .venv/bin/activate
uv sync --extra dev

# Lint & format
uv run ruff format .
uv run ruff check .

# Type‑check
uv run mypy fuzzyevolve/

# Run tests
uv run pytest -q
```

---

## Coding Agents

See `AGENTS.md` to get up to speed on project conventions.

## License

This project is licensed under the **Apache 2.0 License**. See [`LICENSE`](LICENSE) for details.
