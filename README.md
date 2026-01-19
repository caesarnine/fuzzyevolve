# fuzzyevolve

Evolve text with LLM mutation + LLM judging, using TrueSkill for noisy multi-metric feedback and MAP-Elites for diversity.

Inspired by AlphaEvolve, but designed for “fuzzy” criteria like *prose*, *coherence*, *originality*, *funny*, *interesting*, etc.

## Quick start

```bash
export GOOGLE_API_KEY=... # default config uses google-gla:* models
uv sync

# Uses ./config.toml if present (or defaults)
uv run fuzzyevolve "This is my starting prompt."
```

Input can be a string, a file path, or stdin:

```bash
uv run fuzzyevolve seed.txt
cat seed.txt | uv run fuzzyevolve
```

Output goes to `best_by_cell.md` by default (override with `--output`). By default it includes the top 20 best-per-cell champions (override with `--top-cells`).

By default, each run is recorded under `.fuzzyevolve/runs/<run_id>/` (checkpoints, events, and raw LLM prompts/outputs). Resume with:

```bash
uv run fuzzyevolve --resume .fuzzyevolve/runs/<run_id> --iterations 100
```

Browse runs in the TUI:

```bash
uv run fuzzyevolve tui
# or open a specific run/checkpoint:
uv run fuzzyevolve tui --run .fuzzyevolve/runs/<run_id>
```

Disable recording with `--no-store`.

Note: the repo’s `config.toml` uses semantic embeddings via `sentence-transformers`. Either install the extra or switch to hash/length descriptors:

```bash
uv sync --extra semantic
```

## What it does

- Critiques the selected parent once per iteration (structured: preserve / issues / rewrite routes).
- Generates children via a set of LLM-backed mutation operators (e.g. “exploit” vs “explore” full rewrites).
- Judges parent/children by ranking them per metric (tiered rankings, ties allowed).
- Updates per-metric TrueSkill ratings (μ/σ) from those rankings (with uncertainty-aware scoring).
- Keeps diversity with a MAP‑Elites archive (top‑k per descriptor cell), optionally with multiple islands + migration.

## Mental model

- A text is a “player” with a TrueSkill rating per metric (e.g. one rating for `prose`, one for `coherence`).
- The judge doesn’t assign absolute scores; it *ranks* candidates relative to each other for each metric.
- The archive is a grid of “niches” (cells) defined by a descriptor (length or a 2D embedding projection).
- Each iteration is: pick a parent → critique → propose children → rank a battle → update ratings → insert children into niches.

## How it works (core loop)

1. **Descriptor**: compute `descriptor = describe(text)` to place texts into MAP‑Elites cells (`length` or `embedding_2d`).
2. **Select parent**: choose an elite from a random island archive (`uniform_cell` or an optimistic UCB-ish policy).
3. **Critique** (optional): ask an LLM for actionable guidance (issues + distinct rewrite routes).
4. **Mutate**: allocate a per-iteration job budget across operators; each job proposes one rewritten child.
5. **Assemble battle**: parent + sampled children (+ optional frozen anchors/opponent), capped by `judging.max_battle_size`.
6. **Judge**: ask an LLM to return tiered rankings for each metric (with validation + optional repair retries).
7. **Update ratings**: apply per-metric TrueSkill updates; score uses a conservative LCB (`mu - c*sigma`) averaged across metrics.
8. **Archive**: add children into MAP‑Elites (top‑k per cell), optionally gating “new cell” inserts.

## Configuration

Config is a single TOML/JSON file. If `config.toml` or `config.json` exists in the current directory it’s auto-detected; pass an explicit file with `--config`.

See `config.toml` for a complete example. The structure is intentionally nested:

- `[task]` and `[metrics]` define what “good” means (goal + metric names/descriptions).
- `[mutation]` defines the operator set, job budget, and per-operator uncertainty.
- `[judging]` controls battle size + judge retries + optional opponents.
- `[rating]` controls TrueSkill parameters and the score’s LCB constant.
- `[descriptor]` defines the MAP‑Elites “diversity axis” (length bins or 2D embedding bins).
- `[anchors]` optionally injects frozen reference anchors (seed + periodic “ghosts”) into battles.
- `[population]` / `[maintenance]` enable multiple islands, migration, and global sparring.

## CLI

`run` is the default command, so these are equivalent:

```bash
uv run fuzzyevolve "Seed text..."
uv run fuzzyevolve run "Seed text..."
```

To open the run browser:

```bash
uv run fuzzyevolve tui
```

### `run` options

- `--config` / `-c`: Path to TOML/JSON config
- `--output` / `-o`: Output path (default `best_by_cell.md`)
- `--top-cells`: How many best-per-cell champions to include (default 20; `0` = all)
- `--iterations` / `-i`: Override `run.iterations`
- `--goal` / `-g`: Override `task.goal`
- `--metric` / `-m`: Override `metrics.names` (repeatable)
- `--resume`: Resume from a previous run directory (or checkpoint file)
- `--store/--no-store`: Enable/disable recording under `.fuzzyevolve/`
- `--log-level` / `-l`: Logging level (`debug|info|warning|error|critical` or a number)
- `--log-file`: Write logs to a specific file
- `--quiet` / `-q`: Hide the progress bar and non-essential logging

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended)
- Any model supported by [`pydantic-ai`](https://ai.pydantic.dev/) (configure via `[llm].judge_model` and `[[llm.ensemble]].model`)
- An API key for the provider you choose

```bash
export GOOGLE_API_KEY=...     # e.g. google-gla:*
export OPENAI_API_KEY=...     # e.g. openai:*
export ANTHROPIC_API_KEY=...  # e.g. anthropic:*
```

Semantic embeddings require:

```bash
uv sync --extra semantic
```

## Development

```bash
uv sync --extra dev
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

## License

Apache 2.0 — see `LICENSE`.
