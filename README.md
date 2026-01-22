# fuzzyevolve

Evolve text with LLM mutation + LLM judging, using TrueSkill for noisy multi-metric feedback and a fixed-size, crowding-based population for diversity.

Inspired by AlphaEvolve, but designed for “fuzzy” criteria like *prose*, *coherence*, *originality*, *funny*, *interesting*, etc.

## Quick start

```bash
export GOOGLE_API_KEY=... # default config uses google-gla:* models
uv sync --extra semantic

# Uses ./config.toml if present (or defaults)
uv run fuzzyevolve "This is my starting prompt."
```

Input can be a string, a file path, or stdin:

```bash
uv run fuzzyevolve seed.txt
cat seed.txt | uv run fuzzyevolve
```

Output goes to `best.md` by default (override with `--output`). By default it includes the top 20 individuals by fitness (override with `--top`).

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

Semantic embeddings require `sentence-transformers` and are the default. Install the extra, or set `[embeddings].model = "hash"` for the built-in hash embedding:

```bash
uv sync --extra semantic
```

## What it does

- Critiques the selected parent once per iteration (structured: preserve / issues / rewrite routes).
- Generates children via a set of LLM-backed mutation operators (e.g. “exploit” vs “explore” full rewrites).
- Judges parent/children by ranking them per metric (tiered rankings, ties allowed).
- Updates per-metric TrueSkill ratings (μ/σ) from those rankings (with uncertainty-aware scoring).
- Keeps diversity with a fixed-size pool + crowding in embedding space (default uses kNN local competition; closest-pair elimination is configurable).

## Mental model

- A text is a “player” with a TrueSkill rating per metric (e.g. one rating for `prose`, one for `coherence`).
- The judge doesn’t assign absolute scores; it *ranks* candidates relative to each other for each metric.
- The population is a fixed-size “portfolio” of texts in embedding space.
- Each iteration is: pick a parent → critique → propose children → rank a battle → update ratings → insert children → apply crowding elimination.

## How it works (core loop)

1. **Embed**: compute `embedding = embed(text)` for parent/children (semantic by default; hash is optional).
2. **Select parent**: mixture policy: uniform sampling, or an optimistic tournament (`μ + β·σ`).
3. **Critique** (optional): ask an LLM for actionable guidance (issues + distinct rewrite routes).
4. **Mutate**: allocate a per-iteration job budget across operators; each job proposes one rewritten child.
5. **Assemble battle**: parent + children + frozen anchors + an opponent (default: far-but-close from the pool).
6. **Judge**: ask an LLM to return tiered rankings for each metric (with validation + optional repair retries).
7. **Update ratings**: apply per-metric TrueSkill updates; score uses a conservative LCB (`mu - c*sigma`) averaged across metrics.
8. **Crowding**: add children to the pool; enforce a fixed-size pool with embedding-space crowding (default: kNN local competition).

## Configuration

Config is a single TOML/JSON file. If `config.toml` or `config.json` exists in the current directory it’s auto-detected; pass an explicit file with `--config`.

See `config.toml` for a complete example. The structure is intentionally nested:

- `[task]` and `[metrics]` define what “good” means (goal + metric names/descriptions).
- `[mutation]` defines the operator set, job budget, and per-operator uncertainty.
- `[judging]` controls judge retries + optional opponents.
- `[rating]` controls TrueSkill parameters and the score’s LCB constant.
- `[embeddings]` defines the embedding model (sentence-transformers by default; use `hash` for a fast fallback).
- `[population]` defines the fixed pool size.
- `[selection]` configures the parent-selection mixture policy.
- `[anchors]` optionally injects frozen reference anchors (seed + periodic “ghosts”) into battles.

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
- `--output` / `-o`: Output path (default `best.md`)
- `--top`: How many top individuals to include (default 20; `0` = all)
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
uv sync --extra dev --extra semantic
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

## License

Apache 2.0 — see `LICENSE`.
