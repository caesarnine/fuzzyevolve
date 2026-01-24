# fuzzyevolve

Inspired by AlphaEvolve, but designed for “fuzzy” tasks like "write a evocative sci-fi short story". 

What you get in practice:
- A repeatable loop that steadily improves a draft when “good” is subjective.
- A population of diverse candidates (not 50 near-identical paraphrases).
- A full run record you can resume, audit, and browse in a TUI.

At the end you get a diverse, high quality set of outputs for your goal. Especially fun to see what "lineages" survive and which ones get pruned.

## Potential applications

This is all an experiment - but here are some things I've played with/plan on playing with:
- Creative writing/short stories
- Prompts for use in downstream tasks/agents
- Prompts for image/video models where the judge actually generates -> evaluates the actual output
- Safety/jailbreaking tests, can you find a niche/diverse set of inputs that jailbreak LLMs

## Quick start

```bash
export GOOGLE_API_KEY=... # default config uses google-gla:* models
uv sync

# Uses ./config.toml if present (or defaults)
uv run fuzzyevolve "This is my starting prompt."
```

fuzzyevolve uses [`pydantic-ai`](https://ai.pydantic.dev/) for LLM calls, so it should work with **Google**, **OpenAI**, or **Anthropic** models (and anything else pydantic-ai supports). Configure models via `[llm].judge_model` and `[[llm.ensemble]].model` in `config.toml`, and set the corresponding API key env var.

### Included examples

- `config.toml` is a working example config you can start from (and `fuzzyevolve` will auto-detect it if it’s in your CWD).
- `best.md` is a real output report from a run (top individuals by fitness + per-metric μ/σ).

Example `config.toml` switch:

```toml
[llm]
judge_model = "openai:gpt-4o-mini"

[[llm.ensemble]]
model = "openai:gpt-4o-mini"
weight = 1.0
temperature = 1.0
```

Input can be a string, a file path, or stdin:

```bash
uv run fuzzyevolve seed.txt
cat seed.txt | uv run fuzzyevolve
```

Output goes to `best.md` by default (override with `--output`). By default it includes the top 20 individuals by fitness (override with `--top`).

Override the goal/metrics quickly from the CLI:

```bash
uv run fuzzyevolve \
  --goal "Write a punchy, helpful README section about caching." \
  --metric clarity --metric usefulness --metric concision \
  "Draft text goes here..."
```

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

Embeddings use `sentence-transformers` (installed by default). Configure the model via `[embeddings].model` in `config.toml`.

## What it does (high level)

- **Critique**: a structured critique of the current parent (preserve / issues / rewrite routes).
- **Mutate:** multiple LLM “operators” propose children (e.g. conservative improvement vs high-variance exploration).
- **Judge:** an LLM ranks parent/children (and optional anchors/opponent) per metric using tiered rankings (ties allowed).
- **Learn:** per-metric TrueSkill updates convert rankings into ratings (μ/σ), then a conservative score selects “best so far”.
- **Stay diverse:** a fixed-size population is maintained using embedding-space crowding/pruning.

## Mental model (the important bits)

- An “individual” is a text plus:
  - an embedding (for diversity), and
  - a **TrueSkill rating per metric** (for quality).
- The judge doesn’t assign absolute scores; it **ranks** candidates relative to each other per metric.
- The population is a fixed-size “portfolio” spread out in embedding space.
- Exploration is encouraged via an optimistic parent selector (`μ + β·σ`), while reporting uses a conservative score (`μ - c·σ`).

## How it works (one iteration, step by step)

1. **Select parent** from the population (mixture policy: uniform sampling or optimistic tournament).
2. **Critique parent** into reusable guidance: what to preserve, what to fix, distinct rewrite routes.
3. **Plan mutation jobs** across operators (minimums + weighted sampling).
4. **Generate children** (LLM rewrites). Exploration operators can intentionally omit the parent text to avoid “paraphrase gravity”.
5. **Assemble a battle**: parent + children (+ optional frozen anchors) (+ optional opponent from the pool).
6. **Judge by ranking**: the LLM returns tiered rankings for each metric (ties allowed; outputs are validated and optionally repaired).
7. **Update ratings** with per-metric TrueSkill, freezing anchors.
8. **Insert children** into the fixed-size pool; enforce diversity with embedding-space crowding/pruning.

## Configuration

Config is a single TOML/JSON file. If `config.toml` or `config.json` exists in the current directory it’s auto-detected; pass an explicit file with `--config`.

See `config.toml` for a complete example. The structure is intentionally nested:

- `[task]` and `[metrics]` define what “good” means (goal + metric names/descriptions).
- `[mutation]` defines the operator set, job budget, and per-operator uncertainty.
- `[judging]` controls judge retries + optional opponents.
- `[rating]` controls TrueSkill parameters and the score’s LCB constant.
- `[embeddings]` defines the sentence-transformers model to use for diversity.
- `[population]` defines the fixed pool size.
- `[selection]` configures the parent-selection mixture policy.
- `[anchors]` optionally injects frozen reference anchors (seed + periodic “ghosts”) into battles.
- `[llm]` chooses the judge model and the mutation ensemble.

### Config Tips

- **Cost/latency**
  - Reduce `[mutation].jobs_per_iteration` and/or `[mutation].max_children`.
  - Use cheaper models in `[[llm.ensemble]]` and/or for `[llm].judge_model`.
  - Disable `[critic].enabled` if you want “mutate + judge” only.
- **Diversity**
  - Tune `[embeddings].model` if you want a different embedding model.
  - Increase population size, or use `population.pruning = "knn_local_competition"` to preserve niches.
- **Stability**
  - Increase `[judging].max_attempts` if the judge sometimes returns invalid structure.
  - Use anchors and/or opponents for better cross-population calibration.

## Run data

When `--store` is enabled (default), each run is recorded under `.fuzzyevolve/runs/<run_id>/`:
- `checkpoints/latest.json` and `checkpoints/it000123.json` (periodic checkpoints)
- `texts/<sha256>.txt` (deduped text blobs)
- `events.jsonl` (structured iteration events)
- `stats.jsonl` (best score + pool size over time)
- `llm/` + `llm.jsonl` (raw prompts/outputs, indexed)

This is great for debugging and iteration, but it also means **your prompts and model outputs are stored locally**. Avoid evolving sensitive content if you don’t want it written to disk.

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
- Any model supported by [`pydantic-ai`](https://ai.pydantic.dev/) (Google/OpenAI/Anthropic all work; configure via `[llm].judge_model` and `[[llm.ensemble]].model`)
- An API key for the provider you choose

```bash
export GOOGLE_API_KEY=...     # e.g. google-gla:*
export OPENAI_API_KEY=...     # e.g. openai:*
export ANTHROPIC_API_KEY=...  # e.g. anthropic:*
```

## Troubleshooting

- **`ImportError: sentence-transformers is required`**
  - Run `uv sync` (or `pip install sentence-transformers`).
- **Judge returns invalid rankings / retries fail**
  - Increase `[judging].max_attempts`, or switch to a more reliable judge model.
- **Runs are expensive**
  - Start with fewer metrics, fewer mutation jobs, and a smaller population. Then scale up.
- **Resume isn’t picking up where you expect**
  - Point `--resume` at a run directory (or a checkpoint file). The latest checkpoint is `checkpoints/latest.json`.

## Development

```bash
uv sync --extra dev
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

## License

Apache 2.0 — see `LICENSE`.
