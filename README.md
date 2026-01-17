# fuzzyevolve

> Inspired by AlphaEvolve — but can evolve any text using fuzzy criteria like “creative”, “funny”, “interesting”, etc.

`fuzzyevolve` is a small experimental playground that:

- *Mutates* text via an LLM that proposes **exact substring search/replace edits** (mechanically checkable).
- *Evaluates* candidates via an LLM that **ranks** texts across multiple metrics.
- Updates per-metric **TrueSkill** ratings (μ/σ) from those rankings.
- Maintains diversity with a **MAP-Elites** archive (top‑k per cell).

The end result is an always-improving, always-diverse population of texts, steered by whatever metrics you configure.

---

## Quick Start

```bash
uv sync
source .venv/bin/activate

# Uses ./config.toml if present (or defaults)
fuzzyevolve "This is my starting prompt."
```

Input can be a string, a file path, or stdin:

```bash
fuzzyevolve seed.txt
cat seed.txt | fuzzyevolve
```

The best result is written to `best.txt` by default.

---

## Configuration

Config lives in one TOML/JSON file (pass with `--config`). If `config.toml` or `config.json` exists in the current directory, it will be used automatically.

See the repo’s `config.toml` for a complete example. The structure is intentionally nested:

- `[run]`: iterations, logging cadence, random seed
- `[population]`: islands, elites per cell
- `[descriptor]`: how texts map into MAP‑Elites cells (`semantic_2d` or `length`)
- `[metrics]`: metric names and optional descriptions (fed to LLM prompts)
- `[rating]`: TrueSkill parameters + scoring and child priors
- `[mutation]`: mutation call budget + prompt goal/instructions
- `[judging]`: battle size + retry/repair + optional opponent
- `[anchors]`: frozen reference anchors injected into battles
- `[maintenance]`: migration and global sparring intervals

---

## CLI Options

- `--config` / `-c`: Path to TOML/JSON config
- `--output` / `-o`: Output path (default `best.txt`)
- `--iterations` / `-i`: Override `run.iterations`
- `--goal` / `-g`: Override `mutation.prompt.goal`
- `--metric` / `-m`: Override `metrics.names` (repeatable)
- `--log-level` / `-l`: Logging level (`debug|info|warning|error|critical` or a number)
- `--log-file`: Write logs to a specific file
- `--quiet` / `-q`: Hide the progress bar and non-essential logging

---

## Workflow & Architecture

The core algorithm is intentionally “ports and adapters”:

- `fuzzyevolve/core/`: domain logic (engine, archive, battle assembly, TrueSkill rating system)
- `fuzzyevolve/adapters/`: integrations (LLM mutator + LLM ranker)

High-level loop (per iteration):

1. Select a parent elite from an island archive.
2. Pick “inspirations” (mentor/champion/random) to show the mutator.
3. Call the mutator LLM (possibly multiple times) → candidate children (via exact patching).
4. Assemble a “battle” (parent + children + optional anchors/opponent).
5. Call the ranker LLM → per-metric tiered rankings.
6. Apply TrueSkill updates; insert judged children into MAP‑Elites (top‑k per cell).
7. Periodically migrate between islands / run global sparring.

---

## Development

```bash
uv sync --extra dev

uv run ruff format .
uv run ruff check .

uv run pytest -q
```

---

## License

Apache 2.0 — see `LICENSE`.

