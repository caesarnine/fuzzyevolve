#!/usr/bin/env python
# main.py – enhanced with structured logging + Rich progress bar
# -----------------------------------------------------------------------------
# AlphaEvolve-style evolutionary loop with:
#   • MAP-Elites archive holding the **top-k** elites per cell
#   • Multiple islands + periodic migration
#   • LLM judge that returns a **ranking** for each metric
#   • TrueSkill run **per metric** to turn those rankings into ratings
#   • Parent + inspirations + child are ranked together every iteration
#
# Deps (new):  pip install rich  # plus the original litellm tqdm pydantic trueskill
# -----------------------------------------------------------------------------

from __future__ import annotations

# ╭──────────────────────────────────────────────────────────╮
# │ 0.  Imports & logging helper                             │
# ╰──────────────────────────────────────────────────────────╯
import argparse
import json
import logging
import random
import re
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import copy

import litellm
from litellm import completion
from pydantic import BaseModel, Field
from rich.logging import RichHandler
from rich.console import Console
from lxml import html  # Added for parsing LLM judge response
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
import trueskill as ts
import os  # Moved import os to the top import block

litellm.suppress_debug_info = True
litellm.set_verbose = False

console = Console()

os.environ["VERTEXAI_LOCATION"] = "global"

# ── logging setup ───────────────────────────────────────────────


def setup_logging(
    log_dir: Path | str = "logs",
    level: int = logging.INFO,
    suppress_libs: List[str] | None = None,
) -> None:
    """Configure root logger with Rich console + rotating file.

    ``suppress_libs`` — list of logger names to silence (set to WARNING).
    By default we silence noisy third‑party libs like LiteLLM / httpx.
    """

    suppress_libs = suppress_libs or ["LiteLLM", "litellm", "httpx", "urllib3"]

    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    ts_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    file_path = log_dir / f"{ts_str}.log"

    handlers: list[logging.Handler] = []

    # console (RichHandler)
    handlers.append(
        RichHandler(
            markup=True,
            rich_tracebacks=True,
            show_path=False,
            log_time_format="%H:%M:%S",
        )
    )

    # file handler (plain text)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    logging.getLogger().info("Logging to %s", file_path)

    # silence noisy libs
    for name in suppress_libs:
        logging.getLogger(name).setLevel(logging.WARNING)


# module-level loggers
log_llm = logging.getLogger("llm")
log_mut = logging.getLogger("mutation")

# ╭──────────────────────────────────────────────────────────╮
# │ 1.  Configuration (Pydantic)                             │
# ╰──────────────────────────────────────────────────────────╯


class LLMEntry(BaseModel):
    model: str
    p: float = Field(..., ge=0)
    temperature: float = 0.7


class Config(BaseModel):
    # core loop
    seed_path: str = "seed.txt"
    iterations: int = 10
    log_every: int = 1

    # map-elites + islands
    num_islands: int = 1
    k_top: int = 4
    migration_every: int = 300
    migrants_per_island: int = 4

    # llm ensemble + judge
    llm_ensemble: List[LLMEntry] = Field(
        default_factory=lambda: [
            LLMEntry(
                model="vertex_ai/gemini-2.5-flash-preview-05-20",
                p=0.85,
                temperature=1,
            ),
            LLMEntry(
                model="vertex_ai/gemini-2.5-pro-preview-06-05", p=0.15, temperature=1
            ),
        ]
    )
    judge_model: str = "vertex_ai/gemini-2.5-pro-preview-06-05"
    metrics: List[str] = ["taste", "quality"]

    # descriptor space
    axes: Dict[str, Any] = Field(
        default_factory=lambda: {
            "lang": ["txt"],
            "len": {"bins": [0, 500, 1000, 2000, 1e9]},
        }
    )

    n_diffs: int = 4
    youth_bias: float = 0.30
    # rarely-needed global sparring (still helpful early on)
    sparring_every: int = 100

    # mutation prompt
    mutation_prompt_goal: str = "write a prompt for a coding agent, the coding agent should be singularly high quality and world class"
    mutation_prompt_instructions: str = (
        "Propose one or more SEARCH/REPLACE diff blocks to improve the PARENT text. "
        "You can rewrite, shorten, or completely change the text. "
        "First, explain your reasoning in a <thinking> block. Then, provide the diffs in a <diffs> block."
    )


def load_cfg(path: str | None) -> Config:
    if not path:
        return Config()
    data = Path(path).read_text()
    try:
        cfg_dict = json.loads(data)
    except json.JSONDecodeError:
        cfg_dict = tomllib.loads(data)
    return Config.model_validate(cfg_dict)


# ╭──────────────────────────────────────────────────────────╮
# │ 2.  LiteLLM helpers                                      │
# ╰──────────────────────────────────────────────────────────╯


def llm_call(response_format, model: str, prompt: str, temperature: float = 0.7) -> str:
    """Thin wrapper around litellm.completion with debug logging."""
    try:
        rsp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format=response_format,
        )
    except Exception as exc:
        log_llm.exception("LLM call failed: %s", exc)
        raise

    log_llm.debug("PROMPT\n%s", prompt)
    log_llm.debug("RAW RESPONSE\n%s", rsp)
    return rsp.choices[0].message.content.strip()


def pick_model(pool: List[LLMEntry]) -> Tuple[str, float]:
    models, probs, temps = zip(*[(e.model, e.p, e.temperature) for e in pool])
    idx = random.choices(range(len(models)), weights=probs)[0]
    return models[idx], temps[idx]


# ╭──────────────────────────────────────────────────────────╮
# │ 3.  MAP-Elites archive (top-k per cell)                  │
# ╰──────────────────────────────────────────────────────────╯


class MixedArchive:
    def __init__(self, axes: Dict[str, Any], k_top: int):
        self.axes, self.k_top = axes, k_top
        self.cell: Dict[Tuple, List[Dict]] = {}
        total = 1
        for spec in axes.values():
            total *= len(spec) if isinstance(spec, list) else (len(spec["bins"]) - 1)
        self.total_cells = total
        self.empty_cells = total

    # --- helpers ----------------------------------------------------
    def _key(self, desc: Dict[str, Any]) -> Tuple:
        key = []
        for name, spec in self.axes.items():
            v = desc[name]
            if isinstance(spec, list):
                key.append(v)
            else:
                edges = [float(x) for x in spec["bins"]]
                idx = max(i for i, e in enumerate(edges) if float(v) >= e)
                key.append(idx)
        return tuple(key)

    def _sort_bucket(self, bucket: List[Dict]):
        bucket.sort(key=lambda e: ts_score(e["rating"]), reverse=True)

    # --- public api -------------------------------------------------
    def add(self, desc: Dict[str, Any], elite: Dict):
        key = self._key(desc)
        elite["cell_key"] = key
        if key not in self.cell:
            self.cell[key] = []
            self.empty_cells -= 1
        bucket = self.cell[key]
        bucket.append(elite)
        self._sort_bucket(bucket)
        del bucket[self.k_top :]

    def resort_elite(self, elite: Dict):
        key = elite["cell_key"]
        if key in self.cell:
            self._sort_bucket(self.cell[key])

    def random_elite(self, youth_bias: float) -> Dict:
        key = random.choice(list(self.cell.keys()))
        bucket = self.cell[key]
        if random.random() < youth_bias and len(bucket) >= 3:
            bucket = sorted(bucket, key=lambda e: e["age"])[
                : max(1, int(len(bucket) * 0.3))
            ]
        return random.choice(bucket)

    @property
    def best(self) -> Dict:
        return max(
            (e for b in self.cell.values() for e in b),
            key=lambda e: ts_score(e["rating"]),
        )


# ╭──────────────────────────────────────────────────────────╮
# │ 4.  TrueSkill helpers                                    │
# ╰──────────────────────────────────────────────────────────╯


def make_envs(metrics: List[str]) -> Dict[str, ts.TrueSkill]:
    return {m: ts.TrueSkill(draw_probability=0.0) for m in metrics}


def ts_score(
    ratings: Dict[str, ts.Rating],
    weights: Dict[str, float] | None = None,
    c: float = 2.0,
) -> float:
    w = weights or {m: 1 / len(ratings) for m in ratings}
    return sum(w[m] * (ratings[m].mu - c * ratings[m].sigma) for m in ratings)


# ╭──────────────────────────────────────────────────────────╮
# │ 5.  Prompt + Judge                                       │
# ╰──────────────────────────────────────────────────────────╯

_RANK_PROMPT_TEMPLATE = """Below are {n} texts, each tagged with its [ID].
Your task is to evaluate these texts based on the following metrics: {metrics_list_str}.

First, provide your step-by-step thinking process within <thinking> tags.
Then, for each metric, provide a comma-separated list of IDs, ordered from best to worst, within its own XML-like tag. Tag names should be lowercase.

Example for metrics {metrics_list_str}:
<response_format>
<thinking>
[Your detailed rationale for rankings. Explain your reasoning for each metric, comparing the candidates.
For instance, for metric 'metric_name_1', candidate [id_x] was ranked higher than [id_y] because...
For metric 'metric_name_2', candidate [id_z] demonstrated stronger qualities in X, Y, Z leading to its top rank.]
</thinking>
<output>
{metric_tags_str}
</output>
</response_format>

Ensure your response strictly follows this format.

Metrics: {metrics_list_str}

Candidates:
{candidates_str}

Follow this exact response format:
<response_format>
<thinking>[Your step-by-step thinking process and rationale for rankings for each metric]</thinking>
<output>
{metric_tags_str}
</output>
</response_format>
"""


def parse_llm_judge_response(
    raw_xml_string: str, metrics: List[str], logger: logging.Logger
) -> Tuple[str | None, Dict[str, List[int]]]:
    """
    Parses the pseudo-XML response from the LLM judge.

    Returns:
        A tuple containing:
        - The text content of the <thinking> tag (or None if not found/error).
        - A dictionary mapping metric names to a list of ranked IDs.
          Returns an empty dict for rankings if parsing fails for the output section.
    """
    thinking_text: str | None = None
    rankings: Dict[str, List[int]] = {}

    try:
        # Use .lower() for tag finding to be robust against LLM casing variations for common tags
        doc = html.fromstring(raw_xml_string.encode("utf-8"))

        thinking_node = doc.find(".//thinking")  # Standard casing
        if thinking_node is None:  # Fallback to lowercase
            thinking_node = doc.find(".//Thinking")
        if thinking_node is not None and thinking_node.text is not None:
            thinking_text = thinking_node.text.strip()

        output_node = doc.find(".//output")  # Standard casing
        if output_node is None:  # Fallback to lowercase
            output_node = doc.find(".//Output")

        if output_node is None:
            logger.error("LLM Judge: <output> tag not found in response.")
            # Try regex fallback before returning empty if critical tags are missing
        else:  # <output> tag found, proceed with parsing metrics
            for metric_name in metrics:
                # Normalize metric name for tag search (e.g., lowercase, replace spaces/underscores if needed)
                # Assuming metric names in cfg.metrics are simple strings usable as tags or can be sanitized.
                # For robustness, prompt asks LLM for lowercase tags.
                tag_name_to_find = metric_name.lower()
                metric_node = output_node.find(f".//{tag_name_to_find}")

                if metric_node is not None and metric_node.text is not None:
                    try:
                        raw_ids_str = metric_node.text.strip()
                        if raw_ids_str.startswith("[") and raw_ids_str.endswith("]"):
                            raw_ids_str = raw_ids_str[1:-1]

                        ids = [
                            int(id_str.strip())
                            for id_str in raw_ids_str.split(",")
                            if id_str.strip()
                        ]
                        rankings[metric_name] = (
                            ids  # Store with original metric name from config
                        )
                    except ValueError as e:
                        logger.error(
                            "LLM Judge: Could not parse IDs for metric '%s' (tag '%s') from text: '%s'. Error: %s",
                            metric_name,
                            tag_name_to_find,
                            metric_node.text,
                            e,
                        )
                else:
                    logger.warning(
                        "LLM Judge: Metric tag <%s> not found or empty in <output> for metric '%s'.",
                        tag_name_to_find,
                        metric_name,
                    )

        # If XML parsing yielded no rankings and output_node was not found, attempt regex fallback
        if (
            not rankings and output_node is None
        ):  # only if main parsing completely failed for output
            logger.warning(
                "LLM Judge: <output> tag not found by XML parser, attempting regex fallback."
            )
            # Regex fallback logic (as previously designed, simplified here for brevity, but should be robust)
            thinking_match = re.search(
                r"<thinking>(.*?)</thinking>", raw_xml_string, re.DOTALL | re.IGNORECASE
            )
            if thinking_match:
                thinking_text = thinking_match.group(
                    1
                ).strip()  # Overwrite if found by regex

            output_match = re.search(
                r"<output>(.*?)</output>", raw_xml_string, re.DOTALL | re.IGNORECASE
            )
            if output_match:
                output_content = output_match.group(1)
                for metric_name in metrics:
                    tag_name_to_find = metric_name.lower()
                    metric_match = re.search(
                        rf"<{tag_name_to_find}>(.*?)</{tag_name_to_find}>",
                        output_content,
                        re.DOTALL | re.IGNORECASE,
                    )
                    if metric_match and metric_match.group(1):
                        raw_ids_str = metric_match.group(1).strip()
                        if raw_ids_str.startswith("[") and raw_ids_str.endswith("]"):
                            raw_ids_str = raw_ids_str[1:-1]
                        try:
                            ids = [
                                int(id_str.strip())
                                for id_str in raw_ids_str.split(",")
                                if id_str.strip()
                            ]
                            rankings[metric_name] = (
                                ids  # Store with original metric name
                            )
                        except ValueError:
                            logger.warning(
                                "LLM Judge: Regex fallback failed to parse IDs for metric '%s' from: '%s'",
                                metric_name,
                                raw_ids_str,
                            )
            if not rankings:  # If regex also failed
                logger.error(
                    "LLM Judge: XML and Regex fallback both failed to extract rankings."
                )

    except html.etree.XMLSyntaxError as e:
        logger.error(
            "LLM Judge: Failed to parse XML response structure. Error: %s\nRaw response fragment:\n%s",
            e,
            raw_xml_string[:500],
        )
        # Simplified Regex fallback can be attempted here too if desired, similar to above.
    except Exception as e:  # Catch any other unexpected errors during parsing
        logger.exception("LLM Judge: Unexpected error during XML parsing. Error: %s", e)

    return thinking_text, rankings


def make_rank_prompt(metrics: List[str], items: List[Tuple[int, Dict]]) -> str:
    candidate_lines = []
    for idx, elite_data in items:
        candidate_lines.append(f"[{idx}]\n{elite_data['txt']}\n")

    candidates_str = "\n".join(candidate_lines)
    # Ensure metric names in the prompt are consistent (e.g. user sees "Prose", LLM uses "prose")
    metrics_list_str = ", ".join(
        f'"{m}"' for m in metrics
    )  # e.g. "Prose", "Plot Quality"

    metric_tags_str_parts = []
    example_ids_str = (
        ", ".join(
            str(i) for i in random.sample(range(len(items)), k=min(len(items), 3))
        )
        if items
        else "id_1, id_2"
    )
    for metric_name in metrics:
        # LLM should use lowercase tags as requested in the main prompt instructions
        tag_name = metric_name.lower()
        metric_tags_str_parts.append(f"<{tag_name}>[{example_ids_str}]</{tag_name}>")
    metric_tags_str = "\n".join(metric_tags_str_parts)

    return _RANK_PROMPT_TEMPLATE.format(
        n=len(items),
        metrics_list_str=metrics_list_str,  # Show original metric names to user in prompt
        candidates_str=candidates_str,
        metric_tags_str=metric_tags_str,  # Example output uses lowercase tags
    )


class MultiMetricJudge:
    def __init__(self, model: str, metrics: List[str]):
        self.model = model
        self.metrics = metrics
        self.envs = make_envs(metrics)

    def ensure_ratings(self, elite: Dict):
        if "rating" not in elite:
            elite["rating"] = {m: self.envs[m].create_rating() for m in self.metrics}

    def rank_and_rate(self, players: List[Dict]):
        for p in players:
            self.ensure_ratings(p)

        if not players:
            log_llm.warning("Rank and Rate called with no players. Skipping.")
            return

        num_players = len(players)

        # 1. Create a list of original indices and shuffle them.
        # These original_indices refer to the positions in the input 'players' list.
        shuffled_original_indices = list(range(num_players))
        random.shuffle(shuffled_original_indices)

        # 2. Prepare items for the prompt with new, temporary IDs (0 to N-1 for LLM).
        # And create a mapping from these temporary prompt_ids back to original player objects.
        prompt_items_for_llm: List[Tuple[int, Dict]] = []
        # Holds the player object at the position of its temporary prompt_id
        temp_prompt_id_to_player_obj_map: List[Dict] = [{} for _ in range(num_players)]

        for temp_prompt_id, original_idx_val in enumerate(shuffled_original_indices):
            player_obj = players[original_idx_val]
            prompt_items_for_llm.append((temp_prompt_id, player_obj))
            temp_prompt_id_to_player_obj_map[temp_prompt_id] = player_obj

        prompt = make_rank_prompt(self.metrics, prompt_items_for_llm)

        raw_response: str | None = None
        try:
            # Expecting raw string with pseudo-XML, so response_format is None
            raw_response = llm_call(
                model=self.model,
                prompt=prompt,
                temperature=0.2,  # Consider if temperature needs adjustment for XML
                response_format=None,
            )
        except Exception:
            # llm_call already logs details of the exception
            log_llm.error(
                "Judge LLM call failed outright — skipping rating update for this batch."
            )
            return

        if not raw_response:
            log_llm.error(
                "Judge LLM returned an empty response — skipping rating update."
            )
            return

        # Log the raw response before parsing attempt, can be very verbose
        # log_llm.debug("Judge LLM raw response:\n%s", raw_response)

        thinking_process, parsed_rankings = parse_llm_judge_response(
            raw_response, self.metrics, log_llm
        )

        if thinking_process:
            log_llm.info(
                "Judge LLM rationale:\n%s", thinking_process
            )  # Log thinking as INFO
        else:
            log_llm.warning("Judge LLM: No thinking process extracted from response.")

        if not parsed_rankings:
            log_llm.error(
                "Judge LLM: No valid rankings extracted from response after parsing. Skipping rating update for this batch."
            )
            return

        # Iterate through metrics defined in config to update ratings
        for metric_name_from_config in self.metrics:
            # 'ranked_temp_prompt_ids' contains the temporary prompt IDs (0..N-1)
            # as ranked by the LLM for this metric (e.g., [2, 0, 1])
            ranked_temp_prompt_ids = parsed_rankings.get(metric_name_from_config)

            if (
                not isinstance(ranked_temp_prompt_ids, list)
                or len(ranked_temp_prompt_ids) < 1
            ):
                log_llm.warning(
                    "Judge LLM: Insufficient or invalid ranking list for metric '%s'. Found: %s. Skipping this metric.",
                    metric_name_from_config,
                    ranked_temp_prompt_ids,
                )
                continue

            # Validate temporary prompt IDs returned by LLM
            valid_ranked_players_for_metric = []
            # Ensure no duplicates from LLM for this metric, using temporary prompt IDs
            seen_temp_prompt_ids_in_ranking = set()
            # The set of valid temporary prompt IDs LLM could have returned
            valid_temp_prompt_id_set = set(range(num_players))

            for temp_prompt_id_from_llm in ranked_temp_prompt_ids:
                if (
                    temp_prompt_id_from_llm in valid_temp_prompt_id_set
                    and temp_prompt_id_from_llm not in seen_temp_prompt_ids_in_ranking
                ):
                    # Map temp_prompt_id back to the actual player object
                    player_obj = temp_prompt_id_to_player_obj_map[
                        temp_prompt_id_from_llm
                    ]
                    valid_ranked_players_for_metric.append(player_obj)
                    seen_temp_prompt_ids_in_ranking.add(temp_prompt_id_from_llm)
                else:
                    log_llm.warning(
                        "Judge LLM: Metric '%s' ranking contains invalid or duplicate temporary prompt ID: %s. It will be ignored.",
                        metric_name_from_config,
                        temp_prompt_id_from_llm,
                    )
            if (
                len(valid_ranked_players_for_metric) < 1
            ):  # Can we rate a single player? TrueSkill typically needs >=2 teams
                # For single player "rating", it's more like assigning a score.
                # If TrueSkill needs >=2, this check should be < 2.
                # ts.rate() needs at least one rating group. A "team" of one is one rating group.
                # If only one player ranked, they are rank 0.
                log_llm.warning(
                    "Judge LLM: Not enough valid players for metric '%s' after validation (%d found). Skipping rating update for this metric.",
                    metric_name_from_config,
                    len(valid_ranked_players_for_metric),
                )
                continue

            # Proceed with TrueSkill update using valid_ranked_players_for_metric
            # Each player is a "team" of one.
            current_ratings_for_trueskill = [
                [p["rating"][metric_name_from_config]]
                for p in valid_ranked_players_for_metric
            ]

            try:
                # Ranks are 0, 1, 2... for the best, second best, etc.
                updated_ratings_tuples = self.envs[metric_name_from_config].rate(
                    current_ratings_for_trueskill,
                    ranks=list(range(len(valid_ranked_players_for_metric))),
                )
                # Update player objects with new TrueSkill Rating objects
                for player_obj, new_rating_tuple in zip(
                    valid_ranked_players_for_metric, updated_ratings_tuples
                ):
                    player_obj["rating"][metric_name_from_config] = new_rating_tuple[
                        0
                    ]  # new_rating_tuple is (Rating(),)
            except IndexError as e:
                log_llm.error(
                    "Judge LLM: IndexError during TrueSkill update for metric '%s', possibly due to issues with "
                    "valid_ranked_players_for_metric or their ratings. Error: %s. Players: %s",
                    metric_name_from_config,
                    e,
                    valid_ranked_players_for_metric,
                )
            except Exception as e:
                log_llm.error(
                    "Judge LLM: Unexpected error during TrueSkill update for metric '%s': %s. Skipping.",
                    metric_name_from_config,
                    e,
                )


# ╭──────────────────────────────────────────────────────────╮
# │ 6.  Diff utilities                                       │
# ╰──────────────────────────────────────────────────────────╯

_DIFF_START = re.compile(r"^<<<<<<< SEARCH", re.M)


def split_blocks(raw: str) -> List[str]:
    blocks, cur = [], []
    for ln in raw.splitlines():
        if ln.startswith("<<<<<<< SEARCH"):
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        cur.append(ln)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def apply_patch(text: str, block: str) -> str:
    try:
        _, rest = block.split("<<<<<<< SEARCH", 1)
        search, rest = rest.split("=======", 1)
        replace, _ = rest.split(">>>>>>> REPLACE", 1)
    except ValueError:
        log_mut.warning(
            "Malformed diff block, cannot apply patch. Block (first 200 chars): '%s'",
            block[:200],
        )
        return text
    search = search.strip("\n\r")
    replace = replace.strip("\n\r")
    idx = text.find(search)
    return text if idx == -1 else text.replace(search, replace, 1)


def parse_llm_mutation_response(
    raw_response: str, logger: logging.Logger
) -> Tuple[str | None, str | None]:
    """
    Parses the response from the mutation LLM.

    Extracts content from <thinking> and <diffs> tags using regex.

    Returns:
        A tuple containing:
        - The text content of the <thinking> tag (or None).
        - The text content of the <diffs> tag (or None).
    """
    thinking, diffs = None, None
    try:
        thinking_match = re.search(
            r"<thinking>(.*?)</thinking>", raw_response, re.DOTALL | re.IGNORECASE
        )
        if thinking_match:
            thinking = thinking_match.group(1).strip()

        diffs_match = re.search(
            r"<diffs>(.*?)</diffs>", raw_response, re.DOTALL | re.IGNORECASE
        )
        if diffs_match:
            diffs = diffs_match.group(1).strip()
        else:
            # Fallback for LLMs that forget the <diffs> wrapper but still output diffs
            if not thinking_match and _DIFF_START.search(raw_response):
                logger.warning(
                    "No <diffs> tag found, but diff-like content detected. Using raw response."
                )
                diffs = raw_response

    except Exception as e:
        logger.exception("Failed to parse mutation LLM response: %s", e)

    if not thinking:
        logger.warning("Mutation LLM: No <thinking> block found in response.")
    if not diffs:
        logger.warning("Mutation LLM: No <diffs> block found in response.")

    return thinking, diffs


# ╭──────────────────────────────────────────────────────────╮
# │ 7.  Prompt builder for mutations                         │
# ╰──────────────────────────────────────────────────────────╯

_MUT_PROMPT_TEMPLATE = """
Your overall goal is: {goal}
Your task is: {instructions}

First, provide your step-by-step thinking process on how to improve the PARENT text.

Analyze its weaknesses and explain the changes you will propose, potentially drawing inspiration from the other texts provided.

Then, provide the diffs using the exact SEARCH/REPLACE syntax inside a <diffs> block.

Use exactly this diff syntax:
<<<<<<< SEARCH
<text to match>
<replacement>
>>>>>>> REPLACE

Example Response:

<thinking>
[Your detailed thought process for improving the parent text.]
</thinking>
<diffs>
<<<<<<< SEARCH
<a paragraph from the middle of the parent text>
=======
<a new, improved paragraph with better pacing>
>>>>>>> REPLACE

<<<<<<< SEARCH
<the last sentence of the parent text>
=======
<a new, more impactful ending>
>>>>>>> REPLACE
</diffs>

──────────────── PARENT ────────────────
Score   : {p_score:.3f}
{p_text}

──────────────── INSPIRATIONS ───────────
{insp_text}
──────────────────────────────────────────

Remember to follow the response format exactly: <thinking>...</thinking><diffs>...</diffs>
"""


def build_mut_prompt(
    parent: Dict, inspirations: List[Dict], goal: str, instructions: str
) -> str:
    insp_lines = [
        f"[{i}] score={ts_score(e['rating']):.3f}\n{e['txt']}"
        for i, e in enumerate(inspirations, 1)
    ]
    return _MUT_PROMPT_TEMPLATE.format(
        goal=goal,
        instructions=instructions,
        p_score=ts_score(parent["rating"]),
        p_text=parent["txt"],
        insp_text="\n\n".join(insp_lines) or "(none)",
    )


# ╭──────────────────────────────────────────────────────────╮
# │ 8.  Main evolutionary driver                             │
# ╰──────────────────────────────────────────────────────────╯


def run(cfg: Config):
    # ─ setup logging & PRNG ─────────────────────────────────────────
    setup_logging(level=logging.INFO)
    random.seed(42)

    # islands & judge
    islands = [MixedArchive(cfg.axes, cfg.k_top) for _ in range(cfg.num_islands)]
    judge = MultiMetricJudge(cfg.judge_model, cfg.metrics)

    # seed
    seed_txt = Path(cfg.seed_path).read_text()
    seed_desc = {"lang": "txt", "len": len(seed_txt)}
    seed_elite = {
        "txt": seed_txt,
        "rating": {m: judge.envs[m].create_rating() for m in cfg.metrics},
        "age": 0,
    }
    for arc in islands:
        arc.add(
            seed_desc, copy.deepcopy(seed_elite)
        )  # deepcopy not needed: ratings are new per env

    # ─ Rich progress bar setup ─────────────────────────────────────
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("• best {task.fields[best]:.3f}"),
        TextColumn("• empty {task.fields[empty]}"),
        TimeElapsedColumn(),
        transient=True,
    )

    with progress:
        task = progress.add_task(
            "evolving", total=cfg.iterations, best=0.0, empty=islands[0].empty_cells
        )

        # ─ main loop ────────────────────────────────────────────────
        for it in range(cfg.iterations):
            isl_idx = random.randrange(cfg.num_islands)
            arc = islands[isl_idx]

            parent = arc.random_elite(cfg.youth_bias)
            # inspirations from same island, exclude parent
            cand = [e for b in arc.cell.values() for e in b if e is not parent]
            inspirations = random.sample(cand, k=min(3, len(cand)))

            # mutate
            mdl, temp = pick_model(cfg.llm_ensemble)
            prompt = build_mut_prompt(
                parent,
                inspirations,
                cfg.mutation_prompt_goal,
                cfg.mutation_prompt_instructions,
            )
            reply = llm_call(
                model=mdl, prompt=prompt, temperature=temp, response_format=None
            )

            thinking, diff_content = parse_llm_mutation_response(reply, log_mut)

            if thinking:
                log_mut.info("Mutator rationale (iter %d):\n%s", it, thinking)

            if not diff_content:
                log_mut.warning(
                    "No diff content found in LLM reply, skipping mutation for iter %d.",
                    it,
                )
                continue

            log_mut.debug("Diff block (iter %d):\n%s", it, diff_content)

            for blk in split_blocks(diff_content)[: cfg.n_diffs]:
                child_txt = apply_patch(parent["txt"], blk)
                if child_txt == parent["txt"]:
                    continue
                child = {
                    "txt": child_txt,
                    "rating": {m: judge.envs[m].create_rating() for m in cfg.metrics},
                    "age": it,
                }
                desc_child = {"lang": "txt", "len": len(child_txt)}
                child["cell_key"] = arc._key(desc_child)

                # rank & update ratings (parent + insp + child)
                group = [parent] + inspirations + [child]
                judge.rank_and_rate(group)
                # resort buckets whose ratings changed
                for e in group:
                    arc.resort_elite(e)

                # add child to archive
                arc.add(desc_child, child)

            # migration
            if (it + 1) % cfg.migration_every == 0:
                for idx, src in enumerate(islands):
                    migrants = random.sample(
                        [e for b in src.cell.values() for e in b],
                        k=min(
                            cfg.migrants_per_island,
                            sum(len(b) for b in src.cell.values()),
                        ),
                    )
                    dst = islands[(idx + 1) % cfg.num_islands]
                    for e in migrants:
                        desc = {"lang": "txt", "len": len(e["txt"])}
                        dst.add(desc, copy.deepcopy(e))

            # optional rare global sparring
            if (it + 1) % cfg.sparring_every == 0:
                elite_to_island_map = {}
                pool = []
                # Corrected variable name to sparring_arc
                for isl_idx, sparring_arc in enumerate(islands):
                    # Iterate over cell keys, then check if bucket exists and is non-empty
                    # Using list(sparring_arc.cell.keys()) to avoid issues if cell modified during iteration (though not expected here)
                    for cell_key_in_arc in list(sparring_arc.cell.keys()):
                        bucket = sparring_arc.cell.get(cell_key_in_arc)
                        if bucket:  # Ensure bucket exists and is not empty
                            chosen_elite = random.choice(bucket)
                            pool.append(chosen_elite)
                            # Map elite object's id to its source island (sparring_arc)
                            elite_to_island_map[id(chosen_elite)] = sparring_arc

                if len(pool) > 1:
                    logging.info("Global sparring with %d elites in pool.", len(pool))
                    judge.rank_and_rate(pool)  # Modifies elites in 'pool' in-place

                    resorted_count = 0
                    for e_updated in pool:
                        original_island = elite_to_island_map.get(id(e_updated))
                        if original_island:
                            original_island.resort_elite(e_updated)
                            resorted_count += 1
                        else:
                            # This case should ideally not happen if map is populated correctly.
                            logging.warning(
                                "Could not find original island for globally sparred elite (id: %s, text snippet: '%s'). Skipping resort for this elite.",
                                id(e_updated),
                                e_updated.get("txt", "N/A")[
                                    :50
                                ],  # Provide default for get
                            )
                    logging.info(
                        "Global sparring: %d elites resorted in their original islands.",
                        resorted_count,
                    )

            # logging & progress update
            best_global = max(
                (arc.best for arc in islands), key=lambda e: ts_score(e["rating"])
            )
            best_score = ts_score(best_global["rating"])
            if (it + 1) % cfg.log_every == 0:
                metric_details_parts = []
                for metric_name in cfg.metrics:  # Iterate through configured metrics
                    rating_obj = best_global["rating"].get(metric_name)
                    if rating_obj:
                        metric_details_parts.append(
                            f"{metric_name}(μ={rating_obj.mu:.2f}, σ={rating_obj.sigma:.2f})"
                        )
                    else:
                        # This case should ideally not happen if ensure_ratings works,
                        # but good for robustness if a metric was somehow missed for the best elite.
                        metric_details_parts.append(f"{metric_name}(N/A)")

                metrics_summary_str = " | ".join(metric_details_parts)
                logging.info(
                    "it %d  best_score %.3f | %s",
                    it + 1,
                    best_score,
                    metrics_summary_str
                    if metrics_summary_str
                    else "No metric details available",
                )

            # The single, definitive progress update at the end of each iteration
            progress.update(
                task,
                advance=1,
                best=best_score,
                empty=islands[0].empty_cells,
            )

    # save final best
    best_final = max((arc.best for arc in islands), key=lambda e: ts_score(e["rating"]))
    Path("best.txt").write_text(best_final["txt"])
    logging.info("DONE – best saved with score %.3f", ts_score(best_final["rating"]))


# ╭──────────────────────────────────────────────────────────╮
# │ 9.  CLI                                                  │
# ╰──────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON/TOML config path")
    cfg = load_cfg(ap.parse_args().config)
    run(cfg)
