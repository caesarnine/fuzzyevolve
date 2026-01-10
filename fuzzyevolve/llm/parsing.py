from __future__ import annotations

import logging
import re
from typing import Sequence


def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(
        rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip()


def _parse_id_list(raw: str) -> list[int]:
    if not raw:
        return []
    cleaned = raw.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    return [int(value) for value in re.findall(r"-?\d+", cleaned)]


def parse_judge_response(
    raw_xml_string: str, metrics: Sequence[str], logger: logging.Logger
) -> tuple[str | None, dict[str, list[int]]]:
    thinking_text = _extract_tag(raw_xml_string, "thinking")
    output_text = _extract_tag(raw_xml_string, "output")
    search_scope = output_text if output_text is not None else raw_xml_string

    if output_text is None:
        logger.error("LLM Judge: <output> tag not found, scanning full response.")

    rankings: dict[str, list[int]] = {}
    for metric_name in metrics:
        tag_name = metric_name.lower()
        metric_body = _extract_tag(search_scope, tag_name)
        if metric_body is None:
            logger.warning(
                "LLM Judge: Metric tag <%s> not found for '%s'.",
                tag_name,
                metric_name,
            )
            continue
        ids = _parse_id_list(metric_body)
        if not ids:
            logger.warning(
                "LLM Judge: No IDs parsed for metric '%s' from '%s'.",
                metric_name,
                metric_body,
            )
            continue
        rankings[metric_name] = ids

    if not rankings:
        logger.error("LLM Judge: No valid rankings extracted from response.")

    return thinking_text, rankings


def parse_mutation_response(
    raw_response: str, logger: logging.Logger
) -> tuple[str | None, str | None]:
    thinking = _extract_tag(raw_response, "thinking")
    diffs = _extract_tag(raw_response, "diffs")

    if diffs is None:
        if re.search(r"<<<<<<<\s+SEARCH", raw_response):
            logger.warning(
                "No <diffs> tag found, but diff-like content detected. Using raw response."
            )
            diffs = raw_response
        else:
            logger.warning("Mutation LLM: No <diffs> block found in response.")

    if thinking is None:
        logger.warning("Mutation LLM: No <thinking> block found in response.")

    return thinking, diffs
