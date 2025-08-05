"""
This module contains functions for parsing responses from Large Language Models (LLMs).
"""

import logging
import re
from typing import Dict, List, Tuple

from lxml import html


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
            if not thinking_match and re.search(r"<<<<<<< SEARCH", raw_response):
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
