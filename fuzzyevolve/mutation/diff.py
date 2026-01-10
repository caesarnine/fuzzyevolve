from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log_mut = logging.getLogger("mutation")

_DIFF_BLOCK_RE = re.compile(r"<<<<<<<\s+SEARCH.*?>>>>>>>\s+REPLACE", re.DOTALL)


@dataclass(frozen=True, slots=True)
class DiffBlock:
    search: str
    replace: str
    raw: str

    def apply(self, text: str) -> str | None:
        idx = text.find(self.search)
        if idx == -1:
            return None
        return text.replace(self.search, self.replace, 1)


def parse_block(block: str, logger: logging.Logger | None = None) -> DiffBlock | None:
    logger = logger or log_mut
    try:
        _, rest = block.split("<<<<<<< SEARCH", 1)
        search, rest = rest.split("=======", 1)
        replace, _ = rest.split(">>>>>>> REPLACE", 1)
    except ValueError:
        logger.warning(
            "Malformed diff block, cannot apply patch. Block (first 200 chars): '%s'",
            block[:200],
        )
        return None
    return DiffBlock(
        search=search.strip("\n\r"),
        replace=replace.strip("\n\r"),
        raw=block,
    )


def extract_blocks(raw: str, logger: logging.Logger | None = None) -> list[DiffBlock]:
    logger = logger or log_mut
    blocks: list[DiffBlock] = []
    for match in _DIFF_BLOCK_RE.finditer(raw):
        parsed = parse_block(match.group(0), logger=logger)
        if parsed is not None:
            blocks.append(parsed)
    if not blocks and raw.strip():
        logger.warning("No diff blocks found in response.")
    return blocks
