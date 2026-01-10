import re
import logging
from typing import List

log_mut = logging.getLogger("mutation")

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
