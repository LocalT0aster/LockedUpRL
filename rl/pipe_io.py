"""
Lightweight pipe IO utilities.

Responsible only for:
- Reading observations sent by the Godot game (metadata line + grid rows).
- Writing back an action string.
- Running a loop with a user-provided decision callback.

Decision logic belongs elsewhere (e.g., escape_game.py).
"""

from __future__ import annotations

import logging
import sys
from typing import Callable, Dict, Iterable, List, Tuple

try:
    from rl.logging_utils import setup_logging  # type: ignore
except Exception:  # pragma: no cover - fallback if import fails
    setup_logging = None

# Ensure logging is configured even if escape_game didn't set it up yet.
if not logging.getLogger().handlers and setup_logging is not None:
    setup_logging()

logger = logging.getLogger(__name__)


Observation = Tuple[List[str], Dict[str, str]]
DecisionFn = Callable[[List[str], Dict[str, str]], str]


def read_observation(stream=None) -> Observation | Tuple[None, None]:
    """
    Reads one observation from the pipe.
    Format:
      - Optional first line: metadata
          * Legacy: key=value tokens separated by spaces
          * New: space-delimited tokens, e.g. "role C2 pos 10 5"
      - Then one or more grid rows (space-separated cell tokens)
      - Blank line terminates the observation
    Returns (rows, meta) or (None, None) on EOF.
    """
    stream = stream or sys.stdin
    meta: Dict[str, str] = {}
    rows: List[str] = []
    while True:
        line = stream.readline()
        if line == "":
            break  # EOF
        stripped = line.strip()
        if stripped == "":
            if rows:
                break
            continue
        if not rows:
            parts = stripped.split()
            if "=" in stripped and all("=" in part for part in parts):
                for part in parts:
                    key, _, value = part.partition("=")
                    if key:
                        meta[key.strip().lower()] = value.strip()
                continue
            # New space-delimited meta format: role <token> pos <x> <y>
            if parts and parts[0].lower() == "role":
                role_val = parts[1] if len(parts) > 1 else ""
                meta["role"] = role_val
                if "pos" in (p.lower() for p in parts):
                    try:
                        pos_idx = [p.lower() for p in parts].index("pos")
                        if len(parts) > pos_idx + 2:
                            px = parts[pos_idx + 1]
                            py = parts[pos_idx + 2]
                            meta["pos"] = f"{px},{py}"
                            meta["pos_x"] = px
                            meta["pos_y"] = py
                    except ValueError:
                        pass
                continue
        rows.append(stripped)
    if not rows and not meta:
        return None, None
    return rows, meta


def write_action(action: str, out=None) -> None:
    out = out or sys.stdout
    print(action, file=out, flush=True)


def run_loop(decide: DecisionFn, stream=None, out=None) -> None:
    """
    Repeatedly reads observations and delegates to `decide(rows, meta)` for an action string.
    """
    stream = stream or sys.stdin
    out = out or sys.stdout
    while True:
        rows, meta = read_observation(stream)
        if rows is None:
            logger.info("[pipe] EOF received; stopping loop")
            break
        logger.info("[pipe] recv meta=%s rows=%d", meta or {}, len(rows))
        logger.info("[pipe] grid:\n%s", "\n".join(rows))
        try:
            action = decide(rows, meta or {})
        except Exception:
            logger.exception("[pipe] decide() raised; exiting pipe loop")
            break
        if action is None:
            action = "stay"
        logger.info("[pipe] action=%s meta=%s rows=%d", action, meta or {}, len(rows))
        write_action(action, out)


def normalize_grid_rows(rows: Iterable[str]) -> List[List[str]]:
    """
    Splits each row into tokens, strips whitespace, keeps them verbatim.
    """
    grid: List[List[str]] = []
    for raw in rows:
        toks = [tok.strip() for tok in raw.split() if tok.strip()]
        if toks:
            grid.append(toks)
    return grid
