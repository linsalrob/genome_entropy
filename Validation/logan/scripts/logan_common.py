"""Shared paths, state handling and provenance for the LOGAN pilot (#102).

Every stage records progress in one small JSON file per accession under
``$LOGAN_ROOT/state/`` and per chunk under ``$LOGAN_ROOT/manifests/chunks/``.
Writes go through a temporary file and ``os.replace`` so a job killed
mid-write leaves the previous state rather than a truncated file.

ACCESSION STATES, in order. A stage refuses to run unless the previous state
has been reached, and skips work whose state is already at or past its own:

    selected -> downloaded -> validated -> prodigal_done -> orfs_called
             -> genome_entropy_done -> cache_validated -> uploaded
             -> remote_verified -> complete

``orfs_called`` is the only addition to the list in issue #102: ORF calling
and Prodigal matching are CPU work done before encoding, and keeping them as
their own state lets the GPU stage start from a known-good ORF table.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

LOGAN_RELEASE = "v1.2"
LOGAN_CONTIG_URL = "https://s3.amazonaws.com/logan-pub/c/{acc}/{acc}.contigs.fa.zst"
MODEL_NAME = "gbouras13/modernprost-50M"
CACHE_VERSION = "v1"
MIN_CONTIG_NT = 120          # LOGAN's Prodigal filter
MIN_ORF_AA = 30              # genome_entropy run default (--min-aa)
GENETIC_CODE = 11
PRODIGAL_ARGS = ["-q", "-p", "meta"]
# LOGAN_REMOTE overrides the destination, e.g. for a smoke test.
REMOTE_ROOT = os.environ.get("LOGAN_REMOTE", "genome_entropy:General/LOGAN")

STATES = (
    "selected",
    "downloaded",
    "validated",
    "prodigal_done",
    "orfs_called",
    "genome_entropy_done",
    "cache_validated",
    "uploaded",
    "remote_verified",
    "complete",
)

# Cache row schema. start/end are 0-based half-open on the contig's forward
# axis and span the whole ORF including its stop codon when there is one; they
# come from genome_entropy.io.genbank.normalise_orf_interval and nothing
# downstream should transform them again.
CACHE_COLUMNS = (
    "accession",
    "contig",
    "orf_id",
    "start",
    "end",
    "strand",
    "contig_len",
    "stop",
    "aa_sequence",
    "three_di",
    "twelve_state",
)


def root() -> Path:
    return Path(os.environ.get(
        "LOGAN_ROOT",
        f"/scratch/{os.environ.get('PAWSEY_PROJECT', 'pawsey1018')}/"
        f"{os.environ.get('USER', 'user')}/Logan"))


def work_dir(acc: str) -> Path:
    return root() / "work" / acc


def state_path(acc: str) -> Path:
    return root() / "state" / f"{acc}.json"


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_state(acc: str) -> Dict[str, Any]:
    p = state_path(acc)
    if not p.exists():
        return {"accession": acc, "state": None, "history": []}
    with open(p) as fh:
        return json.load(fh)


def state_index(state: Optional[str]) -> int:
    return -1 if state is None else STATES.index(state)


def reached(st: Dict[str, Any], state: str) -> bool:
    return state_index(st.get("state")) >= STATES.index(state)


def advance(acc: str, state: str, **info: Any) -> Dict[str, Any]:
    """Move an accession forward to ``state``, merging ``info`` into it.

    Moving backwards is refused: a rerun that wants to redo a step must say
    so explicitly with ``reset``.
    """
    st = load_state(acc)
    if state_index(state) < state_index(st.get("state")):
        raise RuntimeError(f"{acc}: refusing to move state back from "
                           f"{st.get('state')} to {state}")
    st["state"] = state
    st.setdefault("history", []).append({"state": state, "time": now()})
    st.setdefault("info", {}).update(info)
    atomic_write_json(state_path(acc), st)
    return st


def reset(acc: str, state: Optional[str], reason: str) -> None:
    st = load_state(acc)
    st.setdefault("history", []).append(
        {"state": state, "time": now(), "reset_from": st.get("state"),
         "reason": reason})
    st["state"] = state
    atomic_write_json(state_path(acc), st)


def sha256(path: Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tool_versions() -> Dict[str, str]:
    """Versions that every manifest records, read from the environment."""
    out: Dict[str, str] = {}
    try:
        from genome_entropy import __version__ as gev
        out["genome_entropy_version"] = gev
    except Exception:  # pragma: no cover - reported, not fatal here
        out["genome_entropy_version"] = "unknown"
    prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    commit = prefix / "GENOME_ENTROPY_COMMIT"
    dirty = prefix / "GENOME_ENTROPY_SRC_DIRTY"
    out["genome_entropy_commit"] = commit.read_text().strip() if commit.exists() else "unknown"
    out["genome_entropy_src_dirty"] = bool(dirty.exists() and dirty.read_text().strip())
    for name, cmd in (("prodigal", ["prodigal", "-v"]),
                      ("get_orfs", ["get_orfs", "-v"]),
                      ("zstd", ["zstd", "--version"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            txt = (r.stdout + r.stderr).strip().splitlines()
            out[name] = next((t.strip() for t in txt if t.strip()), "unknown")
        except Exception:
            out[name] = "unavailable"
    return out
