#!/usr/bin/env python3
"""Execute manuscript notebooks and check they reproduce the committed images.

Each notebook runs in a fresh kernel inside a temporary copy of manuscript/,
so a check never writes into the repository. Every file the notebook writes
under images/ is compared, byte for byte, with the committed file of the same
name. Every committed image named after the notebook (images/<stem>.*) must
also be rewritten by the run, so a notebook that silently stops producing one
of its outputs fails. This works whether a notebook saves through
msfig.save() or directly to ../images/.

  run_notebooks.py                          # check every notebook
  run_notebooks.py notebooks/figL02_*.ipynb # check some
  run_notebooks.py --update notebooks/fig01_x.ipynb   # run in place, write images/

Exit status is non-zero if any notebook fails to execute, writes no image,
writes an image that differs from or is missing in manuscript/images/, or
does not rewrite a committed images/<stem>.* file.
"""
import argparse
import glob
import hashlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

MS = Path(__file__).resolve().parent.parent
NOTEBOOKS = MS / "notebooks"
IMAGES = MS / "images"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def execute(nb_path, timeout):
    nb = nbformat.read(nb_path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(nb_path.parent)}}).execute()


def check(nb_rel, timeout):
    name = nb_rel.name
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "manuscript"
        shutil.copytree(MS, copy, ignore=shutil.ignore_patterns(".ipynb_checkpoints"))
        # Age every copied image so that only files the notebook writes are
        # newer than `start`, whatever timestamps the checkout carried.
        for p in (copy / "images").rglob("*"):
            if p.is_file():
                os.utime(p, (0, 0))
        start = time.time() - 1
        os.environ.pop("MS_IMAGES_DIR", None)
        try:
            execute(copy / nb_rel, timeout)
        except Exception as e:  # report and carry on with the other notebooks
            print(f"FAIL  {name}: execution error: {str(e).splitlines()[0]}")
            return False
        written = sorted(p for p in (copy / "images").rglob("*")
                         if p.is_file() and p.stat().st_mtime >= start)
        if not written:
            print(f"FAIL  {name}: wrote no images")
            return False
        ok = True
        for p in written:
            rel = p.relative_to(copy / "images")
            ref = IMAGES / rel
            if not ref.exists():
                print(f"FAIL  {name}: images/{rel} is not committed")
                ok = False
            elif sha(p) != sha(ref):
                print(f"FAIL  {name}: images/{rel} differs "
                      f"(new {sha(p)}, committed {sha(ref)})")
                ok = False
            else:
                print(f"ok    {name}: images/{rel} {sha(p)}")
        rewritten = {p.relative_to(copy / "images") for p in written}
        for ref in sorted(IMAGES.glob(f"{nb_rel.stem}.*")):
            if ref.relative_to(IMAGES) not in rewritten:
                print(f"FAIL  {name}: images/{ref.name} is committed but the "
                      f"notebook did not write it")
                ok = False
        return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("notebooks", nargs="*", type=Path)
    ap.add_argument("--update", action="store_true",
                    help="execute in place, writing into manuscript/images/")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    # Accept paths and wildcards relative to the cwd or to manuscript/
    # (notebooks/figL02_*.ipynb from the repository root).
    nbs = []
    for p in args.notebooks:
        hits = glob.glob(str(p)) or glob.glob(str(MS / p))
        if not hits:
            ap.error(f"no notebook matches {p}")
        nbs += [Path(h).resolve() for h in sorted(hits)]
    nbs = nbs or sorted(
        p for p in NOTEBOOKS.glob("*.ipynb") if not p.name.startswith("_"))
    if not nbs:
        print("no notebooks found")
        return 0
    if args.update:
        for nb in nbs:
            execute(nb, args.timeout)
            print(f"executed {nb.name}")
        return 0
    results = [check(nb.relative_to(MS), args.timeout) for nb in nbs]
    print(f"{sum(results)}/{len(results)} notebooks reproduce their images")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
