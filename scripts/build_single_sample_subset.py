#!/usr/bin/env python3
"""Build a one-sample GARF HDF5 subset for controlled eval.

Thin CLI wrapper around the builder used by garf_matched_diagnostic.py so that
reference (known-good) objects can be evaluated in isolation with the same
single-sample pipeline as Juglet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from garf_matched_diagnostic import (
    build_garf_subset_hdf5,
    pick_sample_with_parts,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="Source HDF5.")
    ap.add_argument("--split-key", required=True, help="data_split/<key>/val key.")
    ap.add_argument("--pieces", type=int, required=True, help="Match this part count.")
    ap.add_argument("--out", type=Path, required=True, help="Output subset HDF5.")
    ap.add_argument("--sample", default=None, help="Explicit sample name (overrides --pieces pick).")
    args = ap.parse_args()

    with h5py.File(args.source, "r") as h5:
        sample = args.sample or pick_sample_with_parts(h5, args.split_key, args.pieces)
    build_garf_subset_hdf5(args.source, sample, args.out)
    print(f"sample={sample}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
