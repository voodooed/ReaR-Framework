#!/usr/bin/env python
"""Print the material/permittivity database, so it can be checked by eye.

    python tools/inspect_materials.py
    python tools/inspect_materials.py --label-space carla --freq 77

The database is meant to be auditable, which means someone has to be able to sit
down with Recommendation ITU-R P.2040-4 open and compare it row by row. This is
that view: every material, the constants it resolves to at the radar frequency,
whether that frequency is inside the band the coefficients were fitted over, and
where the number came from.

It reads exactly what the pipeline reads -- `rear/materials/data/*.yaml` and
`configs/materials/*.yaml` -- so a discrepancy between this output and what a
head was trained on is not possible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from rear.materials import (DESCRIPTOR_NAMES, MATERIAL_DB,  # noqa: E402
                            MATERIAL_FEATURE_NAMES, available_label_spaces,
                            load_label_space, wavelength_m)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", type=float, default=None,
                    help="evaluation frequency in GHz (default: the database's own)")
    ap.add_argument("--label-space", default=None,
                    help=f"also print this space's class->material map "
                         f"({', '.join(available_label_spaces())})")
    args = ap.parse_args()

    f = args.freq or MATERIAL_DB.default_frequency_ghz
    lam = wavelength_m(f)

    print("=" * 96)
    print(f"MATERIAL DATABASE  --  {MATERIAL_DB.itu_version}")
    print("=" * 96)
    print(f"  source        : {MATERIAL_DB.itu_source}")
    print(f"  files         : {MATERIAL_DB.itu_path.relative_to(REPO_ROOT)}, "
          f"{MATERIAL_DB.materials_path.relative_to(REPO_ROOT)}")
    print(f"  frequency     : {f:g} GHz   (lambda = {lam * 1e3:.3f} mm, "
          f"lambda/16 = {lam * 1e3 / 16:.3f} mm)")
    print(f"  materials     : {len(MATERIAL_DB)}  (column order is a contract)\n")

    print(f"{'#':>2} {'material':<14}{'eps_r':>8}{'sigma S/m':>12}{'|Gamma|':>9}"
          f"{'dB':>8}{'h_rms mm':>10}{'rho_s':>8}  {'conf':<7}{'band GHz':<16}")
    print("-" * 96)
    for i, r in enumerate(MATERIAL_DB.summary_rows(f)):
        lo, hi = r["band"]
        band = (f"{lo:g}-{hi:g}" if r["in_band"]
                else (f"{lo:g}-{hi:g} EXTRAP" if lo != hi else "not in P.2040"))
        print(f"{i:>2} {r['material']:<14}{r['eps_real']:>8.2f}{r['sigma']:>12.4g}"
              f"{r['gamma_mag']:>9.3f}{r['reflectivity_db']:>8.2f}"
              f"{r['h_rms_mm']:>10.3g}{r['specular_fraction']:>8.3f}  "
              f"{r['confidence']:<7}{band:<16}")
    print()

    weak = [r["material"] for r in MATERIAL_DB.summary_rows(f)
            if r["confidence"] == "low"]
    print(f"  LOW-CONFIDENCE ({len(weak)}): {', '.join(weak)}")
    print("  These carry into the feature block as `frac_low_confidence`, so an")
    print("  ablation that drops every uncertain constant is a filter over an")
    print("  existing column rather than a re-extraction. Vegetation is the weakest")
    print("  entry in the file AND one of the largest background classes -- that")
    print("  combination is a standing risk, not a resolved question.\n")

    print(f"  DESCRIPTOR TABLE columns: {', '.join(DESCRIPTOR_NAMES)}")
    print(f"  FEATURE BLOCK columns   : {', '.join(MATERIAL_FEATURE_NAMES)}\n")

    if args.label_space:
        ls = load_label_space(args.label_space)
        M = ls.mixture_matrix()
        print("=" * 96)
        print(f"LABEL SPACE  --  {ls.name}  ({ls.description})")
        print("=" * 96)
        print(f"  file      : {ls.path.relative_to(REPO_ROOT)}")
        print(f"  classes   : {ls.n_classes} -> {len(MATERIAL_DB)} materials\n")
        print(f"{'id':>3} {'class':<16} mixture")
        print("-" * 96)
        for cid in ls.ids:
            mix = ls.mixtures[cid]
            desc = (", ".join(f"{k} {v:g}" for k, v in mix.items())
                    if mix else "(empty -- returns nothing to the radar)")
            print(f"{cid:>3} {ls.class_names[cid]:<16} {desc}")
        empty = [ls.class_names[c] for c in ls.ids if not ls.mixtures[c]]
        print(f"\n  empty mixtures ({len(empty)}): {', '.join(empty) or 'none'}")
        print("  An empty mixture is a VALUE, never a gap to fill with a uniform")
        print("  prior: inventing reflectivity where the sensor reports none is the")
        print("  one failure mode invisible in every summary statistic.\n")
        # What the head would actually see for a pure instance of each class.
        block_names = MATERIAL_FEATURE_NAMES
        from rear.materials.features import features_from_composition
        F = features_from_composition(M, np.ones(len(M)), freq_ghz=f)
        print(f"{'class':<16}" + "".join(f"{n[:9]:>10}" for n in block_names[:6]))
        print("-" * 96)
        for cid in ls.ids:
            print(f"{ls.class_names[cid]:<16}"
                  + "".join(f"{F[cid, j]:>10.3f}" for j in range(6)))
        print("\n  (first 6 of the 10 feature columns; a pure instance of each class)")


if __name__ == "__main__":
    main()
