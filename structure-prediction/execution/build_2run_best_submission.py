"""
Build a submission by selecting the best iptm model per ligand across
two Boltz prediction runs:

  Run 1 (base):  6TFIB H12-free, 50 diffusion samples
  Run 2 (cross): FRAGS_184_4NY9_50, 50 diffusion samples

For every ligand, all confidence JSONs in both runs are scanned and the
model with the highest iptm is selected. No threshold is applied — the
better run wins outright for each ligand.

Expected score (based on incremental evidence): ~0.512
  - h12free50_best alone:               0.5094
  - isolated 4NY9_50 contribution:     +0.0026 (from bestall_14swap experiment)

Output: outputs/boltz_h12free50_4ny9_2run_submission.zip

Usage:
    python execution/build_2run_best_submission.py --test
    python execution/build_2run_best_submission.py
    python execution/build_2run_best_submission.py --run1 <path> --run2 <path>
"""
import argparse
import json
import logging
import zipfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

RUN1_PREDS = Path("E:/OpenAdmet/Boltz/6TFIB_YASARA_H12FREE_SCREEN_50_OUT/boltz_results_6TFIB_YASARA_H12FREE_SCREEN/predictions")
RUN2_PREDS = Path("E:/OpenAdmet/Boltz/FRAGS_184_4NY9_50_OUT/boltz_results_FRAGS_184_4NY9/predictions")

DEFAULT_OUTPUT = ROOT / "outputs" / "boltz_h12free50_4ny9_2run_submission.zip"

HEADER = (
    "HEADER    NUCLEAR RECEPTOR                       16-APR-24   PXR \n"
    "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
)


def best_model_in_run(preds_dir: Path, lig_id: str) -> tuple[float, int]:
    """Return (best_iptm, best_model_idx) for lig_id in preds_dir, or (-1, -1) if absent."""
    lig_dir = preds_dir / lig_id
    if not lig_dir.exists():
        return -1.0, -1

    best_iptm, best_idx = -1.0, -1
    for conf_json in lig_dir.glob(f"confidence_{lig_id}_model_*.json"):
        try:
            idx = int(conf_json.stem.rsplit("_model_", 1)[-1])
            iptm = float(json.loads(conf_json.read_text()).get("iptm", -1))
            if iptm > best_iptm:
                best_iptm, best_idx = iptm, idx
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return best_iptm, best_idx


def format_pdb(pdb_path: Path) -> str:
    text = pdb_path.read_text(encoding="utf-8", errors="replace")
    lines = [l for l in text.splitlines(keepends=True)
             if not l.startswith("HEADER") and not l.startswith("CRYST1")]
    return HEADER + "".join(lines)


def build(run1: Path, run2: Path, out_zip: Path, threshold: float = 0.0) -> dict:
    """
    threshold: minimum Δiptm (iptm2 - iptm1) required to swap to run2.
               0.0 = swap whenever run2 is strictly better (no threshold).
               0.010 = only swap if run2 beats run1 by at least 0.010.
    """
    lids_1 = {d.name for d in run1.iterdir() if d.is_dir()}
    lids_2 = {d.name for d in run2.iterdir() if d.is_dir()}
    all_lids = sorted(lids_1 | lids_2)
    log.info("Run1: %d  Run2: %d  Union: %d ligands  threshold=%.3f",
             len(lids_1), len(lids_2), len(all_lids), threshold)

    counts = {"run1": 0, "run2": 0, "failed": 0}
    iptm_vals: list[float] = []
    swapped: list[tuple[str, float, float]] = []  # (lid, iptm1, iptm2) when run2 wins

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for lid in all_lids:
            iptm1, idx1 = best_model_in_run(run1, lid)
            iptm2, idx2 = best_model_in_run(run2, lid)

            if iptm2 - iptm1 > threshold:
                src, idx, iptm = run2, idx2, iptm2
                counts["run2"] += 1
                swapped.append((lid, iptm1, iptm2))
            elif iptm1 >= 0:
                src, idx, iptm = run1, idx1, iptm1
                counts["run1"] += 1
            else:
                log.warning("%s: not found in either run — skipping", lid)
                counts["failed"] += 1
                continue

            pdb_path = src / lid / f"{lid}_model_{idx}.pdb"
            if not pdb_path.exists():
                log.warning("%s: PDB missing at %s — skipping", lid, pdb_path)
                counts["failed"] += 1
                continue

            zf.writestr(f"{lid}.pdb", format_pdb(pdb_path))
            iptm_vals.append(iptm)

    total = counts["run1"] + counts["run2"]
    log.info("Written: %s  (%d ligands)", out_zip.name, total)
    log.info("  From run1 (H12-free50): %d  |  From run2 (4NY9_50): %d  |  Failed: %d",
             counts["run1"], counts["run2"], counts["failed"])

    if iptm_vals:
        log.info("  iptm — mean=%.4f  median=%.4f  min=%.4f  max=%.4f",
                 np.mean(iptm_vals), np.median(iptm_vals),
                 np.min(iptm_vals), np.max(iptm_vals))

    if swapped:
        log.info("  %d ligands swapped to 4NY9_50 (iptm2 > iptm1):", len(swapped))
        for lid, i1, i2 in sorted(swapped, key=lambda x: x[2] - x[1], reverse=True):
            log.info("    %-14s  h12free=%.4f  4ny9=%.4f  delta=+%.4f", lid, i1, i2, i2 - i1)

    return {**counts, "iptm_mean": float(np.mean(iptm_vals)) if iptm_vals else 0.0,
            "n_swapped_to_run2": len(swapped)}


def run_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        r1 = tmp / "run1"
        r2 = tmp / "run2"

        # Three ligands: x00001 run1 wins, x00002 run2 wins, x00003 only in run2
        for lig, run, iptm_val, model_idx in [
            ("x00001-1", r1, 0.95, 3),
            ("x00001-1", r2, 0.90, 1),
            ("x00002-1", r1, 0.88, 0),
            ("x00002-1", r2, 0.93, 2),
            ("x00003-1", r2, 0.91, 0),
        ]:
            ld = run / lig
            ld.mkdir(parents=True, exist_ok=True)
            pdb = (f"HETATM    1  C1  LIG B   1       0.000   0.000   0.000  1.00 80.00           C  \nEND\n")
            (ld / f"{lig}_model_{model_idx}.pdb").write_text(pdb)
            conf = {"iptm": iptm_val, "ptm": 0.9}
            (ld / f"confidence_{lig}_model_{model_idx}.json").write_text(json.dumps(conf))

        out = tmp / "test.zip"
        stats = build(r1, r2, out)

        assert stats["run1"] == 1, f"Expected 1 from run1, got {stats['run1']}"
        assert stats["run2"] == 2, f"Expected 2 from run2, got {stats['run2']}"
        assert stats["failed"] == 0

        with zipfile.ZipFile(out) as zf:
            assert set(zf.namelist()) == {"x00001-1.pdb", "x00002-1.pdb", "x00003-1.pdb"}

        log.info("Smoke test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run1", type=Path, default=RUN1_PREDS,
                        help="H12-free50 predictions dir")
    parser.add_argument("--run2", type=Path, default=RUN2_PREDS,
                        help="4NY9_50 predictions dir")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Min delta-iptm to swap from run1 to run2 (default: 0.0 = any improvement)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        run_test()
        import sys; sys.exit(0)

    for name, p in [("run1", args.run1), ("run2", args.run2)]:
        if not p.exists():
            raise FileNotFoundError(f"{name} predictions dir not found: {p}")

    build(args.run1, args.run2, args.output, threshold=args.threshold)
