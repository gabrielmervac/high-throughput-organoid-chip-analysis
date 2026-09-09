"""Headless end-to-end pipeline run: ND2 -> segment -> Z-link -> track ->
metrics -> Excel. Useful for verification and batch processing without the GUI.

Usage:
  python run_headless.py MODEL_PATH ND2_PATH [--positions 0 1] [--timepoints 0 1]
                          [--out OUT.xlsx]
"""
import _env  # noqa: F401
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

from nd2_reader import ND2Reader          # noqa: E402
from segment import load_model, SegParams  # noqa: E402
from analysis import analyze_position, build_records  # noqa: E402
from excel_export import export_records    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("nd2")
    ap.add_argument("--positions", type=int, nargs="*", default=None)
    ap.add_argument("--timepoints", type=int, nargs="*", default=None)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = load_model(args.model)
    seg = SegParams()
    all_rows = []
    with ND2Reader(args.nd2) as r:
        info = r.info
        positions = args.positions if args.positions is not None else list(range(info.nP))
        print(f"Experiment {info.experiment}: {len(positions)} position(s), "
              f"channels={info.channel_names}")
        for p in positions:
            res = analyze_position(
                r, model, p, timepoints=args.timepoints, seg_params=seg,
                iou_threshold=args.iou,
                progress=lambda m, f: print(f"  [{f*100:5.1f}%] {m}"))
            rows = build_records(r, res)
            all_rows.extend(rows)
            n_tracks = len({row["track_id"] for row in rows})
            print(f"  pos {p}: {len(rows)} rows, {n_tracks} tracked organoids")

    out = args.out or str(_env.OUTPUTS_DIR / f"{info.experiment}_analysis.xlsx")
    export_records(all_rows, out)
    print(f"Wrote {out}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
