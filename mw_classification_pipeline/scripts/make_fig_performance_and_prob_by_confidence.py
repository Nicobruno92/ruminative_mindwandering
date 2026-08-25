#!/usr/bin/env python
"""
Render performance_and_probabilities_by_confidence.{svg,png,csv} on its own,
without rerunning make_fig_prob_vs_dim_ws_loso.py's full main() (which also
rebuilds ~30 unrelated per-dimension figures).

The figure itself is built by
``make_fig_prob_vs_dim_ws_loso.py::build_performance_and_prob_by_confidence_combined``
and is also wired into that script's own ``main()`` for full-pipeline
regeneration — this is a fast path for iterating on this one figure only.

Usage (from project root):
    /path/to/miniforge3/envs/plots/bin/python \
        mw_classification_pipeline/scripts/make_fig_performance_and_prob_by_confidence.py
"""

import asyncio
import sys
from pathlib import Path

import kaleido as _kaleido

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from generate_combined_classification_figure import _export_fig_csv  # noqa: E402
from make_fig_prob_vs_dim_ws_loso import (  # noqa: E402
    OUTPUT_DIR,
    build_performance_and_prob_by_confidence_combined,
)

OUT_DIR = OUTPUT_DIR.parent
STEM = "performance_and_probabilities_by_confidence"


async def _write(fig, out_dir: Path, stem: str) -> None:
    async with _kaleido.Kaleido() as k:
        for fmt in ("png", "svg"):
            path = out_dir / f"{stem}.{fmt}"
            await k.write_fig(fig, path=path, opts={
                "format": fmt, "width": fig.layout.width, "height": fig.layout.height, "scale": 2,
            })
            print(f"  Saved: {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_performance_and_prob_by_confidence_combined()
    _export_fig_csv(fig, OUT_DIR / f"{STEM}.csv")
    asyncio.run(_write(fig, OUT_DIR, STEM))


if __name__ == "__main__":
    main()
