"""Measure per-position analysis time at real Cephla dimensions (3000x3000 x24 x2ch),
so we can estimate how long a full 96-position run takes on CPU - no GPU, no cloud.
"""

import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cellscope.analysis import AnalysisSettings, run_analysis
from cellscope.data.mock import MockLoader


def main():
    cores = multiprocessing.cpu_count()
    print(f"CPU cores: {cores}")

    # One synthetic position at the same shape as the real Cephla data.
    loader = MockLoader(n_wells=1, n_time=24, n_channels=2, size=3000, n_cells=150)
    wid = loader.list_wells()[0].well_id

    t = time.time()
    arr = loader.get_well(wid)
    print(f"in-RAM position: shape {arr.shape}, {arr.nbytes/1e6:.0f} MB "
          f"(synth {time.time()-t:.1f}s; real load is disk-bound)")

    t = time.time()
    wa = run_analysis(loader, wid, AnalysisSettings(min_size=40))
    tf = time.time() - t
    print(f"analysis FULL 3000^2 x24: {tf:.1f}s  ({wa.n_tracks} cells)")

    t = time.time()
    wa = run_analysis(loader, wid, AnalysisSettings(min_size=40), downsample=3)
    td = time.time() - t
    print(f"analysis 1/3 (1000^2) x24: {td:.1f}s  ({wa.n_tracks} cells)")

    print("\n--- extrapolation to 96 positions ---")
    print(f"full, 1 core:      {96*tf/60:.0f} min")
    print(f"full, {cores} cores:  {96*tf/60/cores:.0f} min")
    print(f"1/3,  {cores} cores:  {96*td/60/cores:.0f} min")


if __name__ == "__main__":
    main()
