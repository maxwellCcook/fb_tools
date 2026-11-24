"""
Does the baseline shadow mark the cells a treatment changed? Read-only, on existing paired runs.

    python s7_shadow_agreement.py <root> <baseline_dir> <treated_dir> [<treated_dir> ...]

``root`` is the folder the manifests' relative ``layers_dir`` resolve against (e.g. ~/fb_scratch/pyre_example).
Sweeps the gradient-weight exponent: 1 is the package default, 50 is in effect a single steepest path.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fb_tools.spread.transmission import changed_cells, shadow_agreement

root, bdir, tdirs = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
layers = lambda rd: root / json.loads((rd / "manifest.json").read_text())["layers_dir"]   # noqa: E731
pd.set_option("display.width", 200)
for tdir in tdirs:
    trt = changed_cells(layers(bdir), layers(tdir))
    rows = pd.concat([shadow_agreement(bdir, tdir, trt, power=p, tolerances_min=(0.0, 10.0, 60.0)) for p in (1, 2, 4, 8, 50)])
    print(f"\n{tdir.name} vs {bdir.name}: {int(trt.sum())} treated cells")
    print(rows[["power", "tol_min", "n_cells", "n_changed", "n_shadow", "recall", "precision", "auc"]].round(3).to_string(index=False))
