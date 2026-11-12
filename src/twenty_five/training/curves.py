"""Machine-readable learning/sample-efficiency curve writer."""
from __future__ import annotations
import csv
from pathlib import Path

def write_curve(checkpoint: Path, values: list[float], metric: str, unit: str="episode") -> Path:
    path=checkpoint.with_suffix(".curve.csv"); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=[unit,metric]); writer.writeheader(); writer.writerows([{unit:i+1,metric:v} for i,v in enumerate(values)])
    return path
