"""Validate each simulation against its own PCA shape and parameter row."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.mean_base.validate import validate


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data',type=Path)
    a=ap.parse_args()
    groups=defaultdict(list)
    with (a.data/'params.csv').open() as f:
        for row in csv.DictReader(f):
            groups[row['geom']].append(row['case'])
    for geom,names in groups.items():
        validate(a.data,names=names,
                 mesh_path=a.data/'_geometry'/f'dalton_geom_{geom}'/'dalton_lv_mesh.h5')
    print(f'All {sum(map(len,groups.values()))} geometry-variation cases validated',flush=True)


if __name__=='__main__':
    main()
