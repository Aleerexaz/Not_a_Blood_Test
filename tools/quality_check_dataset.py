#!/usr/bin/env python3
"""
Scan a dataset folder (subfolders per label) and extract features using
bloodgroup_ABO_calibrated.extract_features. Produce a CSV and a JSON
quality report, and write a keep_list.txt of files passing basic quality.

Usage (PowerShell):
  python tools/quality_check_dataset.py --input-dir .\clean_dataset \
    --out-csv features_clean_dataset.csv --report quality_report.json

Quality gates (conservative, adjustable):
  - skeleton_pixels >= 1500
  - orientation_entropy <= 2.6 (computed from ori_bin_0..7)
  - optional: contrast >= 60 (very low contrast tends to be weak)
These are heuristics only and can be tuned per dataset.
"""
import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

import sys
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bloodgroup_ABO_calibrated as model


def orientation_entropy_from_features(feats: Dict[str, Any]) -> float:
    bins = [feats.get(f"ori_bin_{i}", 0.0) for i in range(8)]
    arr = np.array(bins, dtype=float)
    s = float(arr.sum())
    if s <= 0:
        return 0.0
    p = np.clip(arr / s, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())


def passes_quality(feats: Dict[str, Any], sk_min=1500, ent_max=2.6, contrast_min=60) -> bool:
    try:
        sk = float(feats.get("skeleton_pixels", 0.0) or 0.0)
    except Exception:
        sk = 0.0
    try:
        contrast = float(feats.get("contrast", 0.0) or 0.0)
    except Exception:
        contrast = 0.0
    oent = orientation_entropy_from_features(feats)
    return (sk >= sk_min) and (oent <= ent_max) and (contrast >= contrast_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True, help='Root dataset folder with label subfolders')
    ap.add_argument('--out-csv', default='features_clean_dataset.csv')
    ap.add_argument('--report', default='quality_report.json')
    ap.add_argument('--sk-min', type=int, default=1500)
    ap.add_argument('--ent-max', type=float, default=2.6)
    ap.add_argument('--contrast-min', type=float, default=60.0)
    ap.add_argument('--limit', type=int, default=0, help='Scan at most N images total (0 = no limit)')
    ap.add_argument('--per-class-limit', type=int, default=0, help='Scan at most N images per class (0 = no limit)')
    args = ap.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise SystemExit(f"Input folder not found: {root}")

    keys = list(model.keys_order)
    extra_cols = [
        'filename', 'orientation_entropy', 'quality_pass',
    ]
    cols = ['label', 'filename'] + keys + extra_cols[1:]

    rows: List[Dict[str, Any]] = []
    bad: List[str] = []
    keep: List[str] = []
    cls_counts: Dict[str, int] = {}
    cls_pass: Dict[str, int] = {}

    seen = 0
    seen_per_class: Dict[str, int] = {}
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = label_dir.name
        seen_per_class[label] = 0
        for fp in label_dir.rglob('*'):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                continue
            if args.per_class_limit and seen_per_class[label] >= args.per_class_limit:
                break
            fpath = str(fp)
            try:
                feats = model.extract_features(fpath)
                oent = orientation_entropy_from_features(feats)
                qpass = passes_quality(feats, args.sk_min, args.ent_max, args.contrast_min)

                row = {
                    'label': label,
                    'filename': fpath,
                    'orientation_entropy': oent,
                    'quality_pass': int(qpass),
                }
                for k in keys:
                    row[k] = feats.get(k)
                rows.append(row)

                cls_counts[label] = cls_counts.get(label, 0) + 1
                if qpass:
                    cls_pass[label] = cls_pass.get(label, 0) + 1
                    keep.append(fpath)
                else:
                    bad.append(fpath)

            except Exception as e:
                bad.append(fpath)
                # Continue scanning despite errors
                continue
            finally:
                seen += 1
                seen_per_class[label] += 1
                if args.limit and seen >= args.limit:
                    break
        if args.limit and seen >= args.limit:
            break

    # Write CSV
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write keep/bad lists for convenience
    keep_list = Path(args.out_csv).with_suffix('')
    keep_txt = str(keep_list) + '_keep.txt'
    bad_txt = str(keep_list) + '_bad.txt'
    with open(keep_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(keep))
    with open(bad_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bad))

    # Build summary
    def safe_ratio(a, b):
        return float(a) / float(b) if b else 0.0

    summary = {
        'input_dir': str(root),
        'rows': len(rows),
        'classes': sorted(cls_counts.keys()),
        'per_class': {
            c: {
                'total': cls_counts.get(c, 0),
                'pass_quality': cls_pass.get(c, 0),
                'pass_rate': round(100.0 * safe_ratio(cls_pass.get(c, 0), cls_counts.get(c, 0)), 2)
            } for c in sorted(cls_counts.keys())
        },
        'thresholds': {
            'skeleton_pixels_min': args.sk_min,
            'orientation_entropy_max': args.ent_max,
            'contrast_min': args.contrast_min,
        },
        'artifacts': {
            'csv': os.path.abspath(args.out_csv),
            'keep_list': os.path.abspath(keep_txt),
            'bad_list': os.path.abspath(bad_txt),
        }
    }

    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
