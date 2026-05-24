"""
personalized_model.py

Train a per-participant regression model to predict participant ratings (1-10)
from Block 1 profiles. Uses a simple Ridge regression with Leave-One-Out CV for
reliability estimation, then fits a final model on all data and prints/saves
coefficients and metrics.

Usage:
  python personalized_model.py --input payload.json --out model.pkl
  python personalized_model.py --input block1.csv --out model.pkl
  python personalized_model.py --demo    # runs synthetic demo

Input formats accepted:
- JSON payload exported from frontend `/api/submit` (contains `block1.profileData`) 
  where each profileData item must include numeric fields: `profile_sincere`,
  `profile_intelligence`, `profile_funny`, `profile_ambition`, `interests_correlate`,
  and `rating` (participant rating 0-10).
- CSV with columns matching the feature names above.

Outputs:
- Prints LOO CV RMSE / R^2 and final model coefficients (normalized to sum to 100)
- Saves model pickle if `--out` provided

This file is intentionally standalone and does not modify the existing codebase.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib

FEATURE_COLS = ['profile_sincere', 'profile_intelligence', 'profile_funny', 'profile_ambition', 'interests_correlate']
TARGET_COL = 'rating'


def load_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Accept either full payload with block1.profileData or a raw list
    if isinstance(data, dict) and 'block1' in data and isinstance(data['block1'], dict):
        items = data['block1'].get('profileData') or data['block1'].get('profiles') or []
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError('JSON file format not recognised: expected block1.profileData or a list of items')

    rows = []
    for it in items:
        row = {}
        for c in FEATURE_COLS:
            if c in it:
                row[c] = float(it[c])
            else:
                # try common variants
                if c.startswith('profile_') and c.replace('profile_', '') in it:
                    row[c] = float(it[c.replace('profile_', '')])
                else:
                    raise KeyError(f'Missing feature "{c}" in JSON item: {it.keys()}')
        # rating can be named 'rating' or 'rating' inside item
        if TARGET_COL in it:
            row[TARGET_COL] = float(it[TARGET_COL])
        elif 'rating' in it:
            row[TARGET_COL] = float(it['rating'])
        else:
            raise KeyError('Missing target rating in JSON item')
        rows.append(row)
    return pd.DataFrame(rows)


def load_from_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise KeyError(f'Missing columns in CSV: {missing}')
    return df[FEATURE_COLS + [TARGET_COL]].astype(float)


def evaluate_and_train(df, alpha=1.0):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    if len(df) < 3:
        raise ValueError('Need at least 3 profiles to train and evaluate reliably')

    model = Ridge(alpha=alpha)

    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])

    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = r2_score(y, preds)

    # final model on all data
    final_model = Ridge(alpha=alpha).fit(X, y)
    coefs = final_model.coef_.astype(float)
    intercept = float(final_model.intercept_)

    # normalized importance (0..100)
    abs_coefs = np.abs(coefs)
    if abs_coefs.sum() == 0:
        norm = np.zeros_like(abs_coefs)
    else:
        norm = abs_coefs / abs_coefs.sum() * 100

    return {
        'rmse_loo': float(rmse),
        'r2_loo': float(r2),
        'coefs': {FEATURE_COLS[i]: float(coefs[i]) for i in range(len(FEATURE_COLS))},
        'intercept': intercept,
        'importance_pct': {FEATURE_COLS[i]: float(norm[i]) for i in range(len(FEATURE_COLS))},
        'model': final_model,
        'predictions': preds,
        'y_true': y,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', help='Input JSON or CSV file with block1 profileData')
    p.add_argument('--out', '-o', help='Output path to save trained model (pickle)')
    p.add_argument('--alpha', type=float, default=1.0, help='Ridge regularization alpha')
    p.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    args = p.parse_args(argv)

    if args.demo:
        # build tiny synthetic dataset (n=30)
        rng = np.random.RandomState(0)
        n = 30
        X = rng.uniform(3, 8, size=(n, len(FEATURE_COLS)))
        # create weights (roughly matching point allocation influence)
        true_w = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        y = X.dot(true_w) + rng.normal(scale=0.8, size=n)
        y = np.clip((y - y.min()) / (y.max() - y.min()) * 9 + 1, 1, 10)  # scale to 1-10
        df = pd.DataFrame(X, columns=FEATURE_COLS)
        df[TARGET_COL] = y

    else:
        if not args.input:
            print('Specify --input JSON/CSV or --demo', file=sys.stderr)
            return 2
        path = Path(args.input)
        if not path.exists():
            print(f'Input file not found: {path}', file=sys.stderr)
            return 2
        if path.suffix.lower() in ('.json', '.js'):
            df = load_from_json(path)
        else:
            df = load_from_csv(path)

    print(f'Loaded {len(df)} profiles')
    res = evaluate_and_train(df, alpha=args.alpha)

    print('\nLOO CV results:')
    print(f"  RMSE: {res['rmse_loo']:.3f}")
    print(f"  R^2 : {res['r2_loo']:.3f}")

    print('\nFinal model coefficients:')
    for f in FEATURE_COLS:
        print(f'  {f:20s}: coef={res['coefs'][f]:.4f}  importance={res['importance_pct'][f]:.1f}%')

    # show first 5 predictions vs truth
    preds = res['predictions']
    y = res['y_true']
    print('\nSample predictions (LOO):')
    for i in range(min(5, len(y))):
        print(f'  true={y[i]:.2f}  pred={preds[i]:.2f}')

    if args.out:
        joblib.dump(res['model'], args.out)
        print(f'Final model saved to {args.out}')

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
