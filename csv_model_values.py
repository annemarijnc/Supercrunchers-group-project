# =============================================================================
# prepare_shap_data.py
#
# Reconstructs per-participant logistic regression models and computes SHAP
# weights for the five modelled traits (sincere, intelligence, funny,
# ambition, interests_correlate), for both block 1 (pre-explanation) and
# block 2 (post-explanation).
#
# Inputs (same directory as this script, or set paths in CONFIG below):
#   all_results.xlsx
#   female_set_A.csv / female_set_B.csv
#   male_set_A.csv   / male_set_B.csv
#
# Output:
#   all_results_with_shap.csv   — one row per participant, all original columns
#                                 plus 10 new SHAP columns (5 traits × 2 blocks)
#                                 plus 5-trait model accuracy for each block.
#
# Profile-set routing:
#   preference == "Women", block1_set == "A"  →  block1 = female_set_A
#   preference == "Women", block1_set == "B"  →  block1 = female_set_B
#   preference == "Men",   block1_set == "A"  →  block1 = male_set_A
#   preference == "Men",   block1_set == "B"  →  block1 = male_set_B
#   Block 2 always receives the other set letter (A↔B), same preference gender.
#
# SHAP weights:
#   For each participant × block, a LogisticRegression is fitted on the 5-trait
#   feature matrix derived from their ratings (binarised at DECISION_THRESHOLD).
#   shap.LinearExplainer computes per-profile SHAP values; mean absolute SHAP
#   across the 30 profiles is taken as the trait importance, then
#   MinMax-normalised and rescaled to sum to 100 — matching the normalisation
#   in the original model.py (compute_normalized_coefficients).
#
#   Columns added to the output:
#     shap_b1_sincere / shap_b2_sincere
#     shap_b1_intelligence / shap_b2_intelligence
#     shap_b1_funny / shap_b2_funny
#     shap_b1_ambition / shap_b2_ambition
#     shap_b1_interests_correlate / shap_b2_interests_correlate
#     shap_b1_model_acc / shap_b2_model_acc   (5-trait refit accuracy)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")   # suppress shap FutureWarnings

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import shap

# ------------------------------------------------------------------
# CONFIG — adjust paths if files are not in the working directory
# ------------------------------------------------------------------
DATA_PATH         = "all_results.xlsx"
FEMALE_A_PATH     = "female_set_A.csv"
FEMALE_B_PATH     = "female_set_B.csv"
MALE_A_PATH       = "male_set_A.csv"
MALE_B_PATH       = "male_set_B.csv"
OUTPUT_PATH       = "all_results_with_shap.csv"
DECISION_THRESHOLD = 6   # rating >= threshold  →  binary "yes" (must match R script)

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
INTEREST_COLS = [
    'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
    'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
    'movies', 'concerts', 'music', 'shopping', 'yoga'
]
PREF_COLS = [
    'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny',
    'pref_o_ambitious', 'pref_o_shared_interests'
]
FIVE_TRAITS = ['sincere', 'intelligence', 'funny', 'ambition', 'interests_correlate']

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
print("Loading data...")
df = pd.read_excel(DATA_PATH)
print(f"  Loaded {len(df)} participants, {len(df.columns)} columns.\n")

profile_sets = {
    ('Women', 'A'): pd.read_csv(FEMALE_A_PATH),
    ('Women', 'B'): pd.read_csv(FEMALE_B_PATH),
    ('Men',   'A'): pd.read_csv(MALE_A_PATH),
    ('Men',   'B'): pd.read_csv(MALE_B_PATH),
}

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def build_feature_matrix(participant_row, profile_df):
    """
    Construct the 30-row × 5-column feature matrix for one participant × one block.
    Columns: sincere, intelligence, funny, ambition, interests_correlate.
    interests_correlate = Pearson r between participant's 17 interest scores
                          and the profile's 17 interest scores.
    """
    p_interests = participant_row[INTEREST_COLS].values.astype(float)
    rows = []
    for _, prof in profile_df.iterrows():
        d_int = prof[INTEREST_COLS].values.astype(float)
        r, _ = pearsonr(p_interests, d_int)
        rows.append({
            'sincere':             prof['sincere'],
            'intelligence':        prof['intelligence'],
            'funny':               prof['funny'],
            'ambition':            prof['ambition'],
            'interests_correlate': r,
        })
    return pd.DataFrame(rows, columns=FIVE_TRAITS)


def compute_shap_weights(participant_row, profile_df, ratings_str):
    """
    Fit a logistic regression on the 5-trait matrix for one participant × block,
    compute mean absolute SHAP values, and normalise to sum to 100.

    Returns:
        norm_weights  : np.array of shape (5,) — normalised SHAP weights (sum ≈ 100)
        refit_acc     : float — accuracy of the refitted 5-trait model
    """
    ratings = list(map(int, ratings_str.split(';')))
    y = np.array([1 if r >= DECISION_THRESHOLD else 0 for r in ratings])

    X = build_feature_matrix(participant_row, profile_df)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)
    refit_acc = model.score(X, y)

    explainer  = shap.LinearExplainer(model, X)
    shap_vals  = explainer.shap_values(X)          # shape (30, 5)
    mean_abs   = np.abs(shap_vals).mean(axis=0)    # shape (5,)

    # Normalise to sum to 100 (MinMax then proportional rescale)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(mean_abs.reshape(-1, 1)).flatten()
    norm   = (scaled / scaled.sum() * 100) if scaled.sum() > 0 else np.zeros(5)

    return norm, refit_acc


# ------------------------------------------------------------------
# MAIN LOOP — compute SHAP for every participant, both blocks
# ------------------------------------------------------------------
print(f"Computing SHAP weights (decision threshold = {DECISION_THRESHOLD})...\n")

shap_b1 = np.full((len(df), 5), np.nan)
shap_b2 = np.full((len(df), 5), np.nan)
acc_b1  = np.full(len(df), np.nan)
acc_b2  = np.full(len(df), np.nan)

for i, row in df.iterrows():
    pref  = row['preference']
    b1set = row['block1_set']
    b2set = 'B' if b1set == 'A' else 'A'

    prof_b1 = profile_sets[(pref, b1set)]
    prof_b2 = profile_sets[(pref, b2set)]

    # Block 1
    w1, a1          = compute_shap_weights(row, prof_b1, row['ratings_block1'])
    shap_b1[i]      = w1
    acc_b1[i]       = a1

    # Block 2
    w2, a2          = compute_shap_weights(row, prof_b2, row['ratings_block2'])
    shap_b2[i]      = w2
    acc_b2[i]       = a2

    print(f"  [{i+1:2d}/{len(df)}] acc_b1={a1:.3f}  acc_b2={a2:.3f}  "
          + "  ".join(f"{t[:4]}={w1[j]:.1f}/{w2[j]:.1f}"
                      for j, t in enumerate(FIVE_TRAITS)))

# ------------------------------------------------------------------
# ATTACH SHAP COLUMNS TO DATAFRAME
# ------------------------------------------------------------------
for j, trait in enumerate(FIVE_TRAITS):
    df[f'shap_b1_{trait}'] = shap_b1[:, j]
    df[f'shap_b2_{trait}'] = shap_b2[:, j]

df['shap_b1_model_acc'] = acc_b1
df['shap_b2_model_acc'] = acc_b2

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDone. Output written to: {OUTPUT_PATH}")
print(f"New columns added: {[c for c in df.columns if c.startswith('shap_')]}")
