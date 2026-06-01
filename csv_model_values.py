import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from model import compute_normalized_coefficients
from app import INTEREST_FIELDS

# ------------------------------------------------------------------
# CONFIG — adjust paths if files are not in the working directory
# ------------------------------------------------------------------
DATA_PATH         = "all_results.csv"
FEMALE_A_PATH     = "profile_sets/female_set_A.csv"
FEMALE_B_PATH     = "profile_sets/female_set_B.csv"
MALE_A_PATH       = "profile_sets/male_set_A.csv"
MALE_B_PATH       = "profile_sets/male_set_B.csv"
OUTPUT_PATH       = "all_results_with_coefficients.csv"
DECISION_THRESHOLD = 6   # rating >= threshold  →  binary "yes"
RANDOM_SEED        = 42  # for reproducibility of logistic regression fitting

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------

PREF_COLS = [
    'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny',
    'pref_o_ambitious', 'pref_o_shared_interests'
]
FIVE_TRAITS = ['sincere', 'intelligence', 'funny', 'ambition', 'interests_correlate']

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
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
    p_interests = participant_row[INTEREST_FIELDS].values.astype(float)
    rows = []
    for _, prof in profile_df.iterrows():
        d_int = prof[INTEREST_FIELDS].values.astype(float)
        r = np.corrcoef(p_interests, d_int)[0, 1]  # Pearson correlation 
        rows.append({
            'sincere':             prof['sincere'],
            'intelligence':        prof['intelligence'],
            'funny':               prof['funny'],
            'ambition':            prof['ambition'],
            'interests_correlate': r,
        })
    return pd.DataFrame(rows, columns=FIVE_TRAITS)

def recompute_coefficients(participant_row, profile_df, ratings_str):
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

    model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    model.fit(X, y)
    refit_acc = model.score(X, y)
    norm_weights = compute_normalized_coefficients(model, X)

    return norm_weights, refit_acc

def rating_to_decision(rating):
    return 1 if rating >= DECISION_THRESHOLD else 0

interest_corr  = np.full(len(df), np.nan)
coeff_b1 = np.full((len(df), 5), np.nan)
coeff_b2 = np.full((len(df), 5), np.nan)
acc_b1  = np.full(len(df), np.nan)
acc_b2  = np.full(len(df), np.nan)

for i, row in df.iterrows():
    pref  = row['preference']
    b1set = row['block1_set']
    b2set = 'B' if b1set == 'A' else 'A'

    prof_b1 = profile_sets[(pref, b1set)]
    prof_b2 = profile_sets[(pref, b2set)]

    interest_ratings_participant = row[INTEREST_FIELDS].values.astype(float)
    interest_ratings_profiles = prof_b1[INTEREST_FIELDS].values.astype(float)  # same for both blocks, since profiles don't change
    row['interests_correlate'] = np.corrcoef(interest_ratings_participant, interest_ratings_profiles)[0, 1]  # same for both blocks, since participant's interests don't change

    # Block 1
    w1, a1          = recompute_coefficients(row, prof_b1, row['ratings_block1'])
    coeff_b1[i]      = w1
    acc_b1[i]       = a1

    # Block 2
    w2, a2          = recompute_coefficients(row, prof_b2, row['ratings_block2'])
    coeff_b2[i]      = w2
    acc_b2[i]       = a2

    print(f"  [{i+1:2d}/{len(df)}] acc_b1={a1:.3f}  acc_b2={a2:.3f}  "
          + "  ".join(f"{t[:4]}={w1[j]:.1f}/{w2[j]:.1f}"
                      for j, t in enumerate(FIVE_TRAITS)))

# ------------------------------------------------------------------
# ATTACH coefficients, accuracies & interest correlation to the original df
# ------------------------------------------------------------------
for j, trait in enumerate(FIVE_TRAITS):
    df[f'coeff_b1_{trait}'] = coeff_b1[:, j]
    df[f'coeff_b2_{trait}'] = coeff_b2[:, j]

df['coeff_b1_model_acc'] = acc_b1
df['coeff_b2_model_acc'] = acc_b2
df['interest_correlation'] = interest_corr

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDone. Output written to: {OUTPUT_PATH}")
print(f"New columns added: {[c for c in df.columns if c.startswith('coeff_')]}")
