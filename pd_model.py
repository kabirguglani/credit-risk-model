import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, brier_score_loss)
from sklearn.pipeline import Pipeline
import joblib

# ── Style ──────────────────────────────────────────────────────────────────────
NAVY = "#1F3864"; BLUE = "#2E5FA3"; GOLD = "#C9993A"
RED  = "#C0392B"; GREY = "#6B7280"; BG   = "#F7F9FC"

plt.rcParams.update({
    'font.family':'DejaVu Sans','axes.facecolor':BG,'figure.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelcolor':NAVY,'xtick.color':GREY,'ytick.color':GREY,
    'axes.edgecolor':'#D1D9E6','grid.color':'#E5EAF2','grid.linewidth':0.6,
})

BASE = '/Users/kabir/Desktop/pro/credit_risk/'

# ── Load cleaned data ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(BASE + 'german_credit_clean.csv')

# ── Feature Engineering ────────────────────────────────────────────────────────
print("Engineering features...")

# Weight of Evidence (WoE) encoding for categorical variables
# WoE = ln(Distribution of Events / Distribution of Non-Events)
def calc_woe_iv(df, feature, target):
    total_events = df[target].sum()
    total_non_events = (df[target] == 0).sum()
    stats = df.groupby(feature)[target].agg(['sum', 'count'])
    stats.columns = ['events', 'total']
    stats['non_events'] = stats['total'] - stats['events']
    stats['dist_events'] = stats['events'] / total_events
    stats['dist_non_events'] = stats['non_events'] / total_non_events
    stats['dist_events'] = stats['dist_events'].replace(0, 0.0001)
    stats['dist_non_events'] = stats['dist_non_events'].replace(0, 0.0001)
    stats['woe'] = np.log(stats['dist_events'] / stats['dist_non_events'])
    stats['iv'] = (stats['dist_events'] - stats['dist_non_events']) * stats['woe']
    return stats['woe'].to_dict(), stats['iv'].sum()

# Categorical columns to encode
cat_cols = ['checking_account', 'credit_history', 'purpose', 'savings_account',
            'employment', 'personal_status', 'other_debtors', 'property',
            'other_installments', 'housing', 'job', 'telephone', 'foreign_worker']

woe_maps = {}
iv_scores = {}

for col in cat_cols:
    woe_map, iv = calc_woe_iv(df, col, 'default')
    woe_maps[col] = woe_map
    iv_scores[col] = iv
    df[f'{col}_woe'] = df[col].map(woe_map)

# Numerical features (keep as-is + add engineered features)
df['log_credit_amount'] = np.log(df['credit_amount'])
df['credit_duration_ratio'] = df['credit_amount'] / df['duration']
df['age_employment_interaction'] = df['age'] * df['installment_rate']

# ── Information Value Summary ──────────────────────────────────────────────────
iv_df = pd.DataFrame(list(iv_scores.items()), columns=['Feature', 'IV'])
iv_df['Predictive Power'] = pd.cut(iv_df['IV'],
    bins=[0, 0.02, 0.1, 0.3, 0.5, float('inf')],
    labels=['Useless', 'Weak', 'Medium', 'Strong', 'Very Strong'])
iv_df = iv_df.sort_values('IV', ascending=False)

print("\n===== INFORMATION VALUE (IV) SCORES =====")
print(iv_df.to_string(index=False))

# ── Feature Selection ──────────────────────────────────────────────────────────
# Use WoE-encoded categoricals + key numerical features
feature_cols = (
    [f'{c}_woe' for c in cat_cols] +
    ['duration', 'log_credit_amount', 'installment_rate', 'age',
     'residence_since', 'existing_credits', 'dependents',
     'credit_duration_ratio', 'age_employment_interaction']
)

X = df[feature_cols].fillna(0)
y = df['default']

# ── Train / Test Split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train default rate: {y_train.mean():.1%} | Test default rate: {y_test.mean():.1%}")

# ── Logistic Regression Model ──────────────────────────────────────────────────
print("\nTraining PD model...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(C=0.1, class_weight='balanced',
                                  solver='lbfgs', max_iter=1000, random_state=42))
])
pipeline.fit(X_train, y_train)

# ── Cross-Validation ───────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
print(f"\n5-Fold CV AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# ── Test Set Evaluation ────────────────────────────────────────────────────────
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

auc     = roc_auc_score(y_test, y_prob)
brier   = brier_score_loss(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
gini    = 2 * auc - 1

print(f"\n===== MODEL PERFORMANCE =====")
print(f"AUC-ROC:     {auc:.3f}")
print(f"Gini:        {gini:.3f}")
print(f"Brier Score: {brier:.3f}  (lower = better, 0.25 = random)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Default','Default']))

# ── PD Scores & Risk Grades ────────────────────────────────────────────────────
# Assign credit risk grades based on PD
def assign_grade(pd_score):
    if pd_score < 0.05:   return 'AAA'
    elif pd_score < 0.10: return 'AA'
    elif pd_score < 0.15: return 'A'
    elif pd_score < 0.20: return 'BBB'
    elif pd_score < 0.30: return 'BB'
    elif pd_score < 0.45: return 'B'
    else:                 return 'CCC'

# Apply to full dataset
df['pd_score'] = pipeline.predict_proba(X.fillna(0))[:, 1]
df['risk_grade'] = df['pd_score'].apply(assign_grade)

grade_summary = df.groupby('risk_grade').agg(
    Count=('pd_score','count'),
    Avg_PD=('pd_score','mean'),
    Actual_Default_Rate=('default','mean')
).round(3)
grade_summary['Expected_Loss_Rate'] = (grade_summary['Avg_PD'] * 0.45).round(3)  # LGD=45%
print("\n===== RISK GRADE DISTRIBUTION =====")
print(grade_summary.sort_values('Avg_PD'))

# ── Save model and scores ──────────────────────────────────────────────────────
joblib.dump(pipeline, BASE + 'pd_model.pkl')
df.to_csv(BASE + 'scored_portfolio.csv', index=False)
print(f"\nModel saved: pd_model.pkl")
print(f"Scored portfolio saved: scored_portfolio.csv")

# ── Visualisations ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('PD Model — Performance & Risk Grade Analysis',
             fontsize=15, fontweight='bold', color=NAVY)

# 1. ROC Curve
axes[0,0].plot(fpr, tpr, color=BLUE, lw=2.2, label=f'PD Model (AUC = {auc:.3f})')
axes[0,0].plot([0,1],[0,1], color=GREY, lw=1, ls='--', label='Random (AUC = 0.500)')
axes[0,0].fill_between(fpr, tpr, alpha=0.1, color=BLUE)
axes[0,0].set_xlabel('False Positive Rate'); axes[0,0].set_ylabel('True Positive Rate')
axes[0,0].set_title('ROC Curve', fontsize=12, fontweight='bold', color=NAVY)
axes[0,0].legend(frameon=False, fontsize=9)
axes[0,0].grid(alpha=0.4)

# 2. PD Score Distribution
axes[0,1].hist(df[df['default']==0]['pd_score'], bins=40,
               alpha=0.6, color=BLUE, label='No Default', density=True)
axes[0,1].hist(df[df['default']==1]['pd_score'], bins=40,
               alpha=0.6, color=RED, label='Default', density=True)
axes[0,1].set_xlabel('PD Score'); axes[0,1].set_ylabel('Density')
axes[0,1].set_title('PD Score Distribution', fontsize=12, fontweight='bold', color=NAVY)
axes[0,1].legend(frameon=False, fontsize=9)
axes[0,1].grid(axis='y', alpha=0.4)

# 3. Risk Grade Distribution
grade_order = ['AAA','AA','A','BBB','BB','B','CCC']
grade_counts = df['risk_grade'].value_counts().reindex(grade_order).fillna(0)
grade_colors = [BLUE,BLUE,'#5B8DB8',GOLD,GOLD,RED,RED]
axes[1,0].bar(grade_order, grade_counts.values, color=grade_colors, edgecolor='white')
axes[1,0].set_xlabel('Risk Grade'); axes[1,0].set_ylabel('Number of Loans')
axes[1,0].set_title('Portfolio Risk Grade Distribution', fontsize=12, fontweight='bold', color=NAVY)
axes[1,0].grid(axis='y', alpha=0.4)
for i, (g, v) in enumerate(zip(grade_order, grade_counts.values)):
    axes[1,0].text(i, v+2, str(int(v)), ha='center', fontsize=9, color=NAVY)

# 4. IV Scores
top_iv = iv_df.head(8)
bar_colors = [RED if v > 0.3 else GOLD if v > 0.1 else BLUE for v in top_iv['IV']]
axes[1,1].barh(top_iv['Feature'].str.replace('_',' ').str.title(),
               top_iv['IV'], color=bar_colors)
axes[1,1].axvline(0.1, color=GOLD, ls='--', lw=1.2, label='Medium (0.1)')
axes[1,1].axvline(0.3, color=RED,  ls='--', lw=1.2, label='Strong (0.3)')
axes[1,1].set_xlabel('Information Value (IV)')
axes[1,1].set_title('Feature Predictive Power (IV)', fontsize=12, fontweight='bold', color=NAVY)
axes[1,1].legend(frameon=False, fontsize=8)
axes[1,1].grid(axis='x', alpha=0.4)

plt.tight_layout()
plt.savefig(BASE + 'fig2_pd_model.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure saved: fig2_pd_model.png")
print("\nPhase 2 complete!")