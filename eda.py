import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ── Style ──────────────────────────────────────────────────────────────────────
NAVY  = "#1F3864"
BLUE  = "#2E5FA3"
GOLD  = "#C9993A"
RED   = "#C0392B"
GREY  = "#6B7280"
BG    = "#F7F9FC"
WHITE = "#FFFFFF"

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.facecolor':   BG,
    'figure.facecolor': WHITE,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.labelcolor':  NAVY,
    'xtick.color':      GREY,
    'ytick.color':      GREY,
    'axes.edgecolor':   '#D1D9E6',
    'grid.color':       '#E5EAF2',
    'grid.linewidth':   0.6,
})

BASE = '/Users/kabir/Desktop/pro/credit_risk/'

# ── Load German Credit Dataset (UCI) ──────────────────────────────────────────
# Column names for the German Credit dataset
col_names = [
    'checking_account', 'duration', 'credit_history', 'purpose',
    'credit_amount', 'savings_account', 'employment', 'installment_rate',
    'personal_status', 'other_debtors', 'residence_since', 'property',
    'age', 'other_installments', 'housing', 'existing_credits', 'job',
    'dependents', 'telephone', 'foreign_worker', 'target'
]

print("Loading German Credit Dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
df = pd.read_csv(url, sep=' ', header=None, names=col_names)

# Target: 1 = Good (no default), 2 = Bad (default) → recode to 0 = good, 1 = default
df['default'] = (df['target'] == 2).astype(int)
df.drop('target', axis=1, inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['default'].mean():.1%}")
print(f"\nFeature types:\n{df.dtypes.value_counts()}")
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ── Summary stats ──────────────────────────────────────────────────────────────
print("\n===== NUMERICAL FEATURES =====")
num_cols = ['duration', 'credit_amount', 'installment_rate', 'age',
            'residence_since', 'existing_credits', 'dependents']
print(df[num_cols].describe().round(2))

print("\n===== DEFAULT RATE BY KEY SEGMENTS =====")

# By checking account status
check_map = {'A11': '<0 DM', 'A12': '0-200 DM', 'A13': '>200 DM', 'A14': 'No account'}
df['checking_label'] = df['checking_account'].map(check_map)
print("\nChecking Account Status:")
print(df.groupby('checking_label')['default'].agg(['mean','count'])
        .rename(columns={'mean':'Default Rate','count':'Count'})
        .sort_values('Default Rate', ascending=False)
        .round(3))

# By employment
emp_map = {'A71':'Unemployed','A72':'<1yr','A73':'1-4yrs','A74':'4-7yrs','A75':'>7yrs'}
df['employment_label'] = df['employment'].map(emp_map)
print("\nEmployment Duration:")
print(df.groupby('employment_label')['default'].agg(['mean','count'])
        .rename(columns={'mean':'Default Rate','count':'Count'})
        .sort_values('Default Rate', ascending=False)
        .round(3))

# By purpose
purpose_map = {
    'A40':'Car (new)','A41':'Car (used)','A42':'Furniture','A43':'TV/Radio',
    'A44':'Appliances','A45':'Repairs','A46':'Education','A48':'Retraining',
    'A49':'Business','A410':'Others'
}
df['purpose_label'] = df['purpose'].map(purpose_map).fillna('Other')
print("\nLoan Purpose:")
print(df.groupby('purpose_label')['default'].agg(['mean','count'])
        .rename(columns={'mean':'Default Rate','count':'Count'})
        .sort_values('Default Rate', ascending=False)
        .round(3))

# ── Visualisations ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Exploratory Data Analysis — German Credit Dataset',
             fontsize=15, fontweight='bold', color=NAVY, y=1.01)

# 1. Default rate (pie)
counts = df['default'].value_counts()
axes[0,0].pie([counts[0], counts[1]],
              labels=['Good (No Default)', 'Bad (Default)'],
              colors=[BLUE, RED], autopct='%1.1f%%',
              startangle=90, textprops={'fontsize':10})
axes[0,0].set_title('Default Rate', fontsize=12, fontweight='bold', color=NAVY)

# 2. Credit amount distribution
axes[0,1].hist(df[df['default']==0]['credit_amount'], bins=40,
               alpha=0.6, color=BLUE, label='No Default')
axes[0,1].hist(df[df['default']==1]['credit_amount'], bins=40,
               alpha=0.6, color=RED, label='Default')
axes[0,1].set_title('Credit Amount by Default', fontsize=12, fontweight='bold', color=NAVY)
axes[0,1].set_xlabel('Credit Amount (DM)')
axes[0,1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x/1000:.0f}k'))
axes[0,1].legend(frameon=False, fontsize=9)
axes[0,1].grid(axis='y', alpha=0.5)

# 3. Age distribution
axes[0,2].hist(df[df['default']==0]['age'], bins=30,
               alpha=0.6, color=BLUE, label='No Default')
axes[0,2].hist(df[df['default']==1]['age'], bins=30,
               alpha=0.6, color=RED, label='Default')
axes[0,2].set_title('Age by Default Status', fontsize=12, fontweight='bold', color=NAVY)
axes[0,2].set_xlabel('Age (years)')
axes[0,2].legend(frameon=False, fontsize=9)
axes[0,2].grid(axis='y', alpha=0.5)

# 4. Duration distribution
axes[1,0].hist(df[df['default']==0]['duration'], bins=30,
               alpha=0.6, color=BLUE, label='No Default')
axes[1,0].hist(df[df['default']==1]['duration'], bins=30,
               alpha=0.6, color=RED, label='Default')
axes[1,0].set_title('Loan Duration by Default', fontsize=12, fontweight='bold', color=NAVY)
axes[1,0].set_xlabel('Duration (months)')
axes[1,0].legend(frameon=False, fontsize=9)
axes[1,0].grid(axis='y', alpha=0.5)

# 5. Default rate by checking account
chk = df.groupby('checking_label')['default'].mean().sort_values(ascending=True)
colors = [RED if v > 0.35 else GOLD if v > 0.25 else BLUE for v in chk.values]
axes[1,1].barh(chk.index, chk.values, color=colors)
axes[1,1].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes[1,1].set_title('Default Rate by Checking Account', fontsize=12, fontweight='bold', color=NAVY)
axes[1,1].grid(axis='x', alpha=0.5)

# 6. Default rate by employment
emp = df.groupby('employment_label')['default'].mean().sort_values(ascending=True)
colors2 = [RED if v > 0.35 else GOLD if v > 0.25 else BLUE for v in emp.values]
axes[1,2].barh(emp.index, emp.values, color=colors2)
axes[1,2].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes[1,2].set_title('Default Rate by Employment', fontsize=12, fontweight='bold', color=NAVY)
axes[1,2].grid(axis='x', alpha=0.5)

plt.tight_layout()
plt.savefig(BASE + 'fig1_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: fig1_eda.png")

# Save cleaned dataset
df.to_csv(BASE + 'german_credit_clean.csv', index=False)
print(f"Cleaned data saved: german_credit_clean.csv")
print("\nPhase 1 complete!")