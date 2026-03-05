import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

# ── Style ──────────────────────────────────────────────────────────────────────
NAVY = "#1F3864"; BLUE = "#2E5FA3"; GOLD = "#C9993A"
RED  = "#C0392B"; GREY = "#6B7280"; BG   = "#F7F9FC"
GREEN = "#1A7A4A"

plt.rcParams.update({
    'font.family':'DejaVu Sans','axes.facecolor':BG,'figure.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelcolor':NAVY,'xtick.color':GREY,'ytick.color':GREY,
    'axes.edgecolor':'#D1D9E6','grid.color':'#E5EAF2','grid.linewidth':0.6,
})

BASE = '/Users/kabir/Desktop/pro/credit_risk/'

# ── Load scored portfolio ──────────────────────────────────────────────────────
print("Loading scored portfolio...")
df = pd.read_csv(BASE + 'scored_portfolio.csv')
model = joblib.load(BASE + 'pd_model.pkl')

# ── Basel II Expected Loss Framework ──────────────────────────────────────────
# Expected Loss (EL) = PD × LGD × EAD
# LGD (Loss Given Default) assumptions by loan purpose
LGD_MAP = {
    'Car (new)':   0.35,   # Collateralised — lower LGD
    'Car (used)':  0.40,
    'Furniture':   0.55,   # Partially collateralised
    'TV/Radio':    0.65,
    'Appliances':  0.65,
    'Repairs':     0.70,
    'Education':   0.75,   # Unsecured — higher LGD
    'Business':    0.60,
    'Retraining':  0.75,
    'Others':      0.65,
}
DEFAULT_LGD = 0.55

df['lgd'] = df['purpose_label'].map(LGD_MAP).fillna(DEFAULT_LGD)
df['ead'] = df['credit_amount']  # EAD = full loan amount (simplification)
df['expected_loss'] = df['pd_score'] * df['lgd'] * df['ead']
df['el_rate'] = df['pd_score'] * df['lgd']

base_el       = df['expected_loss'].sum()
base_pd       = df['pd_score'].mean()
base_el_rate  = df['el_rate'].mean()
total_exposure = df['ead'].sum()

print(f"\n===== BASE CASE (No Stress) =====")
print(f"Total Portfolio Exposure:  DM {total_exposure:,.0f}")
print(f"Average PD:                {base_pd:.1%}")
print(f"Average EL Rate:           {base_el_rate:.1%}")
print(f"Total Expected Loss:       DM {base_el:,.0f}")
print(f"EL as % of Portfolio:      {base_el/total_exposure:.1%}")

# ── Stress Scenarios ───────────────────────────────────────────────────────────
# Macro stress scenarios calibrated to historical events
# PD multiplier = how much PDs increase under stress
# LGD multiplier = collateral value deterioration
# EAD multiplier = credit line drawdown (kept at 1 for term loans)

scenarios = {
    'Base Case': {
        'pd_multiplier':  1.00,
        'lgd_multiplier': 1.00,
        'description': 'Normal economic conditions',
        'color': BLUE
    },
    '2008 GFC': {
        'pd_multiplier':  2.50,   # PDs ~2.5x in severe recession
        'lgd_multiplier': 1.30,   # Collateral values fall 30%
        'description': 'Global Financial Crisis — severe recession, credit crunch',
        'color': RED
    },
    '2020 COVID': {
        'pd_multiplier':  1.75,   # Sharp but short shock
        'lgd_multiplier': 1.15,   # Moderate collateral impact
        'description': 'COVID-19 — sudden unemployment shock, govt support offsets',
        'color': GOLD
    },
    'Rising Rates (+300bps)': {
        'pd_multiplier':  1.40,   # Higher debt servicing costs
        'lgd_multiplier': 1.10,   # Slight collateral pressure
        'description': 'Rapid rate hike cycle — debt burden increases, defaults rise',
        'color': '#8E44AD'
    },
    'Mild Recession': {
        'pd_multiplier':  1.50,
        'lgd_multiplier': 1.10,
        'description': 'Moderate economic downturn — GDP -2%, unemployment +3%',
        'color': '#E67E22'
    },
}

# ── Run Stress Tests ───────────────────────────────────────────────────────────
results = []
grade_results = {}

for scenario, params in scenarios.items():
    stressed_pd  = np.minimum(df['pd_score'] * params['pd_multiplier'], 0.99)
    stressed_lgd = np.minimum(df['lgd'] * params['lgd_multiplier'], 0.99)
    stressed_el  = (stressed_pd * stressed_lgd * df['ead']).sum()
    stressed_pd_avg = stressed_pd.mean()
    stressed_el_rate = (stressed_pd * stressed_lgd).mean()
    capital_required = stressed_el * 1.5  # simplified: capital buffer = 1.5x EL

    results.append({
        'Scenario':          scenario,
        'Avg PD':            stressed_pd_avg,
        'Avg EL Rate':       stressed_el_rate,
        'Total EL (DM)':     stressed_el,
        'EL % Portfolio':    stressed_el / total_exposure,
        'vs Base (DM)':      stressed_el - base_el,
        'EL Increase %':     (stressed_el - base_el) / base_el,
        'Capital Required':  capital_required,
    })

    # Grade migration under stress
    stressed_grades = pd.cut(stressed_pd,
        bins=[0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 1.0],
        labels=['AAA','AA','A','BBB','BB','B','CCC'])
    grade_results[scenario] = stressed_grades.value_counts()

results_df = pd.DataFrame(results)

print("\n===== STRESS TEST RESULTS =====")
display_cols = ['Scenario','Avg PD','Avg EL Rate','EL % Portfolio','EL Increase %']
for _, row in results_df.iterrows():
    print(f"\n{row['Scenario']}:")
    print(f"  Avg PD:          {row['Avg PD']:.1%}")
    print(f"  Avg EL Rate:     {row['Avg EL Rate']:.1%}")
    print(f"  Total EL:        DM {row['Total EL (DM)']:,.0f}")
    print(f"  EL % Portfolio:  {row['EL % Portfolio']:.1%}")
    print(f"  vs Base:         +DM {row['vs Base (DM)']:,.0f} (+{row['EL Increase %']:.0%})")

# ── Grade Migration Matrix ─────────────────────────────────────────────────────
print("\n===== GRADE MIGRATION UNDER STRESS =====")
grade_order = ['AAA','AA','A','BBB','BB','B','CCC']
migration = pd.DataFrame({s: grade_results[s].reindex(grade_order).fillna(0)
                          for s in scenarios.keys()})
print(migration.to_string())

# ── Capital Adequacy ───────────────────────────────────────────────────────────
print("\n===== CAPITAL ADEQUACY ANALYSIS =====")
tier1_capital = total_exposure * 0.08  # 8% Tier 1 capital ratio (Basel III minimum)
print(f"Portfolio Exposure:    DM {total_exposure:,.0f}")
print(f"Tier 1 Capital (8%):   DM {tier1_capital:,.0f}")
for _, row in results_df.iterrows():
    buffer = tier1_capital - row['Total EL (DM)']
    adequate = "✅ ADEQUATE" if buffer > 0 else "❌ INSUFFICIENT"
    print(f"{row['Scenario']:<25} EL: DM {row['Total EL (DM)']:>10,.0f}  "
          f"Buffer: DM {buffer:>10,.0f}  {adequate}")

# ── Save results ───────────────────────────────────────────────────────────────
results_df.to_csv(BASE + 'stress_test_results.csv', index=False)
migration.to_csv(BASE + 'grade_migration.csv')
df.to_csv(BASE + 'final_portfolio.csv', index=False)

# ── Visualisations ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Credit Risk Stress Testing — Scenario Analysis',
             fontsize=15, fontweight='bold', color=NAVY)

scenario_names = list(scenarios.keys())
colors = [s['color'] for s in scenarios.values()]

# 1. EL by scenario
el_vals = results_df['Total EL (DM)'].values / 1e6
bars = axes[0,0].bar(range(len(scenario_names)), el_vals, color=colors, edgecolor='white', width=0.6)
axes[0,0].axhline(base_el/1e6, color=BLUE, ls='--', lw=1.2, alpha=0.5)
axes[0,0].set_xticks(range(len(scenario_names)))
axes[0,0].set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=8)
axes[0,0].set_ylabel('Expected Loss (DM millions)')
axes[0,0].set_title('Total Expected Loss by Scenario', fontsize=12, fontweight='bold', color=NAVY)
axes[0,0].grid(axis='y', alpha=0.5)
for bar, val in zip(bars, el_vals):
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'DM {val:.1f}M', ha='center', fontsize=8, color=NAVY, fontweight='bold')

# 2. Average PD by scenario
pd_vals = results_df['Avg PD'].values * 100
bars2 = axes[0,1].bar(range(len(scenario_names)), pd_vals, color=colors, edgecolor='white', width=0.6)
axes[0,1].axhline(base_pd*100, color=BLUE, ls='--', lw=1.2, alpha=0.5)
axes[0,1].set_xticks(range(len(scenario_names)))
axes[0,1].set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=8)
axes[0,1].set_ylabel('Average PD (%)')
axes[0,1].set_title('Average Portfolio PD by Scenario', fontsize=12, fontweight='bold', color=NAVY)
axes[0,1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
axes[0,1].grid(axis='y', alpha=0.5)

# 3. Grade migration heatmap
migration_pct = migration.div(migration.sum(axis=0), axis=1) * 100
im = axes[1,0].imshow(migration_pct.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=60)
axes[1,0].set_xticks(range(len(scenario_names)))
axes[1,0].set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=8)
axes[1,0].set_yticks(range(len(grade_order)))
axes[1,0].set_yticklabels(grade_order)
axes[1,0].set_title('Grade Migration Heatmap (%)', fontsize=12, fontweight='bold', color=NAVY)
for i in range(len(grade_order)):
    for j in range(len(scenario_names)):
        val = migration_pct.values[i, j]
        axes[1,0].text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=8,
                       color='white' if val > 35 else NAVY)
plt.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)

# 4. Capital adequacy waterfall
tier1 = tier1_capital / 1e6
el_scenario = results_df.set_index('Scenario')['Total EL (DM)'] / 1e6
cap_colors = [GREEN if tier1 - v > 0 else RED for v in el_scenario.values]
axes[1,1].barh(scenario_names, el_scenario.values, color=cap_colors, edgecolor='white')
axes[1,1].axvline(tier1, color=NAVY, lw=2, ls='--', label=f'Tier 1 Capital (DM {tier1:.1f}M)')
axes[1,1].set_xlabel('Expected Loss (DM millions)')
axes[1,1].set_title('Capital Adequacy vs Expected Loss', fontsize=12, fontweight='bold', color=NAVY)
axes[1,1].legend(frameon=False, fontsize=9)
axes[1,1].grid(axis='x', alpha=0.5)

plt.tight_layout()
plt.savefig(BASE + 'fig3_stress_test.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: fig3_stress_test.png")
print("\nPhase 3 complete!")