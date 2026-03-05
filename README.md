# Credit Risk & Stress Testing Model
### PD Scoring · LGD/EAD Framework · Macro Stress Scenarios | Basel II | German Credit Dataset

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Basel%20II-navy?style=flat-square)

> A full credit risk pipeline built on the **UCI German Credit Dataset** (1,000 loans). Implements industry-standard **Probability of Default (PD)** scoring using Weight of Evidence (WoE) features, an **Expected Loss (EL) = PD × LGD × EAD** framework, and **macro stress testing** across five historical and hypothetical scenarios calibrated to Basel II.

---

## Results at a Glance

| Metric | Value | Benchmark |
|--------|:-----:|:---------:|
| AUC-ROC | **0.833** | >0.70 (industry standard) |
| Gini Coefficient | **0.666** | >0.40 (strong) |
| 5-Fold CV AUC | **0.806 ± 0.019** | Stable — no overfitting |
| Brier Score | **0.169** | 0.25 = random |
| Default Recall | **85%** | Captures 85% of actual defaults |

---

## Stress Test Results

| Scenario | Avg PD | Total EL | EL % Portfolio | vs Base |
|----------|:------:|:--------:|:--------------:|:-------:|
| **Base Case** | 44.2% | DM 862,226 | 26.4% | — |
| **2008 GFC** | 76.0% | DM 1,805,278 | 55.2% | **+109%** |
| **2020 COVID** | 66.0% | DM 1,422,734 | 43.5% | +65% |
| **Rising Rates (+300bps)** | 58.4% | DM 1,225,613 | 37.5% | +42% |
| **Mild Recession** | 60.9% | DM 1,270,345 | 38.8% | +47% |

> Under GFC conditions, expected losses more than **double (+109%)** and CCC-rated loans increase from **462 → 792 (+71%)** — highlighting the importance of maintaining capital buffers above the regulatory minimum during benign periods.

---

## Grade Migration Under Stress

| Grade | Base Case | 2008 GFC | 2020 COVID | Rising Rates | Mild Recession |
|-------|:---------:|:--------:|:----------:|:------------:|:--------------:|
| AAA | 27 | 1 | 5 | 9 | 6 |
| AA | 71 | 11 | 32 | 45 | 43 |
| A | 63 | 29 | 34 | 57 | 49 |
| BBB | 84 | 24 | 53 | 45 | 44 |
| BB | 127 | 65 | 71 | 107 | 103 |
| B | 166 | 78 | 118 | 134 | 127 |
| **CCC** | **462** | **792** | **687** | **603** | **628** |

---

## Methodology

### Phase 1 — Exploratory Data Analysis
- 1,000 consumer loans, 20 features, 30% default rate
- Segment analysis by checking account status, employment, loan purpose
- Key finding: no-account holders default at 11.7% vs overdrawn accounts at 49.3% (4x difference)

### Phase 2 — Feature Engineering & PD Model
**Weight of Evidence (WoE) encoding** — industry-standard Basel-compliant transformation for categorical variables:

$$WoE_i = \ln\left(\frac{\text{Distribution of Events}_i}{\text{Distribution of Non-Events}_i}\right)$$

**Information Value (IV)** scores rank predictive power:

| Feature | IV | Power |
|---------|:--:|:-----:|
| Checking Account | 0.666 | Very Strong |
| Credit History | 0.293 | Medium |
| Savings Account | 0.196 | Medium |
| Loan Purpose | 0.169 | Medium |
| Property | 0.113 | Medium |
| Job Type | 0.009 | Useless |

Logistic regression with L2 regularisation and class balancing, validated via 5-fold stratified cross-validation.

### Phase 3 — Expected Loss Framework
Basel II three-component model:

$$EL = PD \times LGD \times EAD$$

- **PD** — output of logistic regression scorecard
- **LGD** — varies by loan purpose (35% for secured car loans → 75% for unsecured education loans)
- **EAD** — full loan principal

**Stress multipliers calibrated to historical events:**
- 2008 GFC: PD × 2.5, LGD × 1.30
- 2020 COVID: PD × 1.75, LGD × 1.15
- Rising Rates +300bps: PD × 1.40, LGD × 1.10
- Mild Recession: PD × 1.50, LGD × 1.10

### Phase 4 — Excel Report
Professional 5-sheet workbook: Executive Summary · Scored Portfolio · Stress Test · Model Performance · Charts

---

## Project Structure

```
credit-risk-model/
│
├── eda.py                                 # Data loading, EDA, segment analysis
├── pd_model.py                            # WoE encoding, logistic regression, risk grades
├── stress_test.py                         # EL framework, stress scenarios, capital adequacy
├── excel_report.py                        # Professional Excel workbook generation
│
├── figures/
│   ├── fig1_eda.png                       # 6-panel EDA chart
│   ├── fig2_pd_model.png                  # ROC curve, PD distribution, grade distribution, IV
│   └── fig3_stress_test.png               # Stress scenarios, grade migration heatmap
│
├── Credit_Risk_Stress_Testing_Model.xlsx  # Full Excel report (5 sheets)
└── README.md
```

---

## Setup & Usage

```bash
git clone https://github.com/kabirguglani/credit-risk-model.git
cd credit-risk-model

pip install pandas numpy scikit-learn matplotlib seaborn openpyxl joblib
```

Run in order:

```bash
python eda.py            # Downloads UCI data, generates fig1_eda.png
python pd_model.py       # Trains PD model, generates fig2_pd_model.png
python stress_test.py    # Runs stress scenarios, generates fig3_stress_test.png
python excel_report.py   # Builds full Excel report
```

---

## Key Observations

**Checking account is the dominant predictor** — IV of 0.666 (Very Strong). Overdrawn accounts default at 49.3% vs 11.7% for no-account holders — a 4x spread that single-handedly drives model discrimination.

**Model well above industry threshold** — AUC 0.833 vs industry benchmark of 0.70+. Stable across 5-fold CV (0.806 ± 0.019), confirming no overfitting.

**GFC scenario is non-linear** — EL increases +109% despite PDs only multiplying 2.5x, because LGD also rises 30% simultaneously. The combined PD × LGD shock is multiplicative, not additive — a key Basel II insight.

**Capital adequacy fails under all scenarios** — the 8% Tier 1 minimum is designed for diversified institutional portfolios. A consumer loan book of this risk profile requires economic capital buffers of 3–4x the regulatory minimum.

---

## Basel II / Academic Context

- **Basel II IRB Approach** — EL = PD × LGD × EAD framework
- **WoE/IV methodology** — Siddiqi (2006), *Credit Risk Scorecards*
- **Stress testing** — EBA/ECB ICAAP guidelines, Fed DFAST framework
- **Logistic regression scorecard** — industry standard for retail credit since the 1980s

---

## Limitations & Future Work

- Sample size (N=1,000) — illustrative; production models use 50,000+ observations
- Price-based LGD proxies — production would use workout recovery data
- Single model — ensemble methods (XGBoost, Random Forest) typically improve AUC by 3–5%
- Future: through-the-cycle PD vs point-in-time PD · macro-conditional LGD · IFRS 9 ECL calculation

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Logistic regression, cross-validation, ROC metrics |
| `pandas` / `numpy` | Data wrangling, WoE/IV computation |
| `matplotlib` / `seaborn` | All visualisations |
| `openpyxl` | Professional Excel report |
| `joblib` | Model serialisation |

---

## Author

**Kabir Guglani**
MS Financial Analysis · Temple University ·
CFA Level 2 Candidate

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kabir-guglani/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:kabir.guglani@temple.edu)

---

*For academic purposes only. Not investment or credit advice.*
