# 🫀 Aayuro — Health Risk Intelligence

> **आयुर्** · *āyur* · Sanskrit/Sinhala for *lifespan*

Aayuro is a personalised mortality risk prediction system built on real epidemiological data. It uses a Cox Proportional Hazards model trained on the US NHANES 2013–2014 dataset (linked to 7-year mortality follow-up) to estimate a person's 5-year and 10-year survival probability — and explains exactly which factors are driving their risk using SHAP.

**🔗 Live App:** [aayuro.streamlit.app](https://aayuro-dpnsedzebffd5y8d8d4jnn.streamlit.app/)

---

## 📊 Key Statistics

| Metric | Value |
|---|---|
| 📁 Dataset | NHANES 2013–2014 + Linked Mortality File |
| 👥 Participants | 6,100 eligible adults |
| 💀 Death events | 467 recorded deaths |
| ⏱️ Follow-up period | Up to 83 months (~7 years) |
| 🧠 Model | Cox Proportional Hazards (lifelines) |
| 🎯 C-index (accuracy) | **0.8315** (excellent — published clinical scores avg ~0.75) |
| 🔍 Explainability | SHAP via GradientBoosting surrogate (R² = 0.9943) |
| 📈 Features used | 10 health, lifestyle & socioeconomic factors |
| 🖥️ Deployment | Streamlit Community Cloud |

---

## 🧬 How It Works

```
Raw NHANES Survey Data (.XPT files)          Linked Mortality File (.dat)
       (health, lifestyle, demographics)      (who died + when)
                        └──────────── merged on SEQN ────────────┘
                                             ↓
                              nhanes_merged_day1.csv
                              (6,100 people × 16 features)
                                             ↓
                         Cox Proportional Hazards Model
                         (survival analysis, handles censoring)
                                             ↓
                    ┌────────────────────────┴────────────────────────┐
                    │                                                  │
             Risk Score (0–100)                            SHAP Explainability
          5-yr & 10-yr survival %                   (why is THIS person's score high?)
                    │                                                  │
                    └────────────────────────┬────────────────────────┘
                                             ↓
                              Personalised Recommendations
                                  (modifiable factors only)
                                             ↓
                                   Streamlit Web App
```

---

## 🔬 What the Model Learned

### Feature Importance (SHAP values — higher = more impact on risk)

| Rank | Feature | SHAP Score | Direction |
|---|---|---|---|
| 1 | **Age** | 2.5529 | ↑ Raises risk |
| 2 | **Diabetes** | 0.6375 | ↑ Raises risk |
| 3 | **Ever Smoked** | 0.6323 | Complex (survival bias) |
| 4 | **Income Level** | 0.6285 | ↓ Low income raises risk |
| 5 | **Sedentary Hours/Day** | 0.5069 | ↑ Raises risk |
| 6 | **High Blood Pressure** | 0.4417 | ↑ Raises risk |
| 7 | **Physically Active** | 0.3643 | ↓ Lowers risk |
| 8 | **Education Level** | 0.2062 | ↓ Higher = lower risk |
| 9 | **Sex (Male)** | 0.1824 | Slight risk increase |
| 10 | **BMI** | 0.1274 | Indirect effect |

### Hazard Ratios (Cox Model)

| Factor | Hazard Ratio | Meaning |
|---|---|---|
| High Blood Pressure | 1.242 | +24% mortality risk |
| Diabetes | 1.399 | +40% mortality risk |
| Sedentary hours | 1.033 | +3.3% risk per extra hour/day |
| Physical activity | 0.758 | −24% mortality risk |
| Income ratio | 0.927 | −7.3% risk per unit increase |

### Example Predictions

| Person | 5-yr Survival | Risk Score | Category |
|---|---|---|---|
| Healthy 30-year-old woman | 97.5% | 2.5/100 | 🟡 Moderate |
| 50-year-old male smoker | 93.3% | 6.7/100 | 🟡 Moderate |
| 65-year-old diabetic, high BP | 81.5% | 18.5/100 | 🔴 High |
| 72-year-old active woman | 91.8% | 8.2/100 | 🔴 High |

---

## 📅 Build Journey (5-Day Holiday Project)

| Day | What was built |
|---|---|
| **Day 1** | Downloaded NHANES data, parsed the mortality file, merged 6 survey files on SEQN → clean dataset |
| **Day 2** | Kaplan-Meier survival curves for 8 risk factors, log-rank tests for statistical significance |
| **Day 3** | Cox Proportional Hazards model, hazard ratio analysis, risk scores (0–100) per person |
| **Day 4** | SHAP explainability via surrogate model, individual risk explanations, recommendations engine |
| **Day 5** | Full Streamlit web app with dark theme, live predictions, SHAP chart, personalised recommendations |

---

## 🛠️ Tech Stack

```
Python 3.14
├── pandas              — data loading, merging, wrangling
├── numpy               — numerical operations
├── matplotlib          — survival curves, hazard ratio plots, SHAP charts
├── lifelines           — Cox Proportional Hazards model, Kaplan-Meier
├── scikit-learn        — GradientBoosting surrogate model, train/test split
├── shap                — SHAP explainability values
├── streamlit           — web app framework
└── requests            — downloading NHANES .XPT files
```

---

## 📁 Project Structure

```
aayuro/
├── app.py                        ← Streamlit web app (main entry point)
├── requirements.txt              ← Python dependencies
├── nhanes_model_day4.csv         ← Processed dataset used by the app
│
├── day1_nhanes_setup.py          ← Data download, merge, feature extraction
├── day2_survival_curves.py       ← Kaplan-Meier survival curve plots
├── day3_cox_model.py             ← Cox model training + risk scoring
├── day4_shap.py                  ← SHAP explainability + recommendations
│
└── plots/                        ← Generated charts
    ├── 01_age_groups.png
    ├── 02_smoking.png
    ├── 03_bmi.png
    ├── 04_diabetes.png
    ├── 05_blood_pressure.png
    ├── 06_physical_activity.png
    ├── 07_sex.png
    ├── 08_income.png
    ├── 09_hazard_ratios.png
    ├── 10_risk_scores.png
    ├── 11_global_shap.png
    └── 12_shap_beeswarm.png
```

---

## 📦 Data Sources

| File | Source | Description |
|---|---|---|
| `DEMO_H.XPT` | CDC NHANES 2013–2014 | Demographics: age, sex, income, education |
| `BMX_H.XPT` | CDC NHANES 2013–2014 | Body measures: BMI, weight, height, waist |
| `BPX_H.XPT` | CDC NHANES 2013–2014 | Blood pressure readings |
| `SMQ_H.XPT` | CDC NHANES 2013–2014 | Smoking history |
| `DIQ_H.XPT` | CDC NHANES 2013–2014 | Diabetes diagnosis |
| `PAQ_H.XPT` | CDC NHANES 2013–2014 | Physical activity & sedentary time |
| `NHANES_2013_2014_MORT_2019_PUBLIC.dat` | CDC Linked Mortality File | Vital status + follow-up months |

**Download from:** https://www.cdc.gov/nchs/data-linkage/mortality-public.htm

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/maneesha-bogahawatta/aayuro.git
cd aayuro

# Install dependencies
pip3 install -r requirements.txt

# Run the app
streamlit run app.py
```

> **Note:** The app loads `nhanes_model_day4.csv` which is included in the repo.
> First run takes ~30 seconds to train the model — subsequent runs are instant (cached).

---

## ⚠️ Limitations & Disclaimers

- **Educational tool only** — this is NOT a medical device and does NOT replace professional medical advice
- Trained on **US adults (NHANES 2013–2014)**. Risk estimates may not perfectly apply to other populations including Sri Lanka
- The model predicts **statistical population-level risk**, not individual fate. Many people with high scores live long, healthy lives
- Some variables have missing data (e.g. smoking had 57.8% missing — excluded from final model)
- **Survival bias** affects the smoking result — the "ever smoked" coefficient should be interpreted cautiously
- C-index of 0.83 is strong but the model should be validated on an independent dataset before clinical use

---

## 🗺️ Roadmap

- [ ] Add more NHANES cycles (1999–2018) for 10× more training data
- [ ] Upgrade to Random Survival Forest for better non-linear capture
- [ ] Add cause-specific risk (heart disease vs cancer vs other)
- [ ] "What-if" simulator — *"if I quit smoking, my score drops by X"*
- [ ] PDF report download
- [ ] Recalibrate model with local Sri Lankan health data
- [ ] Mobile-optimised layout

---

## 👨‍💻 Author

Built by **Maneesha Bogahawatta** during a focused 5-day holiday project.

- GitHub: [@maneesha-bogahawatta](https://github.com/maneesha-bogahawatta)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

> *"Statistics are human beings with the tears wiped away."* — Paul Brodeur
>
> Aayuro puts the humanity back in — showing not just the number, but what's driving it and what you can do about it.
