# MONDY Markers Protocol

## Overview  

This repository contains the data analysis as described in the **methods protocol** for the project:  

> **Can AI-enabled analyses of smartphones and wearable data collected over one year provide patients suffering from depression with an objective marker of disease severity?**  

The protocol outlines the predefined methodology for analyzing **long-term smartphone and wearable sensor data** to detect objective markers of depression severity.  

This protocol is **registered in the Open Science Framework (OSF)**:  
➡️ [https://doi.org/10.17605/OSF.IO/XDU3P](https://doi.org/10.17605/OSF.IO/XDU3P)  

It builds upon the **original study protocol by Reich et al. (2025)**:  
➡️ *Links between self-monitoring data collected through smartphones and smartwatches and the individual disease trajectories of adult patients with depressive disorders: Study protocol of a one-year observational trial*  
[https://doi.org/10.1016/j.conctc.2025.101492](https://doi.org/10.1016/j.conctc.2025.101492)  

---

## Study Background  

This proof-of-concept study investigates whether **AI-based analysis of individual time series data** (smartphone usage, voice, activity, self-ratings) can provide **objective indicators of depressive symptom severity** in patients with clinically diagnosed depression.  

Data were collected as part of the **MONDY exploratory n-of-1 trial**  
- **N = 15 adults** with recurrent major depression  
- Monitoring period: **up to 12 months**  
- Daily self-reports: **PHQ-2** (evening)  
- Weekly self-reports: **PHQ-9**  
- Wearables: **Samsung Galaxy Watch 5**  
- Smartphone app: **iTrackDepression**  

Clinical trial registration: [DRKS00032618](https://drks.de/search/en/trial/DRKS00032618)  

---

## Research Questions  

1. **Prediction Feasibility:**  
   Can passive, individual-level time-series data predict **daily depression severity (PHQ-2)**?  

2. **Feature Contributions:**  
   For patients where prediction is feasible, which **sensor or app-derived data sources** contribute most?  

3. **Nonlinear Models:**  
   Does using **nonlinear ensemble models** (e.g., Random Forests) improve prediction accuracy and shift dominant predictors?  

---

## Data Sources  

The dataset integrates **active self-ratings** and **passive smartphone & wearable sensors**:  

- **Phone & Communication Data**  
  - Total call duration, frequency, contacts, missed calls  
- **Speech & Acoustic Features**  
  - Intensity, pitch, jitter, shimmer, MFCCs  
- **Physical Activity**  
  - Daily steps, ENMO (Euclidean Norm minus One), sedentary/light/moderate-vigorous activity  
- **App Usage & Social Interaction**  
  - Foreground time for communication/social apps, network traffic  
- **Heart-Rate Variability (HRV)**  
  - 35 time & frequency-domain HRV features, reduced to 15 non-redundant indices  
- **Sleep Data**  
  - HRV-based automated sleep staging  

All passive data were restricted to **daytime (06:00–18:00)** to precede evening PHQ-2 ratings.  

---

## Data Processing  

- **Imputation:** Missing values handled via **Multivariate Imputation by Chained Equations (MICE)**  
- **Scaling:** Min-Max scaling for comparability  
- **Partitioning:** 70% training / 30% test split with 5-fold cross-validation for hyperparamter tuning 
- **Symptom Variability Filtering:** Low-variance participants flagged as non-responders  

---

## Modeling  

Two complementary regression approaches were used:  

- **Elastic Net Regression (linear)**  
  - Combines L1 & L2 regularization  
  - Identifies interpretable feature associations using model coefficients

- **Random Forest Regression (nonlinear)**  
  - Captures complex, nonlinear interactions  
  - Feature importance via **Tree SHAP**  

**Performance Metrics:**  
- Mean Absolute Error (MAE)  
- R²
- Comparison of clinically meaningful thresholds (e.g., R² > 0.3)  

---

## Evaluation  

- **RQ1:** Proportion of patients with models exceeding predefined accuracy thresholds  
- **RQ2:** Aggregated feature importance from linear models  
- **RQ3:** Added value of nonlinear models (improved accuracy & feature shifts)  

---

## Deviations from Original Protocol  

- Multilevel models replaced by **idiographic Elastic Net models** for individual-level focus  
- Tree SHAP used for Random Forests (instead of Kernel SHAP) for computational efficiency  

---

## Tools & Libraries  

- **Python**  
- **scikit-learn** for machine learning pipelines  
- **openSMILE** for acoustic feature extraction  
- **NeuroKit2** for HRV processing  
- **SciKit-Digital-Health (SKDH)** for accelerometer processing  
- **SleepECG** for HRV-based sleep staging  

---

## Citation  

If you use this protocol, please cite:  

> Hegerl U., Schreynemackers S., Petrovic M., Ludwig S., Leimhofer J., Dominik A., Heider D., Reich H., & MONDY Consortium.  
> *Can AI-enabled analyses of smartphones and wearable data collected over one year provide patients suffering from depression with an objective marker of disease severity?*  
> OSF, 2025. [https://doi.org/10.17605/OSF.IO/XDU3P](https://doi.org/10.17605/OSF.IO/XDU3P)

---

## Acknowledgements  

This study is part of the **MONDY (Secure and Open Platform for AI-based Healthcare Apps) Consortium**.  
Special thanks to all contributors, researchers, and participants who made this work possible.  

---

## License  

This repository is licensed under the **MIT License**.  

You are free to use, modify, and distribute this work, provided that the original authors are credited. See the [LICENSE](LICENSE) file for details.  

---

## Links  

- **OSF Protocol:** [https://doi.org/10.17605/OSF.IO/XDU3P](https://doi.org/10.17605/OSF.IO/XDU3P)  
- **Study Protocol by Reich et al.:** [https://doi.org/10.1016/j.conctc.2025.101492](https://doi.org/10.1016/j.conctc.2025.101492)  
- **Clinical Trial Registration:** [DRKS00032618](https://drks.de/search/en/trial/DRKS00032618)  

---
