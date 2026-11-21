# Auditable behavioural screener for occupational sleep risk (ICECE 2026, paper 640)

Code and aggregate results for *An Auditable Behavioural Screener for Occupational Sleep Risk
Validated Across 35 European Countries and Two Survey Waves*.

## Data (not included)
The microdata cannot be redistributed. Register with the UK Data Service and download:
- EWCS 2024, SN 9511 (tab format, edition 1; edition 2 gives the same results)
- EWCS 2015, SN 8098 (SPSS format)

Edit the data paths at the top of each script (`P24`, `P15` in `30_camera_ready.py`; `P24`, `PV2` in
`31_review_checks.py`; `P24` in `32_policy_eu27_and_h7.py`) and the output folder (`ROOT`/`OUT`) to your own folders.

Note: the right-to-disconnect results reported in the paper (EU-27 primary analysis, contact-only test, all codings)
come from `32_policy_eu27_and_h7.py`. The policy block inside `30_camera_ready.py` is an earlier all-35-country
version, kept for transparency; it is the "all 35" sensitivity analysis.

## Reproduce
```
pip install -r requirements.txt
python code/30_camera_ready.py     # all paper results + figures (about 15 min on a laptop CPU)
python code/31_review_checks.py    # survey-weighted/PSU, edition and cut-off sensitivity checks
python code/32_policy_eu27_and_h7.py  # right-to-disconnect (EU-27 primary, contact-only) + seven-item 2015 model
```
Python 3.9.13, random seed 0 throughout.

## What is where
| File | Content |
|---|---|
| `code/30_camera_ready.py` | data preparation, DTI, association models, screener training, nested thresholds, decision curves, leave-one-country-out, region transfer, subgroups, EWCS 2015 validation, right-to-disconnect tests, figures |
| `code/31_review_checks.py` | survey-weighted, PSU-clustered inference, edition 1 vs 2 check, outcome cut-offs, extended adjustment |
| `code/32_policy_eu27_and_h7.py` | right-to-disconnect moderation (EU-27 primary, all-35 sensitivity; DTI and contact-only; wild-cluster bootstrap) and the harmonised seven-item model used for EWCS 2015 |
| `results/policy_h7.json` | all policy interaction estimates and the seven-item model equation |
| `results/supplement.tex` | the supplementary tables S1-S4 sent to reviewers |
| `results/proposed_screener_equation.json` | the complete proposed screener (intercept, coefficients on raw codes, imputation medians) |
| `results/metrics_cr.json` | every number reported in the paper |
| `results/per_country.csv` | per-country n, events, prevalence, AUROC with 95% CI, O/E, sensitivity, specificity, referral share |
| `results/subgroup.csv` | subgroup AUROC, sensitivity, specificity, referral fraction, calibration |
| `results/right_to_disconnect_coding.md` | legal coding, in-force dates and scope, with source |

## Using the screener
`logit p = 0.4077 - 0.1380*contact - 0.1906*freetime + 0.1704*rest_under_11h + 0.0288*night
          + 0.0218*long_days - 0.0080*home - 0.0083*device - 0.0115*weekly_hours`

Frequency items use EWCS codes (1 = daily/always ... 5 = never; device use 1-7); rest_under_11h is 1 if yes;
night and long days are times per month (0 if never). Recalibrate the intercept to local prevalence before use.
The screener is a first flag for human follow-up, not a diagnosis.
