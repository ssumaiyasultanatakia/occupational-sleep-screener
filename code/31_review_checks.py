# -*- coding: utf-8 -*-
"""Extra checks for the camera-ready (pre-submission audit):
 (1) dataset edition: variables used are identical in SN 9511 edition 1 (.tab) and edition 2 (v2 .sav)?
 (2) full design-based inference: calibration weights + PSU (samplingpointid) clustering
 (3) outcome-threshold sensitivity (daily only; any difficulty)
 (4) extended adjustment set (occupation, sector, country fixed effects)
 (5) PSU-cluster bootstrap CI for the proposed screener AUROC
Outputs: results/camera_ready/review_checks.json
"""
import os, json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
import statsmodels.api as sm, pyreadstat
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

OUT = r'F:\psychology\journal3\results\camera_ready'
P24 = r'F:\psychology\EWCS 2024-1\European Work Condition Survey\9511tab\UKDA-9511-tab\tab\ewcs24_dataset_ukds.tab'
PV2 = r'F:\psychology\EWCS 2024-2\UKDA-9511-spss\spss\spss28\ewcs24_dataset_ukda_v2.sav'
SEED = 0; rng = np.random.RandomState(SEED); R = {}
use = ['uniquerespid', 'contact_out_wh', 'freetime_work', 'wlb_worry', 'wlb_tired', 'norest', 'night', 'longday', 'loc_home',
       'computer', 'usual_hours_week', 'sleeping_difficulties', 'wellbeing', 'quit_job', 'health_anxiety', 'country',
       'age', 'sex_id', 'isced', 'calweight', 'stratumid', 'samplingpointid', 'ISCO_1', 'NACE1']
def num(s): return pd.to_numeric(s.astype(str).str.strip().replace({'': np.nan}), errors='coerce')

# (1) edition comparison
v1 = pd.read_csv(P24, sep='\t', usecols=use, encoding='latin-1', low_memory=False)
v2, _ = pyreadstat.read_sav(PV2, usecols=[c for c in use], encoding='latin1')
v1 = v1.set_index('uniquerespid').sort_index(); v2 = v2.set_index('uniquerespid').sort_index()
same_ids = bool((v1.index.astype(str) == v2.index.astype(str)).all()) if len(v1) == len(v2) else False
diff = {}
for c in use[1:]:
    if c in ('stratumid', 'samplingpointid'):
        a = v1[c].astype(str).str.strip().values; b = v2[c].astype(str).str.strip().values
        diff[c] = int((a != b).sum())
    else:
        a = num(v1[c]).values; b = pd.to_numeric(v2[c], errors='coerce').values
        diff[c] = int((~((a == b) | (np.isnan(a) & np.isnan(b)) | (np.isclose(a, b, equal_nan=True)))).sum())
R['edition_check'] = {'n_v1': len(v1), 'n_v2': len(v2), 'same_ids_order': same_ids, 'n_cells_differing': diff}
print(R['edition_check'])

# common prep (edition 1, as in 30_camera_ready.py)
df = v1.reset_index()
for c in use[1:]:
    if c not in ('stratumid', 'samplingpointid'): df[c] = num(df[c])
df['night'] = df['night'].replace(-889, 0); df['longday'] = df['longday'].replace(-889, 0)
for c in use[1:]:
    if c not in ('night', 'longday', 'stratumid', 'samplingpointid'): df.loc[df[c] < 0, c] = np.nan
df['norest_b'] = np.where(df['norest'].isna(), np.nan, (df['norest'] == 1).astype(float))
E = df[df.contact_out_wh.notna()].copy()
def zs(s): return (s - s.mean()) / s.std()
Z = pd.DataFrame({c: zs(6 - E[c]) for c in ['contact_out_wh', 'freetime_work', 'wlb_worry']})
E['DTI'] = zs(Z.mean(axis=1))
sd = E['sleeping_difficulties']
E['sleep_bad'] = np.where(sd.isna(), np.nan, (sd <= 2).astype(float))
E['sleep_daily'] = np.where(sd.isna(), np.nan, (sd == 1).astype(float))
E['sleep_any'] = np.where(sd.isna(), np.nan, (sd <= 3).astype(float))
E['psu'] = E['samplingpointid'].astype(str) + '_' + E['country'].astype(str)
R['design'] = {'n_psu_employees': int(E['psu'].nunique()), 'n_strata': int(E['stratumid'].astype(str).nunique()),
               'median_psu_size': float(E.groupby('psu').size().median())}
cov = ['age', 'sex_id', 'isced', 'usual_hours_week']
def fit(y, extra_cat=None, weights=False, cluster='country', x='DTI'):
    cols = list(dict.fromkeys([y, x, 'country', 'psu', 'calweight'] + cov + (extra_cat or [])))
    d = E[cols].dropna(subset=[y, x]).copy()
    for c in cov: d[c] = d[c].fillna(d[c].median())
    X = d[[x] + cov]
    for c in (extra_cat or []):
        v = d[c].fillna(-1).round().astype(int); vc = v.value_counts(); v = v.where(v.map(vc) >= 100, -1)
        X = pd.concat([X, pd.get_dummies(v, prefix=c, drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    vw = (d['calweight'] / d['calweight'].mean()).values if weights else None
    m = sm.GLM(d[y], X, family=sm.families.Binomial(), var_weights=vw).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d[cluster])[0]})
    ci = m.conf_int().loc[x]
    return {'OR': round(float(np.exp(m.params[x])), 3), 'lo': round(float(np.exp(ci[0])), 3), 'hi': round(float(np.exp(ci[1])), 3), 'n': int(len(d))}
R['dti_sleep'] = {'unweighted_country_cluster': fit('sleep_bad'),
                  'weighted_psu_cluster(design-based)': fit('sleep_bad', weights=True, cluster='psu'),
                  'weighted_country_cluster': fit('sleep_bad', weights=True),
                  'extended_adjustment(+occupation,+sector,+country FE)': fit('sleep_bad', extra_cat=['ISCO_1', 'NACE1', 'country']),
                  'outcome_daily_only': fit('sleep_daily'), 'outcome_any_difficulty': fit('sleep_any')}
# category ORs, design-based
dd = E[['sleep_bad', 'contact_out_wh', 'psu', 'calweight', 'country'] + cov].dropna(subset=['sleep_bad']).copy()
for c in cov: dd[c] = dd[c].fillna(dd[c].median())
D = pd.get_dummies(dd['contact_out_wh'].astype(int), prefix='c').astype(float).drop(columns=['c_5'])
m = sm.GLM(dd['sleep_bad'], sm.add_constant(pd.concat([D, dd[cov]], axis=1)), family=sm.families.Binomial(),
           var_weights=(dd.calweight / dd.calweight.mean()).values).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(dd['psu'])[0]})
R['category_OR_design_based'] = {k: [round(float(np.exp(m.params[k])), 2), round(float(np.exp(m.conf_int().loc[k, 0])), 2),
                                     round(float(np.exp(m.conf_int().loc[k, 1])), 2)] for k in ['c_4', 'c_3', 'c_2', 'c_1']}
E2 = E.copy(); E2['age'] = pd.to_numeric(v2.loc[E2['uniquerespid'].values, 'age'].values, errors='coerce'); E2.loc[E2.age < 0, 'age'] = np.nan
_E = E; E = E2; R['dti_sleep']['edition2_ages'] = fit('sleep_bad'); E = _E
print(R['dti_sleep']); print(R['category_OR_design_based'])

# proposed screener: OOF AUROC for alternative outcomes + PSU-cluster bootstrap
B8 = ['contact_out_wh', 'freetime_work', 'norest_b', 'night', 'longday', 'loc_home', 'computer', 'usual_hours_week']
def oof(y):
    d = E[E[y].notna()]; X = d[B8].values.astype(float); yy = d[y].values.astype(int); p = np.zeros(len(yy))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, yy):
        med = np.nanmedian(X[tr], 0); Xtr = np.where(np.isnan(X[tr]), med, X[tr]); Xte = np.where(np.isnan(X[te]), med, X[te])
        sc = StandardScaler().fit(Xtr); p[te] = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), yy[tr]).predict_proba(sc.transform(Xte))[:, 1]
    return yy, p, d
R['B8_alt_outcomes'] = {}
for y in ['sleep_bad', 'sleep_daily', 'sleep_any']:
    yy, p, d = oof(y)
    R['B8_alt_outcomes'][y] = {'prev': round(100 * yy.mean(), 1), 'auroc': round(roc_auc_score(yy, p), 3)}
    if y == 'sleep_bad':
        g = d['psu'].values; ug = np.unique(g); pos = {k: np.where(g == k)[0] for k in ug}; a = []
        for _ in range(1000):
            b = np.concatenate([pos[k] for k in rng.choice(ug, len(ug), True)]); a.append(roc_auc_score(yy[b], p[b]))
        R['B8_auroc_psu_cluster_ci'] = [round(float(v), 3) for v in np.percentile(a, [2.5, 97.5])]
print(R['B8_alt_outcomes'], R.get('B8_auroc_psu_cluster_ci'))
json.dump(R, open(os.path.join(OUT, 'review_checks.json'), 'w', encoding='utf-8'), indent=2)
print('DONE')
