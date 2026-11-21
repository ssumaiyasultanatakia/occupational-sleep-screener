# -*- coding: utf-8 -*-
"""Camera-ready round 3:
 (1) right-to-disconnect with EU-27 as PRIMARY population (non-EU countries unclassified: the legal source covers EU only);
     all six codings + wild-cluster bootstrap; also all-35 as sensitivity;
 (2) contact-only moderation (standardised q49 score x coverage) in addition to DTI x coverage;
 (3) harmonised seven-item model (2015 validation) coefficients, back-transformed to raw units.
Output: results/camera_ready/policy_h7.json
"""
import os, json, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = r'F:\psychology\journal3\results\camera_ready'
P24 = r'F:\psychology\EWCS 2024-1\European Work Condition Survey\9511tab\UKDA-9511-tab\tab\ewcs24_dataset_ukds.tab'
SEED = 0; R = {}
use = ['contact_out_wh', 'freetime_work', 'wlb_worry', 'norest', 'night', 'longday', 'loc_home', 'computer',
       'usual_hours_week', 'sleeping_difficulties', 'wellbeing', 'country_code', 'age', 'sex_id', 'isced', 'country']
df = pd.read_csv(P24, sep='\t', usecols=use, encoding='latin-1', low_memory=False)
def num(s): return pd.to_numeric(s.astype(str).str.strip().replace({'': np.nan}), errors='coerce')
for c in use:
    if c != 'country_code': df[c] = num(df[c])
df['night'] = df['night'].replace(-889, 0); df['longday'] = df['longday'].replace(-889, 0)
for c in use:
    if c not in ('night', 'longday', 'country_code'): df.loc[df[c] < 0, c] = np.nan
df['norest_b'] = np.where(df['norest'].isna(), np.nan, (df['norest'] == 1).astype(float))
E = df[df.contact_out_wh.notna()].copy()
sd = E['sleeping_difficulties']; E['sleep_bad'] = np.where(sd.isna(), np.nan, (sd <= 2).astype(float))
def zs(s): return (s - s.mean()) / s.std()
E['DTI'] = zs(pd.DataFrame({c: zs(6 - E[c]) for c in ['contact_out_wh', 'freetime_work', 'wlb_worry']}).mean(axis=1))
E['CONTACT'] = zs(6 - E['contact_out_wh'])
cov = ['age', 'sex_id', 'isced', 'usual_hours_week']
code = E['country_code']
NONEU = {'NO', 'CH', 'AL', 'BA', 'ME', 'MK', 'RS', 'XK'}
LAW = {'BE': 'general', 'HR': 'general', 'FR': 'general', 'GR': 'telework', 'IT': 'telework', 'LU': 'ict', 'PT': 'ict', 'SK': 'telework', 'ES': 'ict'}
home = (E['loc_home'] <= 3); ict = (E['computer'] <= 5)
SPEC = {
    'primary_statutory9': code.isin(LAW),
    'Ireland_covered': code.isin(set(LAW) | {'IE'}),
    'all_worker_laws_only': code.isin({k for k, v in LAW.items() if v == 'general'}),
    'individual_eligibility': code.isin({k for k, v in LAW.items() if v == 'general'}) |
                              (code.isin({k for k, v in LAW.items() if v == 'telework'}) & home) |
                              (code.isin({k for k, v in LAW.items() if v == 'ict'}) & ict),
    'as_submitted': code.isin({'FR', 'BE', 'IT', 'ES', 'PT', 'GR', 'SK', 'LU', 'IE'}),
}
def interact(mask, flag, ycol, x):
    d = E[mask].copy(); d['R'] = flag[mask].astype(float).values
    d = d[[ycol, x, 'R', 'country'] + cov].dropna(subset=[ycol, x])
    for c in cov: d[c] = d[c].fillna(d[c].median())
    d['XR'] = d[x] * d['R']
    X = sm.add_constant(d[[x, 'R', 'XR'] + cov])
    fam = sm.families.Gaussian() if ycol == 'wellbeing' else sm.families.Binomial()
    m = sm.GLM(d[ycol], X, family=fam).fit(cov_type='cluster', cov_kwds={'groups': d['country']})
    f = (lambda v: v) if ycol == 'wellbeing' else np.exp
    ci = m.conf_int().loc['XR']
    return {'est': round(float(f(m.params['XR'])), 3), 'lo': round(float(f(ci[0])), 3), 'hi': round(float(f(ci[1])), 3),
            'p': round(float(m.pvalues['XR']), 3), 'n': int(len(d)), 'n_countries': int(d.country.nunique()),
            'covered_countries': int(d.loc[d.R == 1, 'country'].nunique()), 'covered_pct': round(100 * d.R.mean(), 1)}, d
def wild(d, ycol, x, B=1999):
    X = sm.add_constant(d[[x, 'R', 'XR'] + cov]).values; Y = d[ycol].values.astype(float); G = d['country'].values
    ug = np.unique(G); gi = [np.where(G == g)[0] for g in ug]; j = 3
    def t(Yv):
        m = sm.OLS(Yv, X).fit(cov_type='cluster', cov_kwds={'groups': G}); return m.params[j] / m.bse[j]
    t0 = t(Y); Xr = np.delete(X, j, axis=1); fr = Xr @ np.linalg.lstsq(Xr, Y, rcond=None)[0]; ur = Y - fr
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)]); r = np.random.RandomState(SEED); tb = np.empty(B)
    for b in range(B):
        v = r.choice(webb, len(ug)); Ys = fr.copy()
        for k, ix in enumerate(gi): Ys[ix] += v[k] * ur[ix]
        tb[b] = t(Ys)
    return round(float(np.mean(np.abs(tb) >= abs(t0))), 3)
EU = ~code.isin(NONEU); ALL = pd.Series(True, index=E.index)
for popname, pop in [('EU27', EU), ('all35', ALL)]:
    R[popname] = {}
    for sname, flag in SPEC.items():
        R[popname][sname] = {}
        for x in ['DTI', 'CONTACT']:
            for ycol in ['sleep_bad', 'wellbeing']:
                res, d = interact(pop, flag, ycol, x)
                if sname == 'primary_statutory9': res['p_wild'] = wild(d, ycol, x)
                R[popname][sname]['%s|%s' % (x, ycol)] = res
    lo = [interact(pop & (code != cc), SPEC['primary_statutory9'], 'sleep_bad', 'DTI')[0]['est'] for cc in LAW]
    lo2 = [interact(pop & (code != cc), SPEC['primary_statutory9'], 'sleep_bad', 'CONTACT')[0]['est'] for cc in LAW]
    R[popname]['leave_one_covered_out'] = {'DTI': [min(lo), max(lo)], 'CONTACT': [min(lo2), max(lo2)]}
    dd = E[pop][['contact_out_wh', 'country'] + cov].copy(); dd['R'] = SPEC['primary_statutory9'][pop].astype(float).values
    dd['weekly'] = (dd['contact_out_wh'] <= 2).astype(float)
    for c in cov: dd[c] = dd[c].fillna(dd[c].median())
    m = sm.GLM(dd['weekly'], sm.add_constant(dd[['R'] + cov]), family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': dd['country']})
    R[popname]['exposure_level'] = {'weekly_pct_covered': round(100 * dd.loc[dd.R == 1, 'weekly'].mean(), 1),
                                    'weekly_pct_uncovered': round(100 * dd.loc[dd.R == 0, 'weekly'].mean(), 1),
                                    'OR': round(float(np.exp(m.params['R'])), 3), 'lo': round(float(np.exp(m.conf_int().loc['R', 0])), 3),
                                    'hi': round(float(np.exp(m.conf_int().loc['R', 1])), 3)}
    print(popname, json.dumps(R[popname]['primary_statutory9']), R[popname]['leave_one_covered_out'], R[popname]['exposure_level'])

# (3) harmonised seven-item model as trained on 2024 for the 2015 validation (logistic, C=1, standardised; back-transformed)
H7 = ['freetime_work', 'norest_b', 'night', 'longday', 'loc_home', 'computer', 'usual_hours_week']
S = E[E.sleep_bad.notna()]
X = S[H7].values.astype(float); y = S['sleep_bad'].values.astype(int)
med = np.nanmedian(X, axis=0); Xi = np.where(np.isnan(X), med, X)
sc = StandardScaler().fit(Xi); m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xi), y)
bs = m.coef_[0]; braw = bs / sc.scale_; b0 = m.intercept_[0] - np.sum(bs * sc.mean_ / sc.scale_)
R['H7_equation'] = {'intercept': round(float(b0), 4), 'coef_raw': {f: round(float(b), 5) for f, b in zip(H7, braw)},
                    'imputation_median_2024': {f: float(v) for f, v in zip(H7, med)}}
print(R['H7_equation'])
json.dump(R, open(os.path.join(OUT, 'policy_h7.json'), 'w', encoding='utf-8'), indent=2)
print('DONE')
