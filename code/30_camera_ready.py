# -*- coding: utf-8 -*-
"""ICECE 2026 camera-ready analyses (Paper 640) - answers to Reviewers 1-3.

Fixes found while preparing the response:
  * q49 (contact) and q101 (quit intention) are asked of EMPLOYEES only (routing q6==1|q7==1);
    the submitted analysis coded the 6,305 non-asked self-employed as 'never contacted'.
    -> primary analytic population = employees (also the only workers a right-to-disconnect
       law can entitle).
  * missing outcomes were coded as negatives (x<=2 on NaN -> False). -> excluded per outcome.
  * right-to-disconnect coding: Ireland (code of practice, not statute) was coded covered and
    Croatia (statute in force 1 Jan 2023) uncovered. -> recoded from Eurofound (2023) Table 1.

Outputs: results/camera_ready/metrics_cr.json, per_country.csv, fold_assignment.csv,
         subgroup.csv, figs (cr_*.png)
"""
import os, json, sys, time, warnings, platform
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sklearn, statsmodels, scipy
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc, roc_curve
import statsmodels.api as sm

ROOT = r'F:\psychology\journal3'
OUT = os.path.join(ROOT, 'results', 'camera_ready'); os.makedirs(OUT, exist_ok=True)
P24 = r'F:\psychology\EWCS 2024-1\European Work Condition Survey\9511tab\UKDA-9511-tab\tab\ewcs24_dataset_ukds.tab'
P15 = r'F:\psychology\EWCS 2015\UKDA-8098-spss\spss\spss25\ewcs6_2015_ukda_1904.sav'
SEED = 0; rng = np.random.RandomState(SEED)
R = {'env': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
             'statsmodels': statsmodels.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
             'scipy': scipy.__version__, 'seed': SEED}}
T0 = time.time()
def log(*a): print('[%5.0fs]' % (time.time() - T0), *a, flush=True)
def save(): json.dump(R, open(os.path.join(OUT, 'metrics_cr.json'), 'w', encoding='utf-8'), indent=2, default=float)
def num(s): return pd.to_numeric(s.astype(str).str.strip().replace({'': np.nan}), errors='coerce')
def zs(s): s = pd.to_numeric(s, errors='coerce'); return (s - s.mean()) / s.std()

# ============================================================ data
use = ['contact_out_wh', 'freetime_work', 'wlb_worry', 'wlb_tired', 'norest', 'night', 'longday', 'loc_home',
       'computer', 'usual_hours_week', 'sleeping_difficulties', 'wellbeing', 'quit_job', 'health_anxiety',
       'country', 'country_code', 'age', 'sex_id', 'isced', 'calweight', 'stratumid']
raw = pd.read_csv(P24, sep='\t', usecols=use, encoding='latin-1', low_memory=False)
df = raw.copy()
for c in use:
    if c != 'country_code': df[c] = num(df[c])
df['night'] = df['night'].replace(-889, 0); df['longday'] = df['longday'].replace(-889, 0)  # -889 = does not work nights/long days
for c in use:
    if c not in ('night', 'longday', 'country_code'): df.loc[df[c] < 0, c] = np.nan
df['norest_b'] = np.where(df['norest'].isna(), np.nan, (df['norest'] == 1).astype(float))
def bin_out(x, cond):
    o = pd.Series(np.nan, index=x.index); o[x.notna()] = cond(x[x.notna()]).astype(float); return o
df['sleep_bad'] = bin_out(df['sleeping_difficulties'], lambda x: x <= 2)
df['screenpos'] = bin_out(df['wellbeing'], lambda x: x <= 50)
df['anxiety'] = bin_out(df['health_anxiety'], lambda x: x == 1)
df['turnover'] = bin_out(df['quit_job'], lambda x: x <= 2)
df['employee'] = df['contact_out_wh'].notna()          # q49 routed to employees only
W = df['calweight'].clip(lower=0)

FEAT10 = ['contact_out_wh', 'freetime_work', 'wlb_worry', 'wlb_tired', 'norest_b', 'night', 'longday',
          'loc_home', 'computer', 'usual_hours_week']
B8 = [f for f in FEAT10 if f not in ('wlb_worry', 'wlb_tired')]
NAMES = {'contact_out_wh': 'Off-hours work contact', 'freetime_work': 'Working in free time',
         'wlb_worry': 'Off-work rumination', 'wlb_tired': 'Post-work fatigue', 'norest_b': '<11 h rest between days',
         'night': 'Night work (times/month)', 'longday': 'Long days >10 h (times/month)',
         'loc_home': 'Works from own home', 'computer': 'Digital-device use', 'usual_hours_week': 'Usual weekly hours'}

E = df[df.employee].reset_index(drop=True)
OUTS = ['sleep_bad', 'anxiety', 'screenpos', 'turnover']
def wprev(d, y):
    m = d[y].notna(); return float(np.average(d.loc[m, y], weights=W.loc[d.index][m] if False else d.loc[m, 'calweight'].clip(lower=0)))
R['sample'] = {'n_total': int(len(df)), 'n_employees': int(len(E)), 'n_selfemployed_not_asked_q49': int((~df.employee).sum()),
               'n_countries': int(E.country.nunique()), 'n_strata': int(raw.stratumid.astype(str).nunique()),
               'outcomes': {y: {'n_analytic': int(E[y].notna().sum()), 'n_missing_excluded': int(E[y].isna().sum()),
                                'events': int(E[y].sum()), 'prev_unweighted': round(100 * E[y].mean(), 1),
                                'prev_weighted': round(100 * wprev(E, y), 1)} for y in OUTS},
               'outcomes_allworkers': {y: {'n_analytic': int(df[y].notna().sum()), 'prev_unweighted': round(100 * df[y].mean(), 1),
                                           'prev_weighted': round(100 * wprev(df, y), 1)} for y in OUTS},
               'item_missing_pct_employees': {f: round(100 * float(E[f].isna().mean()), 2) for f in FEAT10}}
log('sample', R['sample']); save()

# ============================================================ DTI + association
cov = ['age', 'sex_id', 'isced', 'usual_hours_week']
def dti(d, items):
    Z = pd.DataFrame({c: zs(6 - d[c]) for c in items})   # orient: higher = more tethered; z-score each item
    s = Z.mean(axis=1, skipna=True)                      # mean of available components
    return zs(s), Z.notna().sum(axis=1)
E['DTI'], ncomp = dti(E, ['contact_out_wh', 'freetime_work', 'wlb_worry'])
E['DTI_beh'], _ = dti(E, ['contact_out_wh', 'freetime_work'])
R['dti'] = {'formula': 'DTI = z( mean_k z(6 - x_k) ), k in {q49, q50, q48a}; mean over available components',
            'n_components_dist': {int(k): int(v) for k, v in ncomp.value_counts().sort_index().items()}}
def assoc(d, y, x, weighted=False, extra=None):
    cols = [y, x, 'country'] + cov + (extra or [])
    dd = d[cols + ['calweight']].dropna(subset=[y, x, 'country']).copy()
    for c in cov: dd[c] = dd[c].fillna(dd[c].median())
    X = sm.add_constant(dd[[x] + cov + (extra or [])])
    if y == 'wellbeing':
        fam = sm.families.Gaussian()
    else:
        fam = sm.families.Binomial()
    vw = dd['calweight'].clip(lower=1e-6) / dd['calweight'].clip(lower=1e-6).mean() if weighted else None
    m = sm.GLM(dd[y], X, family=fam, var_weights=vw).fit(cov_type='cluster', cov_kwds={'groups': dd['country']})
    b, lo, hi = m.params[x], m.conf_int().loc[x, 0], m.conf_int().loc[x, 1]
    f = (lambda v: v) if y == 'wellbeing' else np.exp
    return {'est': round(float(f(b)), 3), 'lo': round(float(f(lo)), 3), 'hi': round(float(f(hi)), 3),
            'p': float(m.pvalues[x]), 'n': int(len(dd))}
A = {}
for y in ['sleep_bad', 'anxiety', 'turnover', 'wellbeing']:
    A[y] = {'DTI': assoc(E, y, 'DTI'), 'DTI_weighted': assoc(E, y, 'DTI', True), 'DTI_behavioural': assoc(E, y, 'DTI_beh')}
R['association'] = A; log('assoc', {k: v['DTI'] for k, v in A.items()}); save()

# category ORs + nonlinearity tests (employees, contact categories)
dd = E[['sleep_bad', 'contact_out_wh', 'country'] + cov].dropna(subset=['sleep_bad']).copy()
for c in cov: dd[c] = dd[c].fillna(dd[c].median())
dd['score'] = 5 - dd['contact_out_wh']                    # 0 never .. 4 daily
D = pd.get_dummies(dd['contact_out_wh'].astype(int), prefix='c').astype(float).drop(columns=['c_5'])
Xcat = sm.add_constant(pd.concat([D, dd[cov]], axis=1)); Xlin = sm.add_constant(dd[['score'] + cov])
mcat = sm.Logit(dd['sleep_bad'], Xcat).fit(disp=0, cov_type='cluster', cov_kwds={'groups': dd['country']})
mlin = sm.Logit(dd['sleep_bad'], Xlin).fit(disp=0)
mcat0 = sm.Logit(dd['sleep_bad'], Xcat).fit(disp=0)
lr = 2 * (mcat0.llf - mlin.llf)
# cluster-robust joint Wald test of departure from linearity: score + 3 deviation dummies
Xdev = sm.add_constant(pd.concat([dd[['score']], D[['c_2', 'c_3', 'c_4']], dd[cov]], axis=1))
mdev = sm.Logit(dd['sleep_bad'], Xdev).fit(disp=0, cov_type='cluster', cov_kwds={'groups': dd['country']})
wt = mdev.wald_test('c_2 = 0, c_3 = 0, c_4 = 0', scalar=True)
# + country fixed effects
CF = pd.get_dummies(dd['country'].astype(int), prefix='k', drop_first=True).astype(float)
mdevfe = sm.Logit(dd['sleep_bad'], pd.concat([Xdev, CF], axis=1)).fit(disp=0, maxiter=200, cov_type='cluster',
                                                                       cov_kwds={'groups': dd['country']})
wtfe = mdevfe.wald_test('c_2 = 0, c_3 = 0, c_4 = 0', scalar=True)
lab = {1: 'Daily', 2: 'Several times a week', 3: 'Several times last month', 4: 'Less often', 5: 'Never'}
cat = {}
for k in [5, 4, 3, 2, 1]:
    s = dd[dd.contact_out_wh == k]
    row = {'n': int(len(s)), 'events': int(s.sleep_bad.sum()), 'prev': round(100 * s.sleep_bad.mean(), 1)}
    if k != 5:
        ci = mcat.conf_int().loc['c_%d' % k]
        row.update({'OR': round(float(np.exp(mcat.params['c_%d' % k])), 2), 'lo': round(float(np.exp(ci[0])), 2),
                    'hi': round(float(np.exp(ci[1])), 2), 'p': float(mcat.pvalues['c_%d' % k])})
    cat[lab[k]] = row
R['dose'] = {'categories': cat, 'lr_naive': {'chi2': round(float(lr), 1), 'df': 3, 'p': float(stats.chi2.sf(lr, 3))},
             'wald_cluster': {'chi2': round(float(wt.statistic), 1), 'df': 3, 'p': float(wt.pvalue)},
             'wald_cluster_countryFE': {'chi2': round(float(wtfe.statistic), 1), 'df': 3, 'p': float(wtfe.pvalue)}}
log('dose', R['dose']); save()

# ============================================================ screening machinery
def make(kind):
    if kind == 'lr': return LogisticRegression(C=1.0, max_iter=2000)
    return GBC(max_iter=300, learning_rate=0.05, random_state=SEED)
def fit_pred(kind, Xtr, ytr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    if kind == 'lr':
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = make(kind).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m.predict_proba(Xtr)[:, 1]
def inner_oof(kind, X, y, k=3):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=SEED + 1).split(X, y):
        p[te] = fit_pred(kind, X[tr], y[tr], X[te])[0]
    return p
def thr_for_sens(p, y, s=0.80): return float(np.quantile(p[y == 1], 1 - s))
CAP = [0.10, 0.20, 0.30]
def run_cv(kind, X, y, nested=True):
    """Stratified 5-fold OOF predictions + thresholds chosen INSIDE training folds (inner 3-fold OOF)."""
    p = np.zeros(len(y)); thr = {'sens80': np.zeros(len(y))}; thr.update({c: np.zeros(len(y)) for c in CAP})
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        p[te] = fit_pred(kind, X[tr], y[tr], X[te])[0]
        if nested:
            pi = inner_oof(kind, X[tr], y[tr])
            thr['sens80'][te] = thr_for_sens(pi, y[tr])
            for c in CAP: thr[c][te] = float(np.quantile(pi, 1 - c))
    return p, thr
def boot_auc(y, p, n=1000, groups=None):
    a = []; idx = np.arange(len(y))
    if groups is None:
        for _ in range(n):
            b = rng.choice(idx, len(idx), True); a.append(roc_auc_score(y[b], p[b]))
    else:
        ug = np.unique(groups); pos = {g: np.where(groups == g)[0] for g in ug}
        for _ in range(n):
            b = np.concatenate([pos[g] for g in rng.choice(ug, len(ug), True)]); a.append(roc_auc_score(y[b], p[b]))
    return [round(float(v), 3) for v in np.percentile(a, [2.5, 97.5])]
def calib(y, p, w=None):
    pc = np.clip(p, 1e-6, 1 - 1e-6); lo = np.log(pc / (1 - pc))
    m = sm.GLM(y, sm.add_constant(lo), family=sm.families.Binomial(), var_weights=w).fit()
    off = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lo, var_weights=w).fit()
    bins = np.quantile(p, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    ece = sum(np.mean(idx == b) * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(10) if (idx == b).any())
    return {'slope': round(float(m.params[1]), 3), 'citl': round(float(off.params[0]), 3), 'ece': round(float(ece), 4),
            'OE': round(float(y.mean() / p.mean()), 3)}
def opstats(y, p, t):
    f = p >= t; tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum()); fn = int((~f & (y == 1)).sum()); tn = int((~f & (y == 0)).sum())
    prev = y.mean(); ppv = tp / max(tp + fp, 1)
    return {'flag_pct': round(100 * f.mean(), 1), 'sens': round(tp / max(tp + fn, 1), 3), 'spec': round(tn / max(tn + fp, 1), 3),
            'ppv': round(ppv, 3), 'npv': round(tn / max(tn + fn, 1), 3), 'lift': round(ppv / prev, 2),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
def nb(y, p, pt): f = p >= pt; n = len(y); return ((f & (y == 1)).sum() - (f & (y == 0)).sum() * pt / (1 - pt)) / n

ys = E['sleep_bad'].values; okS = ~np.isnan(ys)
S = E[okS].reset_index(drop=True); y = S['sleep_bad'].values.astype(int); grp = S['country'].values
w = S['calweight'].clip(lower=1e-6).values; w = w / w.mean()
MODELS = {'M10_gbm': ('gbm', FEAT10), 'M10_lr': ('lr', FEAT10), 'B8_gbm': ('gbm', B8), 'B8_lr': ('lr', B8)}
P = {}; TH = {}; SC = {}
for nm, (kind, feats) in MODELS.items():
    X = S[feats].values.astype(float)
    P[nm], TH[nm] = run_cv(kind, X, y)
    p = P[nm]; pp, rr, _ = precision_recall_curve(y, p)
    SC[nm] = {'auroc': round(roc_auc_score(y, p), 3), 'ci_worker': boot_auc(y, p), 'ci_country_cluster': boot_auc(y, p, 1000, grp),
              'auroc_weighted': round(roc_auc_score(y, p, sample_weight=w), 3), 'brier': round(brier_score_loss(y, p), 3),
              'brier_weighted': round(brier_score_loss(y, p, sample_weight=w), 3), 'prauc': round(auc(rr, pp), 3),
              'calibration': calib(y, p), 'calibration_weighted': calib(y, p, w),
              'op_sens80_nested': opstats(y, p, TH[nm]['sens80']),
              'op_capacity_nested': {str(int(c * 100)): opstats(y, p, TH[nm][c]) for c in CAP}}
    log(nm, SC[nm]['auroc'], SC[nm]['ci_country_cluster'], SC[nm]['calibration'], SC[nm]['op_sens80_nested']);
R['screening'] = SC; R['analytic_n_sleep'] = int(len(y)); R['prev_sleep'] = round(100 * y.mean(), 2); save()
# paired AUROC differences
def dboot(a, b, n=2000):
    d = []; idx = np.arange(len(y))
    for _ in range(n):
        bb = rng.choice(idx, len(idx), True); d.append(roc_auc_score(y[bb], P[a][bb]) - roc_auc_score(y[bb], P[b][bb]))
    return {'delta': round(roc_auc_score(y, P[a]) - roc_auc_score(y, P[b]), 4), 'ci': [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])]}
R['deltas'] = {'B8_gbm_minus_lr': dboot('B8_gbm', 'B8_lr'), 'M10_gbm_minus_lr': dboot('M10_gbm', 'M10_lr'),
               'M10_minus_B8_gbm': dboot('M10_gbm', 'B8_gbm'), 'M10_minus_B8_lr': dboot('M10_lr', 'B8_lr')}
# ladder (GBM and LR)
lad = {}
for kind in ['gbm', 'lr']:
    for rung, feats in [('full10', FEAT10), ('drop_fatigue', [f for f in FEAT10 if f != 'wlb_tired']), ('B8', B8), ('contact_only', ['contact_out_wh'])]:
        p, _ = run_cv(kind, S[feats].values.astype(float), y, nested=False)
        lad['%s_%s' % (kind, rung)] = round(roc_auc_score(y, p), 3)
R['ladder'] = lad; log('ladder', lad); save()

# decision curves
pts = np.round(np.arange(0.05, 0.501, 0.01), 2); prev = y.mean()
DC = {'pt': pts.tolist(), 'treat_all': [round(prev - (1 - prev) * t / (1 - t), 4) for t in pts]}
for nm in ['M10_gbm', 'B8_lr', 'B8_gbm', 'M10_lr']:
    DC[nm] = [round(float(nb(y, P[nm], t)), 4) for t in pts]
def superior(nm):
    good = [t for t, a, b in zip(pts, DC[nm], DC['treat_all']) if a > max(b, 0) + 1e-4]
    return [float(min(good)), float(max(good))] if good else None
R['dca'] = {'curves': DC, 'superior_range': {nm: superior(nm) for nm in ['M10_gbm', 'B8_lr', 'B8_gbm', 'M10_lr']}}
log('dca', R['dca']['superior_range']); save()

# ============================================================ geography
def grouped_oof(kind, feats, splitter, groups):
    X = S[feats].values.astype(float); p = np.zeros(len(y)); t80 = np.zeros(len(y))
    for tr, te in splitter.split(X, y, groups):
        p[te], ptr = fit_pred(kind, X[tr], y[tr], X[te]); t80[te] = thr_for_sens(ptr, y[tr])
    return p, t80
GEO = {}; PERC = []
for nm in ['B8_lr', 'M10_lr', 'B8_gbm', 'M10_gbm']:
    kind, feats = MODELS[nm]
    pg, _ = grouped_oof(kind, feats, GroupKFold(10), grp)
    pl, t80 = grouped_oof(kind, feats, LeaveOneGroupOut(), grp)
    rows = []
    for c in np.unique(grp):
        m = grp == c; yc, pc = y[m], pl[m]
        a = roc_auc_score(yc, pc); q = a / (2 - a); q2 = 2 * a * a / (1 + a); n1 = yc.sum(); n0 = len(yc) - n1
        se = np.sqrt((a * (1 - a) + (n1 - 1) * (q - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))   # Hanley-McNeil
        bs = []
        ii = np.arange(len(yc))
        for _ in range(500):
            b = rng.choice(ii, len(ii), True)
            if 0 < yc[b].sum() < len(b): bs.append(roc_auc_score(yc[b], pc[b]))
        o = opstats(yc, pc, t80[m][0])
        rows.append({'model': nm, 'country': int(c), 'code': S.loc[m, 'country_code'].iloc[0], 'n': int(m.sum()), 'events': int(n1),
                     'prev': round(100 * yc.mean(), 1), 'auroc': round(a, 3), 'lo': round(float(np.percentile(bs, 2.5)), 3),
                     'hi': round(float(np.percentile(bs, 97.5)), 3), 'se_hm': float(se), 'OE': round(float(yc.mean() / pc.mean()), 3),
                     'sens': o['sens'], 'spec': o['spec'], 'flag_pct': o['flag_pct']})
    PERC += rows
    # random-effects meta-analysis on logit(AUROC)
    a = np.array([r['auroc'] for r in rows]); se = np.array([r['se_hm'] for r in rows])
    th = np.log(a / (1 - a)); v = (se / (a * (1 - a))) ** 2; wi = 1 / v
    fe = np.sum(wi * th) / wi.sum(); Q = np.sum(wi * (th - fe) ** 2); k = len(th)
    tau2 = max(0, (Q - (k - 1)) / (wi.sum() - (wi ** 2).sum() / wi.sum())); ws = 1 / (v + tau2)
    mu = np.sum(ws * th) / ws.sum(); semu = np.sqrt(1 / ws.sum()); tcrit = stats.t.ppf(0.975, k - 2)
    il = lambda x: 1 / (1 + np.exp(-x))
    GEO[nm] = {'pooled_grouped10': round(roc_auc_score(y, pg), 3), 'pooled_loco': round(roc_auc_score(y, pl), 3),
               'loco_ci_country_cluster': boot_auc(y, pl, 1000, grp),
               'mean_within_country': round(float(a.mean()), 3), 'min_within': round(float(a.min()), 3), 'max_within': round(float(a.max()), 3),
               're_pooled': round(float(il(mu)), 3), 're_ci': [round(float(il(mu - 1.96 * semu)), 3), round(float(il(mu + 1.96 * semu)), 3)],
               're_pi': [round(float(il(mu - tcrit * np.sqrt(tau2 + semu ** 2))), 3), round(float(il(mu + tcrit * np.sqrt(tau2 + semu ** 2))), 3)],
               'tau2': round(float(tau2), 4), 'I2': round(float(max(0, (Q - (k - 1)) / Q) * 100), 1), 'k': k,
               'n_countries_below_0.55': int((a < 0.55).sum()),
               'loco_calibration': calib(y, pl), 'loco_OE_range': [float(min(r['OE'] for r in rows)), float(max(r['OE'] for r in rows))]}
    log('geo', nm, GEO[nm])
R['geography'] = GEO; pd.DataFrame(PERC).to_csv(os.path.join(OUT, 'per_country.csv'), index=False); save()
# fold assignment for GroupKFold(10)
fa = []
for f, (tr, te) in enumerate(GroupKFold(10).split(S, y, grp)):
    for c in np.unique(grp[te]): fa.append({'fold': f + 1, 'country': int(c), 'code': S.loc[grp == c, 'country_code'].iloc[0]})
pd.DataFrame(fa).sort_values(['fold', 'code']).to_csv(os.path.join(OUT, 'fold_assignment.csv'), index=False)

# region transfer: train EU-27 -> test 8 non-EU (EFTA + candidate countries)
NONEU = {'NO', 'CH', 'AL', 'BA', 'ME', 'MK', 'RS', 'XK'}
isnon = S['country_code'].isin(NONEU).values; WB = S['country_code'].isin(NONEU - {'NO', 'CH'}).values
RT = {}
for nm in ['B8_lr', 'M10_lr', 'B8_gbm', 'M10_gbm']:
    kind, feats = MODELS[nm]; X = S[feats].values.astype(float)
    pte, ptr = fit_pred(kind, X[~isnon], y[~isnon], X[isnon]); t = thr_for_sens(ptr, y[~isnon])
    sub = WB[isnon]
    RT[nm] = {'n_test': int(isnon.sum()), 'auroc_nonEU8': round(roc_auc_score(y[isnon], pte), 3), 'ci': boot_auc(y[isnon], pte, 1000, grp[isnon]),
              'auroc_westernBalkans6': round(roc_auc_score(y[isnon][sub], pte[sub]), 3), 'calibration': calib(y[isnon], pte),
              'op_sens80_train_threshold': opstats(y[isnon], pte, t)}
R['region_transfer'] = RT; log('region', RT); save()

# ============================================================ subgroups at the operating threshold
age = S['age'].values; sex = S['sex_id'].values; ed = S['isced'].values
GR = {'Women': sex == 2, 'Men': sex == 1, 'Age <35': age < 35, 'Age 35-49': (age >= 35) & (age < 50), 'Age 50+': age >= 50,
      'Education high (ISCED 5-8)': ed >= 5, 'Education low/medium (ISCED 0-4)': ed < 5}
SG = []
for nm in ['B8_lr', 'M10_lr', 'B8_gbm', 'M10_gbm']:
    for g, m in GR.items():
        o = opstats(y[m], P[nm][m], TH[nm]['sens80'][m]); c = calib(y[m], P[nm][m])
        SG.append({'model': nm, 'group': g, 'n': int(m.sum()), 'prev': round(100 * y[m].mean(), 1), 'auroc': round(roc_auc_score(y[m], P[nm][m]), 3),
                   'ci': boot_auc(y[m], P[nm][m], 500), 'sens': o['sens'], 'spec': o['spec'], 'ppv': o['ppv'], 'referral_pct': o['flag_pct'],
                   'cal_slope': c['slope'], 'citl': c['citl'], 'OE': c['OE']})
pd.DataFrame(SG).to_csv(os.path.join(OUT, 'subgroup.csv'), index=False); R['subgroups'] = SG; save()
log('subgroups done')

# ============================================================ final transparent model (equation) + attribution
X8 = S[B8].values.astype(float); med = np.nanmedian(X8, axis=0); X8i = np.where(np.isnan(X8), med, X8)
sc = StandardScaler().fit(X8i); lrm = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X8i), y)
beta_std = lrm.coef_[0]; beta_raw = beta_std / sc.scale_; b0 = lrm.intercept_[0] - np.sum(beta_std * sc.mean_ / sc.scale_)
R['final_equation_B8_lr'] = {'intercept': round(float(b0), 4),
                             'coef_raw': {f: round(float(b), 5) for f, b in zip(B8, beta_raw)},
                             'OR_per_SD': {f: round(float(np.exp(b)), 3) for f, b in zip(B8, beta_std)},
                             'imputation_median': {f: float(m) for f, m in zip(B8, med)},
                             'coding': 'frequency items 1=most frequent..5=never (computer 1..7); norest_b 1=yes; night/longday times per month (0 if never); hours numeric'}
try:
    import shap
    for nm, feats in [('M10_gbm', FEAT10), ('B8_gbm', B8)]:
        X = S[feats].values.astype(float); Xi = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        g = GBC(max_iter=300, learning_rate=0.05, random_state=SEED).fit(Xi, y)
        ix = rng.choice(len(Xi), 5000, False); sv = shap.TreeExplainer(g).shap_values(Xi[ix]); imp = np.abs(sv).mean(0)
        R['shap_' + nm] = {f: {'mean_abs_logodds': round(float(v), 3), 'share_pct': round(float(100 * v / imp.sum()), 1)}
                           for f, v in sorted(zip(feats, imp), key=lambda t: -t[1])}
    R['shap_' + 'version'] = shap.__version__
except Exception as e:
    R['shap_error'] = repr(e)
save(); log('equation+shap done')

# ============================================================ right-to-disconnect (Eurofound 2023, Table 1)
LAW = {  # code: (in force, scope, statute?)
    'BE': ('2023-01-01', 'all workers, private firms >20 employees (federal civil servants from 2022-02-01)', 'general'),
    'HR': ('2023-01-01', 'employees (Labour Act amendment passed 2022)', 'general'),
    'FR': ('2017-01-01', 'employees; negotiation duty in firms >50 (charter >11)', 'general'),
    'GR': ('2023-01-01', 'teleworkers (private sector); public-sector teleworkers', 'telework'),
    'IT': ('2017-06-06', 'smart-working contract holders only', 'telework'),
    'LU': ('2023-07-04', 'all workers who use ICT for work', 'general'),
    'PT': ('2022-01-01', 'all workers who use ICT for work', 'general'),
    'SK': ('2021-03-01', 'teleworkers incl. occasional', 'telework'),
    'ES': ('2018-12-07', 'all workers (in practice remote ICT users)', 'general')}
R['r2d_coding'] = {'source': 'Eurofound (2023) Right to disconnect: implementation and impact at company level, Table 1 (pp. 8-9)',
                   'fieldwork': 'EWCS 2024 main fieldwork Feb-Nov 2024; all provisions in force before fieldwork',
                   'laws': LAW, 'not_statutory': {'IE': 'Code of Practice from 2021-04-01 (no formal status in law)'},
                   'non_EU': 'NO, CH, AL, BA, ME, MK, RS, XK: no statutory provision coded (outside Eurofound EU scope)'}
code = E['country_code']
E['home_any'] = (E['loc_home'] <= 3).astype(float)          # works from home at least 'sometimes'
E['ict'] = (E['computer'] <= 5).astype(float)                # device use at least ~1/4 of the time
SPEC = {
    'primary_statutory9': code.isin(LAW.keys()),
    'as_submitted(IE in, HR out)': code.isin({'FR', 'BE', 'IT', 'ES', 'PT', 'GR', 'SK', 'LU', 'IE'}),
    'general_scope_only6': code.isin({k for k, v in LAW.items() if v[2] == 'general'}),
    'statutory9_plus_IE': code.isin(set(LAW) | {'IE'}),
    'individual_eligibility': (code.isin({'BE', 'HR', 'FR'})) | (code.isin({'GR', 'IT', 'SK'}) & (E['home_any'] == 1)) |
                              (code.isin({'LU', 'PT', 'ES'}) & (E['ict'] == 1)),
}
def interact(d, cov_flag, ycol, extra_mask=None):
    dd = d.copy(); dd['R'] = cov_flag.astype(float).values if hasattr(cov_flag, 'values') else cov_flag
    if extra_mask is not None: dd = dd[extra_mask]
    dd = dd[[ycol, 'DTI', 'R', 'country'] + cov].dropna(subset=[ycol, 'DTI'])
    for c in cov: dd[c] = dd[c].fillna(dd[c].median())
    dd['DxR'] = dd['DTI'] * dd['R']
    X = sm.add_constant(dd[['DTI', 'R', 'DxR'] + cov])
    fam = sm.families.Gaussian() if ycol == 'wellbeing' else sm.families.Binomial()
    m = sm.GLM(dd[ycol], X, family=fam).fit(cov_type='cluster', cov_kwds={'groups': dd['country']})
    f = (lambda v: v) if ycol == 'wellbeing' else np.exp
    ci = m.conf_int().loc['DxR']
    return {'est': round(float(f(m.params['DxR'])), 3), 'lo': round(float(f(ci[0])), 3), 'hi': round(float(f(ci[1])), 3),
            'p': round(float(m.pvalues['DxR']), 3), 'n': int(len(dd)), 'covered_countries': int(dd.loc[dd.R == 1, 'country'].nunique()),
            'covered_pct': round(100 * dd.R.mean(), 1)}, dd
def wild_cluster(dd, ycol, B=1999):
    """WCR wild-cluster bootstrap-t (Webb 6-point) for DxR in a linear model (LPM for binary)."""
    X = sm.add_constant(dd[['DTI', 'R', 'DxR'] + cov]).values; Y = dd[ycol].values.astype(float); G = dd['country'].values
    ug = np.unique(G); gi = [np.where(G == g)[0] for g in ug]; j = 3
    def tstat(Yv):
        m = sm.OLS(Yv, X).fit(cov_type='cluster', cov_kwds={'groups': G}); return m.params[j] / m.bse[j]
    t0 = tstat(Y)
    Xr = np.delete(X, j, axis=1); br = np.linalg.lstsq(Xr, Y, rcond=None)[0]; fr = Xr @ br; ur = Y - fr
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    r = np.random.RandomState(SEED); tb = np.empty(B)
    for b in range(B):
        v = r.choice(webb, len(ug)); Ys = fr.copy()
        for k, ix in enumerate(gi): Ys[ix] += v[k] * ur[ix]
        tb[b] = tstat(Ys)
    return {'t': round(float(t0), 3), 'p_wcb': round(float(np.mean(np.abs(tb) >= abs(t0))), 3), 'B': B}
RD = {}
for nmS, flag in SPEC.items():
    RD[nmS] = {}
    for yc in ['sleep_bad', 'wellbeing']:
        res, dd = interact(E, flag, yc); RD[nmS][yc] = res
        if nmS == 'primary_statutory9': RD[nmS][yc]['wild_cluster_LPM'] = wild_cluster(dd, yc)
RD['EU27_only_primary'] = {yc: interact(E, SPEC['primary_statutory9'], yc, extra_mask=~code.isin(NONEU))[0] for yc in ['sleep_bad', 'wellbeing']}
# leave-one-covered-country-out
lo_s = []
for cc in LAW:
    lo_s.append(interact(E, SPEC['primary_statutory9'], 'sleep_bad', extra_mask=(code != cc))[0]['est'])
RD['leave_one_covered_out_sleep_OR_range'] = [float(min(lo_s)), float(max(lo_s))]
# exposure-level ('first stage') comparison: is contact frequency lower where covered?
dd = E[['contact_out_wh', 'country'] + cov].copy(); dd['R'] = SPEC['primary_statutory9'].astype(float).values
dd['weekly'] = (dd['contact_out_wh'] <= 2).astype(float)
for c in cov: dd[c] = dd[c].fillna(dd[c].median())
m = sm.GLM(dd['weekly'], sm.add_constant(dd[['R'] + cov]), family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': dd['country']})
RD['exposure_level'] = {'weekly_contact_pct_covered': round(100 * dd.loc[dd.R == 1, 'weekly'].mean(), 1),
                        'weekly_contact_pct_uncovered': round(100 * dd.loc[dd.R == 0, 'weekly'].mean(), 1),
                        'adj_OR': round(float(np.exp(m.params['R'])), 3), 'lo': round(float(np.exp(m.conf_int().loc['R', 0])), 3),
                        'hi': round(float(np.exp(m.conf_int().loc['R', 1])), 3), 'p': round(float(m.pvalues['R']), 3)}
R['r2d'] = RD; log('r2d', json.dumps(RD)[:1500]); save()

# ============================================================ temporal external validation: EWCS 2015
try:
    import pyreadstat
    c15 = ['Country', 'Q7', 'Q24', 'Q30i', 'Q35e', 'Q37a', 'Q37d', 'Q38', 'Q45a', 'Q45b', 'Q46', 'Q79a', 'Q79b']
    d15, _ = pyreadstat.read_sav(P15, usecols=c15)
    d15 = d15[d15.Q7 == 1].copy()                                     # employees
    for c in ['Q30i', 'Q35e', 'Q38', 'Q45a', 'Q45b', 'Q46', 'Q79a', 'Q79b']: d15.loc[d15[c] >= 7, c] = np.nan
    for c in ['Q37a', 'Q37d']: d15.loc[d15[c] > 31, c] = np.nan
    d15.loc[(d15.Q24 > 100) | (d15.Q24 < 1), 'Q24'] = np.nan
    H = pd.DataFrame({'freetime_work': d15.Q46, 'norest_b': np.where(d15.Q38.isna(), np.nan, (d15.Q38 == 1).astype(float)),
                      'night': d15.Q37a, 'longday': d15.Q37d, 'loc_home': d15.Q35e, 'computer': d15.Q30i,
                      'usual_hours_week': d15.Q24, 'wlb_worry': d15.Q45a, 'wlb_tired': d15.Q45b})
    ok15 = d15.Q79a.notna() & d15.Q79b.notna()
    y15 = ((d15.Q79a <= 2) | (d15.Q79b <= 2)).astype(int).values[ok15.values]; H = H[ok15.values]; g15 = d15.Country.values[ok15.values]
    H7 = ['freetime_work', 'norest_b', 'night', 'longday', 'loc_home', 'computer', 'usual_hours_week']; H9 = H7 + ['wlb_worry', 'wlb_tired']
    TV = {'n_2015_employees': int(len(y15)), 'prev_2015': round(100 * y15.mean(), 1), 'countries_2015': int(len(np.unique(g15))),
          'outcome_2015': 'difficulty falling asleep OR repeated waking, daily/several times a week (last 12 months)'}
    for tag, feats in [('H7_behavioural', H7), ('H9_with_fatigue_rumination', H9)]:
        for kind in ['lr', 'gbm']:
            Xtr = S[feats].values.astype(float); Xte = H[feats].values.astype(float)
            pte, ptr = fit_pred(kind, Xtr, y, Xte); t = thr_for_sens(ptr, y)
            p24, _ = run_cv(kind, Xtr, y, nested=False)
            p15i, _ = run_cv(kind, Xte, y15, nested=False)
            TV['%s_%s' % (tag, kind)] = {'internal_2024_cv': round(roc_auc_score(y, p24), 3),
                                         'train2024_test2015': round(roc_auc_score(y15, pte), 3), 'ci_country_cluster': boot_auc(y15, pte, 1000, g15),
                                         'internal_2015_cv': round(roc_auc_score(y15, p15i), 3), 'calibration_2015': calib(y15, pte),
                                         'op_2024threshold': opstats(y15, pte, t)}
    R['temporal_2015'] = TV; log('2015', TV)
except Exception as e:
    R['temporal_2015_error'] = repr(e); log('2015 failed', repr(e))
save()

# ============================================================ sensitivity: submitted population (all workers, q49 not-asked -> never)
A2 = df[df.sleep_bad.notna()].copy(); A2['contact_out_wh'] = A2['contact_out_wh'].fillna(5); ya = A2['sleep_bad'].values.astype(int)
R['sens_all_workers'] = {'n': int(len(ya))}
for nm in ['M10_gbm', 'B8_gbm', 'B8_lr']:
    kind, feats = MODELS[nm]; p, _ = run_cv(kind, A2[feats].values.astype(float), ya, nested=False)
    R['sens_all_workers'][nm] = round(roc_auc_score(ya, p), 3)
log('sens_all', R['sens_all_workers']); save()

# ============================================================ figures
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 300})
# Fig A: decision curves
fig, ax = plt.subplots(figsize=(3.45, 2.35))
ax.plot(pts, DC['treat_all'], color='#7f7f7f', ls='--', lw=1, label='Refer all')
ax.axhline(0, color='k', lw=0.8, label='Refer none')
ax.plot(pts, DC['M10_lr'], color='#8c8c8c', lw=1.4, label='Reference model (10 items)')
ax.plot(pts, DC['B8_lr'], color='#1f5fa8', lw=1.8, label='Proposed screener (8 items)')
ax.set_ylim(-0.02, max(DC['M10_lr']) + 0.02); ax.set_xlim(0.05, 0.5)
ax.set_xlabel('Threshold probability (referral cost : benefit)'); ax.set_ylabel('Net benefit')
ax.legend(fontsize=6.3, frameon=False, loc='upper right'); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'cr_dca.png')); plt.close(fig)
# Fig B: per-country AUROC, both models, LOCO (wide, short)
pa = pd.DataFrame([r for r in PERC if r['model'] == 'B8_lr']).sort_values('auroc').reset_index(drop=True)
pm = pd.DataFrame([r for r in PERC if r['model'] == 'M10_lr']).set_index('code').loc[pa.code].reset_index()
fig, ax = plt.subplots(figsize=(7.1, 2.05)); xx = np.arange(len(pa))
for G_, dfp, colr, mk, lab_ in [('M10_lr', pm, '#9a9a9a', 's', 'Reference model (10 items)'), ('B8_lr', pa, '#1f5fa8', 'o', 'Proposed screener (8 items)')]:
    off = -0.17 if G_ == 'M10_lr' else 0.17
    ax.errorbar(xx + off, dfp.auroc, yerr=[dfp.auroc - dfp.lo, dfp.hi - dfp.auroc], fmt=mk, ms=3, color=colr, ecolor=colr, elinewidth=0.6, label=lab_)
    g = GEO[G_]; ax.axhspan(g['re_pi'][0], g['re_pi'][1], color=colr, alpha=0.10, lw=0)
ax.axhline(0.5, color='k', ls=':', lw=0.8)
ax.set_xticks(xx); ax.set_xticklabels(pa.code, fontsize=6.5)
for t in ax.get_xticklabels():
    if t.get_text() in NONEU: t.set_color('#c0392b'); t.set_fontweight('bold')
ax.set_ylabel('AUROC, held-out country', fontsize=7); ax.tick_params(axis='y', labelsize=6.5); ax.set_xlim(-0.7, len(pa) - 0.3)
ax.legend(fontsize=6.3, frameon=False, loc='upper left', ncol=2); fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, 'cr_forest.png')); plt.close(fig)
# Fig C: ladder with CIs (GBM and LR)
fig, ax = plt.subplots(figsize=(3.45, 2.1))
rungs = ['full10', 'drop_fatigue', 'B8', 'contact_only']; lbl = ['All 10\n(reference)', 'Drop\nfatigue', 'Behavioural 8\n(proposed)', 'Contact\nitem only']
xx = np.arange(4)
ax.bar(xx - 0.18, [lad['gbm_' + r] for r in rungs], 0.36, color='#bdbdbd', edgecolor='k', lw=0.5, label='Gradient boosting')
ax.bar(xx + 0.18, [lad['lr_' + r] for r in rungs], 0.36, color='#1f5fa8', edgecolor='k', lw=0.5, label='Logistic regression')
for i, r in enumerate(rungs):
    ax.text(i + 0.18, lad['lr_' + r] + 0.004, '%.3f' % lad['lr_' + r], ha='center', fontsize=5.8)
    ax.text(i - 0.18, lad['gbm_' + r] + 0.004, '%.3f' % lad['gbm_' + r], ha='center', fontsize=5.8)
ax.axhline(0.5, color='k', ls=':', lw=0.8); ax.set_ylim(0.48, 0.73); ax.set_xticks(xx); ax.set_xticklabels(lbl, fontsize=6.3)
ax.set_ylabel('Out-of-fold AUROC'); ax.legend(fontsize=6, frameon=False, loc='upper right'); fig.tight_layout()
fig.savefig(os.path.join(OUT, 'cr_ladder.png')); plt.close(fig)
save(); log('ALL DONE')
