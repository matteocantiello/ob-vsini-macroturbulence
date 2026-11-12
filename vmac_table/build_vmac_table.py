"""Macroturbulence table and v_mac(T_eff, log L) fit for comparison with SDSS-V BOSS v sin i.

Inputs
  input/macroturbulence_evol.csv       832 IACOB-family stars (merged; see README)
  input/macroturbulence_catalog.csv    the full merged catalogue (927 stars, incl. those without log g; not used)
  input/vizier/holgado22.tsv           Holgado+2022 O stars (those not in the sources above are added)
  input/vizier/J_A_A_*.tsv             VizieR source tables with positions (coordinates and upper-limit flags)
  MIST v1.2 EEP tracks, [Fe/H] = 0, v/v_crit = 0.4: set the environment variable MIST_DIR to the unpacked folder
    https://mist.science/data/tarballs_v1.2/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS.txz

Outputs (this directory)
  iacob_vmac_mist.csv      one row per star
  vmac_fit_grid.csv        GP fit of log v_mac on (log T_eff, log L): mean, uncertainty, observed scatter, support
  fig_vmac_table.{pdf,png}

Run:  MIST_DIR=/path/to/MIST_..._EEPS python build_vmac_table.py
"""
import glob, os, re, sys, io
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'input')
RAWV = os.path.join(HERE, 'input', 'vizier')
MIST_DIR = os.environ.get('MIST_DIR', os.path.join(HERE, 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS'))


def vz_read_tsv(path):
    """Parse a VizieR asu-tsv file (header / units / dashes / data); numeric columns cast."""
    lines = [l for l in open(path).read().split('\n') if not l.startswith('#')]
    dash = [i for i, l in enumerate(lines) if l and set(l.replace('\t', '')) <= set('- ') and '-' in l]
    if not dash:
        raise ValueError(f'no separator row in {path}')
    d = dash[0]
    cols = [c.strip() for c in lines[d - 2].split('\t')]
    recs = [(l.split('\t') + [''] * len(cols))[:len(cols)] for l in lines[d + 1:] if l.strip()]
    df = pd.DataFrame(recs, columns=cols)
    for c in df.columns:
        v = pd.to_numeric(df[c].str.strip(), errors='coerce')
        df[c] = v if v.notna().sum() > 0.5 * max(len(df), 1) else df[c].str.strip()
    return df


def star_key(name):
    """Normalise a star designation for matching across catalogues ('HD 10125' -> 'HD10125')."""
    if not isinstance(name, str):
        return ''
    s = re.sub(r'\$.*?\$', '', name).upper().replace('\u2212', '-').replace('\u2013', '-')
    s = re.sub(r'^CL\*\s*', '', s)
    s = re.sub(r'\s+', '', s).replace('--', '-').replace('AZV', 'AV').replace('HDE', 'HD')
    s = re.sub(r'^(AV|VFTS|BI|N11|LH|HD|BD|CPD|ALS|HR|SK-\d+)0+(\d)', r'\1\2', s)
    return s
LOG_ELL_SUN = np.log10(5777.0**4 / 27400.0)       # spectroscopic luminosity of the Sun, T^4/g in cgs
SIG_LT, SIG_LELL = 0.02, 0.10                     # adopted 1-sigma errors: log T_eff (dex), log ell (dex)
EEP_ZAMS, EEP_TAMS, EEP_END = 202, 454, 707       # MIST primary EEPs: ZAMS, TAMS, end of core He burning
N_MC = 300


# ------------------------------------------------------------------------------------------- stars
SOURCE_PRIORITY = ['SimonDiaz2017', 'deBurgos2024', 'Holgado2018', 'Holgado2022', 'Burssens2020']


def base_key(key):
    """star_key without a trailing component suffix ('HD14633AAAB', 'HD36861A' -> 'HD14633', 'HD36861').
    Other components ('HD101545B', 'HD5005D') keep their letter and stay distinct."""
    return re.sub(r'(?<=\d)(AAAB|AAB|AB|AA|A)$', '', key)


def load_stars():
    V = pd.read_csv(os.path.join(DATA, 'macroturbulence_evol.csv'))
    V = V[np.isfinite(V.vmac) & (V.vmac > 0) & np.isfinite(V.logL_sp)]
    # Holgado+2018 rows whose T_eff or log g is published only as a limit are not used
    h18 = vizier_blocks(os.path.join(RAWV, 'J_A_A_613_A65.tsv'))[1]
    lim18 = set(h18.Target[(h18.l_Teff.fillna('') != '') | (h18.l_logg.fillna('') != '')].map(star_key))
    V = V[~((V.vmac_src == 'Holgado2018') & V.key.map(star_key).isin(lim18))]
    a = pd.DataFrame(dict(star=V.key, sptype=V.SpType, vmac_source=V.vmac_src, logTeff=V.logTeff_sp,
                          logg=V.logg_sp, logLspec=V.logL_sp, vmac=V.vmac, vsini=V.vsini))
    a['key'] = a.star.map(star_key)
    # Simon-Diaz+2017 flags v_mac upper limits (l_vmac = '<'); the merged catalogue does not carry the flag
    sd = vizier_blocks(os.path.join(RAWV, 'J_A_A_597_A22.tsv'))[0]
    lim = set(sd.Target[sd.l_vmac.fillna('') == '<'].map(star_key))
    a['vmac_upper_limit'] = ((a.vmac_source == 'SimonDiaz2017') & a.key.isin(lim)).astype(int)
    # an upper limit is replaced by the de Burgos+2024 measurement of the same star where there is one
    db = vizier_blocks(os.path.join(RAWV, 'J_A_A_687_A228.tsv'))[0]
    db = db[(db.vmac > 0) & (db.Teff > 0) & np.isfinite(db.logg)].copy(); db['key'] = db.Name.map(star_key)
    db = db.drop_duplicates('key').set_index('key')
    rep = (a.vmac_upper_limit == 1) & a.key.isin(db.index)
    d = db.loc[a.key[rep]]
    a.loc[rep, 'sptype'] = d.SpC.values; a.loc[rep, 'vmac_source'] = 'deBurgos2024'
    a.loc[rep, 'logTeff'] = np.log10(d.Teff.values); a.loc[rep, 'logg'] = d.logg.values
    a.loc[rep, 'logLspec'] = 4 * np.log10(d.Teff.values) - d.logg.values - LOG_ELL_SUN
    a.loc[rep, 'vmac'] = d.vmac.values; a.loc[rep, 'vsini'] = d.vsini.values; a.loc[rep, 'vmac_upper_limit'] = 0
    print(f'upper limits replaced by de Burgos+2024 measurements: {rep.sum()}')
    # Holgado+2022 O stars not already present with a measured v_mac (matched on the name without component
    # suffix); a star present only as an upper limit is resolved in dedupe()
    h = vz_read_tsv(os.path.join(RAWV, 'holgado22.tsv'))
    h = h[~h.Target.map(star_key).map(base_key).isin(set(a.key[a.vmac_upper_limit == 0].map(base_key)))].copy()
    vm = pd.to_numeric(h.vmacro, errors='coerce'); T = pd.to_numeric(h.Teff, errors='coerce') * 1e3   # kK
    L = pd.to_numeric(h.logLsp, errors='coerce')
    ok = (np.isfinite(vm) & (vm > 0) & np.isfinite(T) & np.isfinite(L)
          & (h.l_logLsp.fillna('') == '') & (h.l_Teff.fillna('') == ''))
    b = pd.DataFrame(dict(star=h.Target[ok].str.strip(), sptype=(h.SpType[ok].astype(str) + ' ' + h.LumClass[ok].astype(str)).str.strip(),
                          vmac_source='Holgado2022', logTeff=np.log10(T[ok]), logg=pd.to_numeric(h.loggt, errors='coerce')[ok],
                          logLspec=L[ok], vmac=vm[ok], vsini=pd.to_numeric(h.vsini, errors='coerce')[ok]))
    b['key'] = b.star.map(star_key); b['vmac_upper_limit'] = 0
    return pd.concat([a, b], ignore_index=True)


def dedupe(S, max_sep=10.0):
    """Drop repeated entries of one star: same name without component suffix and within max_sep arcsec.
    Keep a measured v_mac over an upper limit, then the earlier source in SOURCE_PRIORITY."""
    S = S.copy(); S['_b'] = S.key.map(base_key)
    S['_rank'] = S.vmac_upper_limit * 10 + S.vmac_source.map(SOURCE_PRIORITY.index)
    drop = []
    for _, g in S[S._b.duplicated(keep=False)].groupby('_b'):
        g = g.sort_values('_rank'); r0 = g.iloc[0]
        for i, r in g.iloc[1:].iterrows():
            c = np.cos(np.radians(r0.dec))
            sep = 3600 * np.hypot((r.ra - r0.ra) * c, r.dec - r0.dec)
            if sep < max_sep:
                drop.append(i); print(f'  duplicate: keep {r0.star} ({r0.vmac_source}), drop {r.star} ({r.vmac_source}), {sep:.1f}"')
    return S.drop(index=drop).drop(columns=['_b', '_rank']).reset_index(drop=True)


def vizier_blocks(path):
    """Split a multi-table VizieR asu-tsv file into DataFrames."""
    txt = open(path).read(); parts = re.split(r'\n(?=#Table\t)', txt); out = []
    for p in parts:
        try:
            tmp = io.StringIO(p); lines = tmp.getvalue()
            fn = os.path.join(HERE, '_tmp_block.tsv'); open(fn, 'w').write(lines)
            out.append(vz_read_tsv(fn)); os.remove(fn)
        except ValueError:
            pass
    return out


def coordinates():
    """star key -> (RA, Dec) in degrees (ICRS, J2000) from the VizieR source tables (SIMBAD positions)."""
    pos = {}
    for f in sorted(glob.glob(os.path.join(RAWV, 'J_A_A_*.tsv'))) + [os.path.join(RAWV, 'holgado22.tsv')]:
        for d in vizier_blocks(f):
            namec = next((c for c in ['Target', 'Name', 'Star', 'ID', 'Simbad', 'SimbadName'] if c in d.columns
                          and d[c].dtype == object and d[c].str.contains(r'[A-Za-z]', na=False).mean() > 0.5
                          and c != 'Simbad'), None)
            rc = next((c for c in ['_RA', '_RAJ2000', '_RA.icrs'] if c in d.columns), None)
            if namec is None or rc is None:
                continue
            dc = rc.replace('RA', 'DE')
            for n, ra, de in zip(d[namec], pd.to_numeric(d[rc], errors='coerce'), pd.to_numeric(d[dc], errors='coerce')):
                if np.isfinite(ra) and np.isfinite(de):
                    pos.setdefault(star_key(n), (ra, de))
    return pos


# ------------------------------------------------------------------------------------------- MIST
def read_track(path):
    with open(path) as f:
        for line in f:
            if 'star_age' in line:
                cols = line.lstrip('#').split(); break
    d = np.loadtxt(path, comments='#')
    t = pd.DataFrame(d, columns=cols)[['star_age', 'star_mass', 'log_L', 'log_Teff', 'log_g', 'log_R', 'phase']]
    t['eep'] = np.arange(1, len(t) + 1)
    return t


def mist_grid(mmin=3.0, mmax=150.0, dlogm=0.004):
    """Tracks interpolated in log M at fixed EEP, ZAMS to end of core He burning."""
    files = sorted(glob.glob(os.path.join(MIST_DIR, '*M.track.eep')))
    masses = np.array([int(os.path.basename(f)[:5]) / 100 for f in files])
    sel = (masses >= mmin * 0.8) & (masses <= mmax * 1.1)
    tr = {m: read_track(f) for m, f in zip(masses[sel], np.array(files)[sel])}
    ms = np.array(sorted(tr)); lm = np.log10(ms)
    rows = []
    for logm in np.arange(np.log10(mmin), np.log10(mmax) + 1e-9, dlogm):
        j = np.clip(np.searchsorted(lm, logm) - 1, 0, len(ms) - 2)
        A, B = tr[ms[j]], tr[ms[j + 1]]
        n = min(len(A), len(B), EEP_END)
        if n < EEP_TAMS:
            continue
        w = (logm - lm[j]) / (lm[j + 1] - lm[j])
        e = slice(EEP_ZAMS - 1, n)
        q = {c: (1 - w) * A[c].values[e] + w * B[c].values[e] for c in ['log_L', 'log_Teff', 'log_g', 'log_R']}
        q['log_age'] = (1 - w) * np.log10(A.star_age.values[e]) + w * np.log10(B.star_age.values[e])
        q['log_mass'] = (1 - w) * np.log10(A.star_mass.values[e]) + w * np.log10(B.star_mass.values[e])
        t_tams = 10**((1 - w) * np.log10(A.star_age.values[EEP_TAMS - 1]) + w * np.log10(B.star_age.values[EEP_TAMS - 1]))
        q['tau'] = 10**q['log_age'] / t_tams
        q['eep'] = np.arange(EEP_ZAMS, n + 1)
        q['M_ini'] = np.full(q['eep'].size, 10**logm)
        rows.append(pd.DataFrame(q))
    G = pd.concat(rows, ignore_index=True)
    G['log_ell'] = 4 * G.log_Teff - G.log_g - LOG_ELL_SUN
    return G


def place(S, G, seed=1):
    """Nearest point of the interpolated MIST grid in (log T_eff / SIG_LT, log ell / SIG_LELL); Monte Carlo errors."""
    tree = cKDTree(np.column_stack([G.log_Teff / SIG_LT, G.log_ell / SIG_LELL]))
    x0 = np.column_stack([S.logTeff / SIG_LT, S.logLspec / SIG_LELL])
    d, i = tree.query(x0)
    out = pd.DataFrame(dict(mist_dist=d, logL_mist=G.log_L.values[i], logR_mist=G.log_R.values[i],
                            M_mist=10**G.log_mass.values[i], Mini_mist=G.M_ini.values[i], tau_mist=G.tau.values[i],
                            eep_mist=G.eep.values[i]), index=S.index)
    rng = np.random.default_rng(seed)
    xs = x0[None] + rng.normal(size=(N_MC, len(S), 2))
    _, im = tree.query(xs.reshape(-1, 2)); im = im.reshape(N_MC, len(S))
    L = G.log_L.values[im]; Mi = G.M_ini.values[im]; ms = (G.eep.values[im] < EEP_TAMS)
    out['logL_mist_p16'], out['logL_mist_p84'] = np.percentile(L, 16, 0), np.percentile(L, 84, 0)
    out['Mini_mist_p16'], out['Mini_mist_p84'] = np.percentile(Mi, 16, 0), np.percentile(Mi, 84, 0)
    out['frac_MS'] = ms.mean(0)
    out['phase_mist'] = np.where(out.eep_mist < EEP_TAMS, 'MS', 'post-MS')
    return out


if __name__ == '__main__' and 'stage1' in sys.argv:
    S = load_stars(); print('stars', len(S), S.vmac_source.value_counts().to_dict())
    pos = coordinates(); S['ra'] = S.key.map(lambda k: pos.get(k, (np.nan, np.nan))[0])
    S['dec'] = S.key.map(lambda k: pos.get(k, (np.nan, np.nan))[1])
    print('with coordinates', np.isfinite(S.ra).sum(), '; missing:', list(S.star[~np.isfinite(S.ra)])[:30])
    S = dedupe(S); print('after de-duplication', len(S))
    G = mist_grid(); print('MIST grid points', len(G), 'masses', G.M_ini.min(), G.M_ini.max())
    P = place(S, G); S = pd.concat([S, P], axis=1)
    old = pd.read_csv(os.path.join(DATA, 'macroturbulence_evol.csv')).set_index('key')
    m = S.star.isin(old.index); o = old.loc[S.star[m]]
    oldL = 2 * o.ev_logR.values + 4 * (o.logTeff_sp.values - np.log10(5772))
    dL = S.logL_mist[m].values - oldL
    print(f'vs old ev_* logL: median {np.median(dL):+.3f}, MAD {np.median(abs(dL - np.median(dL))):.3f}, N {m.sum()}')
    print('dlog Mini vs old', np.round(np.percentile(np.log10(S.Mini_mist[m].values / o.ev_Mini.values), [16, 50, 84]), 3))
    print(S[['mist_dist', 'logL_mist', 'Mini_mist', 'tau_mist', 'frac_MS']].describe().round(3).to_string())
    print('logL err (p84-p16)/2 median', np.median((S.logL_mist_p84 - S.logL_mist_p16) / 2).round(3))
    print('phase', S.phase_mist.value_counts().to_dict(), ' dist>1:', (S.mist_dist > 1).sum())
    S.to_pickle(os.path.join(HERE, '_stage1.pkl'))


# ------------------------------------------------------------------------------------------- GP fit
from sklearn.gaussian_process import GaussianProcessRegressor                       # noqa: E402
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel  # noqa: E402

GRID_LT = np.round(np.arange(4.00, 4.7001, 0.005), 3)
GRID_LL = np.round(np.arange(2.40, 6.4001, 0.02), 3)
SUPPORT = (0.06, 0.25, 3)        # >= 3 stars within an ellipse of semi-axes 0.06 dex (log T_eff), 0.25 dex (log L)
K_LOCAL = 50                     # neighbours for the local scatter


def fit_gp(S, seed=0):
    X = np.column_stack([S.logTeff, S.logL_mist]); y = np.log10(S.vmac.values)
    xm, xs = X.mean(0), X.std(0); ym, ys = y.mean(), y.std()
    k = C(1.0) * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(0.05, 10), nu=2.5) + \
        WhiteKernel(0.3, (1e-3, 1.0))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=5, random_state=seed)
    gp.fit((X - xm) / xs, (y - ym) / ys)
    white = gp.kernel_.k2.noise_level
    def predict(lt, ll):
        Z = (np.column_stack([np.ravel(lt), np.ravel(ll)]) - xm) / xs
        mu, sd = gp.predict(Z, return_std=True)
        sd_latent = np.sqrt(np.clip(sd**2 - white, 0, None))           # uncertainty of the mean relation only
        return mu * ys + ym, sd_latent * ys
    return gp, predict, np.sqrt(white) * ys


def local_scatter(S, resid, lt, ll):
    """Robust s.d. (1.4826 MAD) of the residuals of the K_LOCAL nearest stars, in scaled (log T_eff, log L) units."""
    sc = np.array([SUPPORT[0], SUPPORT[1]])
    tree = cKDTree(np.column_stack([S.logTeff, S.logL_mist]) / sc)
    _, idx = tree.query(np.column_stack([np.ravel(lt), np.ravel(ll)]) / sc, k=K_LOCAL)
    r = resid[idx]; med = np.median(r, 1)
    return 1.4826 * np.median(np.abs(r - med[:, None]), 1)


def support(S, lt, ll):
    sc = np.array([SUPPORT[0], SUPPORT[1]])
    tree = cKDTree(np.column_stack([S.logTeff, S.logL_mist]) / sc)
    return np.array([len(c) for c in tree.query_ball_point(np.column_stack([np.ravel(lt), np.ravel(ll)]) / sc, r=1.0)])


# ------------------------------------------------------------------------------------------- figure
def figure(S, Gr, G):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rc('font', family='serif'); plt.rc('xtick', top=True, direction='out', labelsize=7)
    plt.rc('ytick', right=True, direction='out', labelsize=7); plt.rc('axes', labelsize=8)
    fig = plt.figure(1, figsize=(7.0, 2.9)); gs = plt.GridSpec(1, 3); gs.update(wspace=0.08)
    ax = [plt.subplot(gs[0, j]) for j in range(3)]
    fig.subplots_adjust(left=0.07, bottom=0.15, right=0.98, top=0.80)
    norm = matplotlib.colors.LogNorm(vmin=10, vmax=150); cmap = matplotlib.colormaps['turbo']
    lt = Gr.logTeff.values.reshape(len(GRID_LL), len(GRID_LT)); ll = Gr.logL.values.reshape(lt.shape)
    ok = Gr.n_support.values.reshape(lt.shape) >= SUPPORT[2]
    mean = np.where(ok, 10**Gr.logvmac_mean.values.reshape(lt.shape), np.nan)
    scat = np.where(ok, Gr.logvmac_scatter_local.values.reshape(lt.shape), np.nan)
    d = S.vmac_upper_limit == 0
    ax[0].scatter(S.logTeff[d], S.logL_mist[d], c=S.vmac[d], cmap=cmap, norm=norm, s=4, lw=0)
    ax[0].scatter(S.logTeff[~d], S.logL_mist[~d], c=S.vmac[~d], cmap=cmap, norm=norm, s=6, marker='v', lw=0)
    ax[1].pcolormesh(lt, ll, mean, cmap=cmap, norm=norm, shading='nearest')
    cs = ax[1].contour(lt, ll, mean, levels=[20, 30, 50, 70, 90], colors='k', linewidths=0.5)
    ax[1].clabel(cs, fontsize=5, fmt='%d')
    n2 = matplotlib.colors.Normalize(0.0, 0.3)
    im2 = ax[2].pcolormesh(lt, ll, scat, cmap=matplotlib.colormaps['viridis'], norm=n2, shading='nearest')
    for a in ax:
        for m in [8, 12, 16, 20, 30, 40, 60]:
            t = G[(np.abs(G.M_ini - m) == np.abs(G.M_ini - m).min()) & (G.eep <= 605)]
            a.plot(t.log_Teff, t.log_L, color='0.3', lw=0.5)
            if a is ax[0]:
                a.text(t.log_Teff.values[0] + 0.005, t.log_L.values[0] - 0.05, f'{m}', fontsize=5, ha='left', va='top', color='0.3')
        a.set_xlim(4.72, 4.00); a.set_ylim(2.5, 6.2); a.minorticks_on(); a.set_xlabel(r'Effective temperature $\log T_\mathrm{eff}~(\mathrm{K})$')
    ax[0].set_ylabel(r'Luminosity $\log L~(L_\odot)$'); ax[1].set_yticklabels([]); ax[2].set_yticklabels([])
    for (x0, x1), cm, nn, lab in [((0.07, 0.65), cmap, norm, r'(a) stars and (b) GP mean, $v_\mathrm{mac}~(\mathrm{km\,s^{-1}})$'),
                                  ((0.68, 0.98), matplotlib.colormaps['viridis'], n2, r'(c) local observed scatter of $\log v_\mathrm{mac}$ (dex)')]:
        cax = fig.add_axes([x0, 0.86, x1 - x0, 0.025])
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm=nn, orientation='horizontal', ticklocation='top')
        cb.set_label(lab, fontsize=7); cb.ax.tick_params(labelsize=6)
        if nn is norm:
            cb.set_ticks([10, 20, 30, 50, 70, 100, 150]); cb.set_ticklabels(['10', '20', '30', '50', '70', '100', '150'])
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'fig_vmac_table.{ext}'), dpi=200)


# ------------------------------------------------------------------------------------------- main
if __name__ == '__main__' and 'stage1' not in sys.argv:
    S = load_stars(); pos = coordinates()
    S['ra'] = S.key.map(lambda k: pos.get(k, (np.nan, np.nan))[0]); S['dec'] = S.key.map(lambda k: pos.get(k, (np.nan, np.nan))[1])
    S = dedupe(S); print('stars', len(S), S.vmac_source.value_counts().to_dict(), 'upper limits', S.vmac_upper_limit.sum())
    G = mist_grid(); S = pd.concat([S, place(S, G)], axis=1)
    gp, predict, white = fit_gp(S)
    det = S.vmac_upper_limit.values == 0
    gp_d, predict_d, white_d = fit_gp(S[det])
    print(f'upper limits: {np.sum(~det)}; detections-only GP white s.d. {white_d:.3f} dex')
    S['logvmac_fit'], _ = predict(S.logTeff.values, S.logL_mist.values)
    resid = np.log10(S.vmac.values) - S.logvmac_fit.values
    print('GP kernel:', gp.kernel_); print(f'white-noise s.d. {white:.3f} dex; rms residual {resid.std():.3f} dex')
    # rotation dependence of the residuals
    ok = S.vsini > 0
    sl = np.polyfit(np.log10(S.vsini[ok]), resid[ok], 1)[0]
    from scipy.stats import spearmanr
    print(f'residual vs log vsini: slope {sl:+.3f} dex/dex, Spearman {spearmanr(S.vsini[ok], resid[ok])[0]:+.2f}')
    LT, LL = np.meshgrid(GRID_LT, GRID_LL)
    mu, sd = predict(LT, LL)
    mu_d, _ = predict_d(LT, LL)
    Gr = pd.DataFrame(dict(logTeff=LT.ravel(), logL=LL.ravel(), logvmac_mean=mu, logvmac_mean_err=sd,
                           logvmac_scatter_local=local_scatter(S, resid, LT, LL), logvmac_mean_detections_only=mu_d,
                           n_support=support(S, LT, LL)))
    Gr['ok'] = (Gr.n_support >= SUPPORT[2]).astype(int)
    dd = (Gr.logvmac_mean_detections_only - Gr.logvmac_mean)[Gr.ok == 1]
    for lo, hi in [(2.4, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 6.4)]:
        k = (Gr.ok == 1) & (Gr.logL >= lo) & (Gr.logL < hi)
        print(f'detections-only minus all, logL {lo}-{hi}: median {np.median((Gr.logvmac_mean_detections_only - Gr.logvmac_mean)[k]):+.3f} dex, max {np.max(np.abs((Gr.logvmac_mean_detections_only - Gr.logvmac_mean)[k])):.3f}')
    Gr.round(4).to_csv(os.path.join(HERE, 'vmac_fit_grid.csv'), index=False)
    S['logvmac_scatter_local'] = local_scatter(S, resid, S.logTeff.values, S.logL_mist.values)
    cols = ['star', 'ra', 'dec', 'sptype', 'vmac_source', 'logTeff', 'logg', 'logLspec', 'vmac', 'vmac_upper_limit', 'vsini',
            'logL_mist', 'logL_mist_p16', 'logL_mist_p84', 'Mini_mist', 'Mini_mist_p16', 'Mini_mist_p84', 'M_mist',
            'logR_mist', 'tau_mist', 'phase_mist', 'frac_MS', 'mist_dist', 'logvmac_fit', 'logvmac_scatter_local']
    T = S[cols].copy(); T.insert(5, 'Teff', np.round(10**T.logTeff).astype(int))
    T.sort_values('ra').round(dict(ra=5, dec=5, logTeff=3, logg=2, logLspec=2, logL_mist=3, logL_mist_p16=3, logL_mist_p84=3,
                                   Mini_mist=2, Mini_mist_p16=2, Mini_mist_p84=2, M_mist=2, logR_mist=3, tau_mist=3,
                                   frac_MS=2, mist_dist=3, logvmac_fit=3, logvmac_scatter_local=3)
                                   ).to_csv(os.path.join(HERE, 'iacob_vmac_mist.csv'), index=False)
    print('wrote iacob_vmac_mist.csv', len(T), 'and vmac_fit_grid.csv', len(Gr), f'(ok {Gr.ok.sum()})')
    # numbers for the README
    for lo, hi in [(3.0, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 6.2)]:
        k = (S.logL_mist >= lo) & (S.logL_mist < hi)
        print(f'logL {lo}-{hi}: N {k.sum():3d}  vmac 16/50/84 {np.percentile(S.vmac[k], [16, 50, 84]).round(0)}  '
              f'local scatter med {np.median(S.logvmac_scatter_local[k]):.2f} dex')
    figure(S, Gr, G)
