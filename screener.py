#!/usr/bin/env python3
"""
PEA relative-value screener (long-only).

Signals
  PAIR  : Engle-Granger cointegrated pairs inside a sector, OU fit on the spread,
          long the cheap leg when |z| >= 2 (Gatev/Goetzmann/Rouwenhorst pre-selection,
          Vidyamurthy/Engle-Granger testing).
  RESID : Avellaneda-Lee s-score of the stock's residual vs its sector peer basket
          (60d window, AR(1)->OU, kappa > 252/30), long when s < -1.25.
Overlays
  VALUE : Damodaran-style market regressions of log multiples on fundamentals with
          sector fixed effects -> residual cheapness.
  MACRO : weekly ridge regression of each stock on macro factor moves (rates, oil,
          gas, copper, EURUSD, VSTOXX, gold; Euro Stoxx 50 as control) x the current
          4-week factor regime.

Output: data/latest.json (+ data/history/<ISO week>.json). See METHODOLOGY.md.
Run: python screener.py            (real data via yfinance)
     python screener.py --synthetic (offline self-test with simulated data)
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
VERSION = "1.0"

FX_BY_SUFFIX = {".ST": "SEK", ".CO": "DKK", ".OL": "NOK"}
NORDIC = {"SEK", "DKK", "NOK"}

# key, label, yahoo symbols (first that works), transform, unit
FACTORS = [
    ("SX5E", "Euro Stoxx 50", ["^STOXX50E"], "log", "pts"),
    ("VSTOXX", "VSTOXX", ["^V2TX", "V2TX.DE", "^VIX"], "log", "pts"),
    ("US10Y", "Taux US 10 ans", ["^TNX"], "diff", "%"),
    ("EURUSD", "EUR/USD", ["EURUSD=X"], "log", ""),
    ("BRENT", "Brent", ["BZ=F"], "log", "$"),
    ("TTF", "Gaz TTF", ["TTF=F"], "log", "€/MWh"),
    ("COPPER", "Cuivre", ["HG=F"], "log", "$/lb"),
    ("GOLD", "Or", ["GC=F"], "log", "$"),
]
MARKET_KEY = "SX5E"

# thresholds (documented in METHODOLOGY.md)
FORM_WIN = 252          # pair formation window (days)
RESID_WIN = 60          # Avellaneda-Lee window
MIN_CORR = 0.50         # pre-filter on daily return correlation
COINT_P = 0.05
HL_MIN, HL_MAX = 3, 60
RESID_HL_MIN = 3        # a weekly process cannot trade a residual that reverts in 1-2 days
BETA_MIN, BETA_MAX = 0.25, 4.0
MAX_EXP = 0.30          # sanity cap on the expected move to TP
Z_ENTRY, Z_WATCH, Z_EXIT, Z_STOP = 2.0, 1.5, 0.5, 3.5
S_ENTRY, S_WATCH, S_EXIT = 1.25, 1.0, 0.5
KAPPA_MIN = 252 / 30
N_SIGNALS, N_WATCH = 25, 15


# ----------------------------------------------------------------------------- utils
def clip(x, lo, hi):
    return max(lo, min(hi, x))


def fnum(x, nd=4):
    """JSON-safe rounded float (None for nan/inf)."""
    try:
        if x is None:
            return None
        x = float(x)
        if not math.isfinite(x):
            return None
        return round(x, nd)
    except Exception:
        return None


def ccy_of(ticker):
    for suf, c in FX_BY_SUFFIX.items():
        if ticker.endswith(suf):
            return c
    return "EUR"


def iso_week(d):
    y, w, _ = d.isocalendar()
    return f"{y}-S{w:02d}"


# ----------------------------------------------------------------------------- data
def yf_close(tickers, period="3y"):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                                 progress=False, threads=True, group_by="column")
                if df is None or df.empty:
                    raise RuntimeError("empty")
                close = df["Close"]
                if isinstance(close, pd.Series):
                    close = close.to_frame(chunk[0])
                for t in chunk:
                    if t in close.columns and close[t].notna().sum() > 0:
                        out[t] = close[t]
                break
            except Exception as e:  # noqa
                print(f"  download retry {attempt + 1} ({e})", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index()


FUND_KEYS = ["shortName", "forwardPE", "trailingPE", "enterpriseToEbitda", "priceToBook",
             "returnOnEquity", "earningsGrowth", "revenueGrowth", "ebitdaMargins", "beta",
             "payoutRatio", "marketCap", "targetMeanPrice", "numberOfAnalystOpinions",
             "recommendationKey", "currency"]


def yf_fundamentals(tickers):
    import yfinance as yf
    res = {}
    for n, t in enumerate(tickers):
        d = {}
        tk = yf.Ticker(t)
        for attempt in range(2):
            try:
                info = tk.info or {}
                d = {k: info.get(k) for k in FUND_KEYS}
                break
            except Exception:
                time.sleep(3)
        try:
            cal = tk.calendar
            ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if ed:
                ed = ed[0] if isinstance(ed, (list, tuple)) else ed
                if str(ed)[:10] >= dt.date.today().isoformat():
                    d["earnings_date"] = str(ed)[:10]
        except Exception:
            pass
        res[t] = d
        if n % 25 == 0:
            print(f"  fundamentals {n}/{len(tickers)}", file=sys.stderr)
        time.sleep(0.3)
    return res


def synthetic_data(universe, seed=7):
    """Simulated prices/fundamentals/factors to self-test the pipeline offline."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=760)
    n = len(idx)
    fac = {}
    for key, _, _, tr, _ in FACTORS:
        steps = rng.normal(0, 0.01, n)
        fac[key] = pd.Series(np.exp(np.cumsum(steps)) * (4.0 if tr == "diff" else 100), index=idx)
    mkt = np.diff(np.log(fac["SX5E"].values), prepend=np.log(fac["SX5E"].values[0]))
    prices = {}
    sectors = universe.groupby("sector")
    for sec, g in sectors:
        sec_f = rng.normal(0, 0.008, n)
        tick = list(g.ticker)
        for j, t in enumerate(tick):
            beta = rng.uniform(0.6, 1.4)
            idio = rng.normal(0, 0.012, n)
            lp = np.cumsum(beta * mkt + sec_f + idio)
            if j % 2 == 1:  # make every other name cointegrated with its predecessor
                base = np.log(prices[tick[j - 1]].values)
                ou = np.zeros(n)
                for k in range(1, n):
                    ou[k] = 0.93 * ou[k - 1] + rng.normal(0, 0.012)
                lp = base * 0.9 + 0.4 + ou
                if j == 1:
                    lp[-1] -= 0.08  # force a live dislocation
            prices[t] = pd.Series(np.exp(lp - lp[0]) * rng.uniform(10, 200), index=idx)
    px = pd.DataFrame(prices)
    fx = {c: pd.Series(np.exp(np.cumsum(rng.normal(0, 0.003, n))) * v, index=idx)
          for c, v in [("SEK", 11.2), ("DKK", 7.46), ("NOK", 11.6)]}
    fund = {}
    for t in px.columns:
        fund[t] = dict(shortName=t, forwardPE=float(rng.uniform(6, 30)),
                       enterpriseToEbitda=float(rng.uniform(4, 18)),
                       priceToBook=float(rng.uniform(0.5, 4)),
                       returnOnEquity=float(rng.uniform(0.02, 0.25)),
                       earningsGrowth=float(rng.normal(0.08, 0.15)),
                       revenueGrowth=float(rng.normal(0.05, 0.08)),
                       ebitdaMargins=float(rng.uniform(0.08, 0.4)),
                       beta=float(rng.uniform(0.6, 1.5)), payoutRatio=float(rng.uniform(0.2, 0.7)),
                       targetMeanPrice=float(px[t].iloc[-1] * rng.uniform(0.9, 1.3)),
                       numberOfAnalystOpinions=int(rng.integers(3, 30)),
                       earnings_date=str((idx[-1] + pd.Timedelta(days=int(rng.integers(2, 60)))).date()))
    return px, pd.DataFrame(fx), pd.DataFrame(fac), {k: k for k in fac}, fund


# ----------------------------------------------------------------------------- stats
def ar1_ou(x):
    """AR(1) fit x_{t+1} = a + b x_t + e -> OU params. None if not mean reverting."""
    x = np.asarray(x, float)
    if len(x) < 20 or not np.all(np.isfinite(x)):
        return None
    xl, y = x[:-1], x[1:]
    A = np.column_stack([np.ones_like(xl), xl])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    if not (0 < b < 0.9999):
        return None
    e = y - A @ coef
    var_e = e.var(ddof=2)
    m = a / (1 - b)
    s_eq = math.sqrt(var_e / (1 - b * b))
    if s_eq <= 0:
        return None
    return dict(b=b, m=m, s_eq=s_eq, hl=math.log(2) / -math.log(b),
                kappa=-math.log(b) * 252, z=(x[-1] - m) / s_eq)


def hurst(ts, max_lag=40):
    ts = np.asarray(ts, float)
    lags = np.arange(2, min(max_lag, len(ts) // 4))
    tau = np.array([np.std(ts[l:] - ts[:-l]) for l in lags])
    ok = tau > 0
    if ok.sum() < 5:
        return None
    return float(np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0])


def ols(y, x):
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0], coef[1], y - A @ coef


def kalman_beta(y, x, delta=1e-4):
    """Dynamic hedge ratio (random-walk state for alpha, beta). Returns final beta."""
    n = len(y)
    a0, b0, res = ols(y[:60], x[:60])
    theta = np.array([a0, b0])
    P = np.eye(2) * 1e-2
    Vw = delta / (1 - delta) * np.eye(2)
    ve = max(res.var(), 1e-8)
    for t in range(n):
        F = np.array([1.0, x[t]])
        R = P + Vw
        q = F @ R @ F + ve
        e = y[t] - F @ theta
        K = R @ F / q
        theta = theta + K * e
        P = R - np.outer(K, F) @ R
    return float(theta[1])


def coint_p(y, x):
    from statsmodels.tsa.stattools import coint
    try:
        return float(coint(y, x, trend="c", maxlag=1, autolag=None)[1])
    except Exception:
        return 1.0


# ----------------------------------------------------------------------------- PAIRS
def wf_backtest_pair(la, lb, lng, win=FORM_WIN, refit=21):
    """Walk-forward, out-of-sample: params from trailing window, refit monthly.
    lng='A' -> long A when z<=-2 ; lng='B' -> long B when z>=+2. Long-only returns,
    plus beta-hedged spread return for reference."""
    n = len(la)
    trades, pos = [], None
    for start in range(win, n, refit):
        y, x = la[start - win:start], lb[start - win:start]
        a, b, s = ols(y, x)
        ou = ar1_ou(s)
        if not ou or b <= 0:
            continue
        maxd = int(clip(round(2 * ou["hl"]), 10, 60))
        for t in range(start, min(start + refit, n)):
            z = (la[t] - a - b * lb[t] - ou["m"]) / ou["s_eq"]
            zz = z if lng == "A" else -z
            if pos is None:
                if zz <= -Z_ENTRY:
                    pos = dict(t0=t, maxd=maxd, b=b)
            else:
                held = t - pos["t0"]
                if zz >= -Z_EXIT or zz <= -Z_STOP or held >= pos["maxd"] or t == n - 1:
                    t0 = pos["t0"]
                    ra = la[t] - la[t0]
                    rb = lb[t] - lb[t0]
                    r_long = ra if lng == "A" else rb
                    r_hedge = (ra - pos["b"] * rb) if lng == "A" else (rb - ra / pos["b"])
                    if t != n - 1 or zz >= -Z_EXIT:
                        trades.append(dict(days=held, r=math.expm1(r_long), rh=math.expm1(r_hedge),
                                           win=zz >= -Z_EXIT))
                    pos = None
    if not trades:
        return dict(n=0)
    return dict(n=len(trades), hit=float(np.mean([t["win"] for t in trades])),
                avg=float(np.mean([t["r"] for t in trades])),
                avg_hedged=float(np.mean([t["rh"] for t in trades])),
                pos_rate=float(np.mean([t["r"] > 0 for t in trades])),
                days=float(np.mean([t["days"] for t in trades])))


def scan_pairs(lpx, universe):
    """lpx: DataFrame of log prices in EUR."""
    rets = lpx.diff()
    found, tested, cointegrated = [], 0, 0
    for sec, g in universe.groupby("sector"):
        names = [t for t in g.ticker if t in lpx.columns]
        if len(names) < 2:
            continue
        sub = rets[names].iloc[-FORM_WIN:]
        corr = sub.corr()
        for A, B in combinations(names, 2):
            c = corr.loc[A, B]
            if not np.isfinite(c) or c < MIN_CORR:
                continue
            la_full, lb_full = lpx[A].values, lpx[B].values
            ok = np.isfinite(la_full) & np.isfinite(lb_full)
            if ok[-FORM_WIN:].sum() < FORM_WIN - 5:
                continue
            la_full = pd.Series(la_full).ffill().bfill().values
            lb_full = pd.Series(lb_full).ffill().bfill().values
            y, x = la_full[-FORM_WIN:], lb_full[-FORM_WIN:]
            tested += 1
            a, b, s = ols(y, x)
            if not (BETA_MIN <= b <= BETA_MAX):
                continue
            p = coint_p(y, x)
            if p >= COINT_P:
                continue
            cointegrated += 1
            ou = ar1_ou(s)
            if not ou or not (HL_MIN <= ou["hl"] <= HL_MAX):
                continue
            H = hurst(s)
            z = ou["z"]
            if abs(z) < Z_WATCH:
                found.append(None)  # counted as cointegrated, not dislocated
                continue
            # robustness
            p2 = coint_p(la_full[-2 * FORM_WIN:], lb_full[-2 * FORM_WIN:]) if len(la_full) >= 2 * FORM_WIN else None
            h = FORM_WIN // 2
            _, b1, _ = ols(y[:h], x[:h])
            _, b2, _ = ols(y[h:], x[h:])
            beta_stab = abs(b1 - b2) / max(abs(b), 1e-6)
            kb = kalman_beta(y, x)
            lng = "A" if z < 0 else "B"
            # leg-specific OU (long leg as dependent variable) -> symmetric TP/SL
            if lng == "A":
                leg_ou = dict(ou, last=s[-1])
            else:
                _, bb, sb = ols(x, y)
                o2 = ar1_ou(sb)
                if not o2 or not (BETA_MIN <= bb <= BETA_MAX):
                    continue
                leg_ou = dict(o2, last=sb[-1])
            bt = wf_backtest_pair(la_full, lb_full, lng)
            found.append(dict(sector=sec, A=A, B=B, corr=float(c), alpha=a, beta=b, p=p, p_long=p2,
                              ou=ou, hurst=H, beta_stab=beta_stab, kalman_beta=kb, z=z, long=lng,
                              bt=bt, spread=s, leg_ou=leg_ou))
    found = [f for f in found if f]
    return found, tested, cointegrated


# ----------------------------------------------------------------------------- RESID
def sector_baskets(rets, universe):
    out = {}
    for sec, g in universe.groupby("sector"):
        names = [t for t in g.ticker if t in rets.columns]
        for t in names:
            peers = [p for p in names if p != t]
            if len(peers) >= 2:
                out[t] = rets[peers].mean(axis=1, skipna=True)
    return out


def sscore_at(r, rb):
    a, b, eps = ols(r, rb)
    X = np.cumsum(eps)
    ou = ar1_ou(X)
    if not ou:
        return None
    ou["beta"] = b
    ou["X"] = X
    return ou


def scan_resid(rets, baskets, lookback=504):
    out = []
    for t, rb_s in baskets.items():
        r = rets[t].values
        rb = rb_s.values
        ok = np.isfinite(r) & np.isfinite(rb)
        r, rb = np.where(ok, r, 0.0), np.where(ok, rb, 0.0)
        if ok[-RESID_WIN:].sum() < RESID_WIN - 3:
            continue
        cur = sscore_at(r[-RESID_WIN:], rb[-RESID_WIN:])
        if not cur or cur["kappa"] < KAPPA_MIN:
            continue
        s = cur["z"]
        # rolling path + walk-forward backtest (each day uses only trailing 60d)
        n = len(r)
        start = max(RESID_WIN, n - lookback)
        path, trades, pos = [], [], None
        for k in range(start, n + 1):
            o = sscore_at(r[k - RESID_WIN:k], rb[k - RESID_WIN:k])
            sv = o["z"] if (o and o["kappa"] >= KAPPA_MIN) else None
            path.append(sv)
            day = k - 1
            if sv is None:
                continue
            if pos is None:
                if sv < -S_ENTRY and k < n:
                    pos = dict(t0=day, maxd=int(clip(round(2 * o["hl"]), 5, 40)), beta=o["beta"])
            else:
                held = day - pos["t0"]
                if sv > -S_EXIT or held >= pos["maxd"]:
                    seg = slice(pos["t0"] + 1, day + 1)
                    rl = float(np.sum(r[seg]))
                    rh = float(np.sum(r[seg] - pos["beta"] * rb[seg]))
                    trades.append(dict(days=held, r=math.expm1(rl), rh=math.expm1(rh), win=sv > -S_EXIT))
                    pos = None
        bt = dict(n=0)
        if trades:
            bt = dict(n=len(trades), hit=float(np.mean([x["win"] for x in trades])),
                      avg=float(np.mean([x["r"] for x in trades])),
                      avg_hedged=float(np.mean([x["rh"] for x in trades])),
                      pos_rate=float(np.mean([x["r"] > 0 for x in trades])),
                      days=float(np.mean([x["days"] for x in trades])))
        out.append(dict(ticker=t, s=s, ou=cur, bt=bt, path=path[-180:]))
    return out


# ----------------------------------------------------------------------------- VALUE
def winsor(s, lo=0.05, hi=0.95):
    return s.clip(s.quantile(lo), s.quantile(hi))


def value_regression(df, ycol, xcols, rng, mask=None, min_n=25):
    d = df if mask is None else df[mask]
    d = d[[ycol, "sector"] + xcols].copy()
    d = d[(d[ycol] > rng[0]) & (d[ycol] < rng[1])]
    for c in xcols:
        d[c] = winsor(d[c].astype(float))
        d[c] = d[c].fillna(d[c].median())
    d = d.dropna(subset=[ycol])
    if len(d) < min_n:
        return pd.Series(dtype=float), None
    y = np.log(d[ycol].astype(float).values)
    X = d[xcols].astype(float)
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    dummies = pd.get_dummies(d["sector"], drop_first=True, dtype=float)
    M = np.column_stack([np.ones(len(d)), X.values, dummies.values])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    fit = M @ coef
    resid = y - fit
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else None
    return pd.Series(resid, index=d.index), r2


def value_block(universe, fund):
    df = universe.set_index("ticker").copy()
    for k in ["forwardPE", "enterpriseToEbitda", "priceToBook", "returnOnEquity", "earningsGrowth",
              "revenueGrowth", "ebitdaMargins", "beta", "payoutRatio"]:
        df[k] = [fund.get(t, {}).get(k) for t in df.index]
        df[k] = pd.to_numeric(df[k], errors="coerce")
    fin = df["sector"].isin(["Banques", "Assurance", "Services financiers"])
    r_pe, r2_pe = value_regression(df, "forwardPE", ["earningsGrowth", "returnOnEquity", "beta", "payoutRatio"], (2, 80))
    r_ev, r2_ev = value_regression(df, "enterpriseToEbitda", ["revenueGrowth", "ebitdaMargins", "beta"], (1, 60), mask=~fin)
    r_pb, r2_pb = value_regression(df, "priceToBook", ["returnOnEquity", "beta"], (0.1, 10), mask=fin, min_n=20)
    zs = {}
    for name, r in [("pe", r_pe), ("ev", r_ev), ("pb", r_pb)]:
        if len(r):
            zs[name] = r / r.std()
    out = {}
    med = df.groupby("sector")[["forwardPE", "enterpriseToEbitda", "priceToBook"]].median()
    for t in df.index:
        comp = {}
        for name, r in [("pe", r_pe), ("ev", r_ev), ("pb", r_pb)]:
            if t in r.index:
                comp[name] = dict(resid_pct=math.expm1(r[t]), z=float(zs[name][t]))
        weights = {"pb": 2.0, "pe": 1.0, "ev": 1.0} if fin[t] else {"pe": 1.0, "ev": 1.0}
        num = sum(weights.get(k, 0) * v["z"] for k, v in comp.items() if k in weights)
        den = sum(weights.get(k, 0) for k in comp if k in weights)
        sec = df.loc[t, "sector"]
        out[t] = dict(
            fpe=fnum(df.loc[t, "forwardPE"], 2), ev_ebitda=fnum(df.loc[t, "enterpriseToEbitda"], 2),
            pb=fnum(df.loc[t, "priceToBook"], 2), roe=fnum(df.loc[t, "returnOnEquity"], 4),
            eps_growth=fnum(df.loc[t, "earningsGrowth"], 4), rev_growth=fnum(df.loc[t, "revenueGrowth"], 4),
            sec_fpe=fnum(med.loc[sec, "forwardPE"], 2), sec_ev_ebitda=fnum(med.loc[sec, "enterpriseToEbitda"], 2),
            sec_pb=fnum(med.loc[sec, "priceToBook"], 2),
            components={k: {kk: fnum(vv, 4) for kk, vv in v.items()} for k, v in comp.items()},
            value_z=fnum(num / den, 3) if den else None)
    r2 = dict(pe=fnum(r2_pe, 3), ev=fnum(r2_ev, 3), pb=fnum(r2_pb, 3),
              n_pe=int(len(r_pe)), n_ev=int(len(r_ev)), n_pb=int(len(r_pb)))
    return out, r2


# ----------------------------------------------------------------------------- MACRO
def factor_changes(fac_w):
    ch = {}
    for key, _, _, tr, _ in FACTORS:
        if key not in fac_w:
            continue
        s = fac_w[key]
        ch[key] = s.diff() if tr == "diff" else np.log(s).diff()
    return pd.DataFrame(ch)


def macro_block(px_eur, fac, universe, weeks=104, lam=2.0):
    w = px_eur.resample("W-FRI").last()
    fw = fac.resample("W-FRI").last().ffill()
    rw = np.log(w).diff()
    fch = factor_changes(fw)
    common = rw.index.intersection(fch.index)
    rw, fch = rw.loc[common].iloc[-weeks:], fch.loc[common].iloc[-weeks:]
    fch = fch.dropna(axis=1, thresh=int(0.8 * len(fch))).fillna(0)
    sd = fch.std().replace(0, np.nan)
    Z = (fch / sd).fillna(0)
    keys = list(Z.columns)
    # regime: standardized 4-week move and 13-week trend
    full = factor_changes(fac.resample("W-FRI").last().ffill()).iloc[-weeks:]
    regime = {}
    for k in keys:
        s = full[k].dropna()
        sd4 = s.std() * 2.0
        regime[k] = dict(z4=clip(float(s.iloc[-4:].sum() / sd4), -3, 3) if sd4 > 0 else 0.0,
                         z13=clip(float(s.iloc[-13:].sum() / (s.std() * math.sqrt(13))), -3, 3) if s.std() > 0 else 0.0,
                         chg1w=float(s.iloc[-1]), chg4w=float(s.iloc[-4:].sum()))
    X = Z.values
    XtX = X.T @ X + lam * np.eye(len(keys))
    betas, scores, r2s = {}, {}, {}
    for t in rw.columns:
        y = rw[t].values
        ok = np.isfinite(y)
        if ok.sum() < 52:
            continue
        Xo, yo = X[ok], y[ok] - np.nanmean(y)
        bvec = np.linalg.solve(Xo.T @ Xo + lam * np.eye(len(keys)), Xo.T @ yo)
        res = yo - Xo @ bvec
        r2s[t] = 1 - res.var() / yo.var() if yo.var() > 0 else 0
        betas[t] = dict(zip(keys, bvec))
        contrib = {k: betas[t][k] * regime[k]["z4"] for k in keys if k != MARKET_KEY}
        scores[t] = dict(score=sum(contrib.values()), contrib=contrib)
    sc = pd.Series({t: v["score"] for t, v in scores.items()})
    pct = sc.rank(pct=True) * 100
    stock = {}
    for t, v in scores.items():
        drivers = sorted(v["contrib"].items(), key=lambda kv: -abs(kv[1]))[:3]
        stock[t] = dict(score=fnum(v["score"] * 100, 3), pctile=fnum(pct[t], 1), r2=fnum(r2s[t], 3),
                        betas={k: fnum(b * 100, 3) for k, b in betas[t].items()},
                        drivers=[dict(f=k, contrib=fnum(c * 100, 3), beta=fnum(betas[t][k] * 100, 3),
                                      z4=fnum(regime[k]["z4"], 2)) for k, c in drivers])
    # sector aggregates
    sectors = []
    for sec, g in universe.groupby("sector"):
        names = [t for t in g.ticker if t in stock]
        if not names:
            continue
        bmean = {k: float(np.mean([betas[t][k] for t in names])) * 100 for k in keys}
        tail = float(np.mean([scores[t]["score"] for t in names])) * 100
        sectors.append(dict(sector=sec, n=len(names), betas={k: fnum(v, 3) for k, v in bmean.items()},
                            tailwind=fnum(tail, 3)))
    return stock, sectors, regime, keys


def regime_text(regime, labels):
    bits = []
    order = sorted(regime.items(), key=lambda kv: -abs(kv[1]["z4"]))
    for k, v in order[:4]:
        if abs(v["z4"]) < 0.5:
            continue
        lab = labels.get(k, k)
        if k == "US10Y":
            mv = f"{v['chg4w'] * 100:+.0f} pb"
        else:
            mv = f"{math.expm1(v['chg4w']) * 100:+.1f} %"
        bits.append(f"{lab} {mv} sur 4 sem. (z {v['z4']:+.1f})")
    return "; ".join(bits) if bits else "Régime macro calme : aucun facteur au-delà de 0,5σ sur 4 semaines."


# ----------------------------------------------------------------------------- SCORING
def levels_from_gap(price, gap_log, s_eq_leg, daily_vol, hl):
    """TP at full reversion of the log-gap, SL at max(1.5 sigma_eq, 1.25 vol*sqrt(hl))."""
    tp = price * math.exp(gap_log)
    risk = max(1.5 * s_eq_leg, 1.25 * daily_vol * math.sqrt(max(hl, 1)))
    sl = price * math.exp(-risk)
    exp_pct = tp / price - 1
    risk_pct = 1 - sl / price
    return dict(entry=fnum(price, 4), tp=fnum(tp, 4), sl=fnum(sl, 4), exp_pct=fnum(exp_pct, 4),
                risk_pct=fnum(risk_pct, 4), rr=fnum(exp_pct / risk_pct if risk_pct > 0 else None, 2),
                horizon_days=int(clip(round(2 * hl), 5, 60)))


def score_signal(strength, quality, value_z, macro_pct, rr, flags):
    v = 0.5 if value_z is None else clip(0.5 - value_z / 2, 0, 1)
    m = 0.5 if macro_pct is None else macro_pct / 100
    r = 0.5 if rr is None else clip(rr / 2, 0, 1)
    s = 100 * (0.30 * strength + 0.25 * quality + 0.20 * v + 0.20 * m + 0.05 * r)
    if "falling_knife" in flags:
        s *= 0.85
    if "earnings_soon" in flags:
        s *= 0.92
    comps = dict(stat=fnum(strength, 3), quality=fnum(quality, 3), value=fnum(v, 3), macro=fnum(m, 3), rr=fnum(r, 3))
    return s, comps


def bt_quality(bt):
    if not bt or bt.get("n", 0) < 3:
        return None
    return 0.6 * bt["hit"] + 0.4 * bt.get("pos_rate", 0)


# ----------------------------------------------------------------------------- TRACKING
def track_history(hist_dir, px_local, as_of):
    """Score previously published ideas against realized prices (first touch of TP/SL)."""
    rows = []
    if not os.path.isdir(hist_dir):
        return rows, {}
    for fn in sorted(os.listdir(hist_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            h = json.load(open(os.path.join(hist_dir, fn)))
        except Exception:
            continue
        d0 = pd.Timestamp(h.get("as_of"))
        if d0 >= as_of:
            continue
        for idea in h.get("ideas", []):
            if idea.get("tier") != "SIGNAL":
                continue
            t = idea["long"]["ticker"]
            lv = idea["levels"]
            if t not in px_local or not lv.get("entry"):
                continue
            path = px_local[t].loc[px_local.index > d0].dropna()
            if path.empty:
                continue
            status, close, cdate = "open", float(path.iloc[-1]), None
            limit = d0 + pd.tseries.offsets.BDay(lv["horizon_days"])
            for d, p in path.items():
                if p >= lv["tp"]:
                    status, close, cdate = "tp", float(p), d
                    break
                if p <= lv["sl"]:
                    status, close, cdate = "sl", float(p), d
                    break
                if d >= limit:
                    status, close, cdate = "expired", float(p), d
                    break
            rows.append(dict(week=h.get("week"), id=idea["id"], ticker=t, name=idea["long"]["name"],
                             type=idea["type"], entry=lv["entry"], tp=lv["tp"], sl=lv["sl"],
                             status=status, last=fnum(close, 4),
                             ret=fnum(close / lv["entry"] - 1, 4),
                             close_date=str(cdate.date()) if cdate is not None else None))
    closed = [r for r in rows if r["status"] != "open"]
    summ = dict(n=len(rows), closed=len(closed),
                hit=fnum(np.mean([r["status"] == "tp" for r in closed]), 3) if closed else None,
                avg_ret=fnum(np.mean([r["ret"] for r in closed]), 4) if closed else None,
                open=len(rows) - len(closed))
    return rows[-200:], summ


# ----------------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()

    universe = pd.read_csv(os.path.join(ROOT, "universe.csv"))
    universe["currency"] = universe.ticker.map(ccy_of)
    tickers = list(universe.ticker)
    warnings_out = []
    t0 = time.time()

    if args.synthetic:
        px, fxdf, fac, fac_sym, fund = synthetic_data(universe)
    else:
        print("prices...", file=sys.stderr)
        px = yf_close(tickers)
        print("fx + factors...", file=sys.stderr)
        fxraw = yf_close([f"EUR{c}=X" for c in ["SEK", "DKK", "NOK"]])
        fxdf = pd.DataFrame({c: fxraw.get(f"EUR{c}=X") for c in ["SEK", "DKK", "NOK"]})
        fac, fac_sym = {}, {}
        allsyms = sorted({s for f in FACTORS for s in f[2]})
        fraw = yf_close(allsyms)
        for key, _, syms, _, _ in FACTORS:
            for s in syms:
                if s in fraw and fraw[s].notna().sum() > 300:
                    fac[key], fac_sym[key] = fraw[s], s
                    break
            if key not in fac:
                warnings_out.append(f"facteur {key} indisponible")
        fac = pd.DataFrame(fac)
        print("fundamentals...", file=sys.stderr)
        fund = yf_fundamentals([t for t in tickers if t in px.columns])

    # --- clean prices (drop today's bar if the run happens before the European close)
    px = px.dropna(how="all")
    now = dt.datetime.utcnow()
    if len(px) and px.index[-1].date() == now.date() and now.hour < 17 and not args.synthetic:
        px = px.iloc[:-1]
    px = px.ffill(limit=3)
    keep = [t for t in px.columns if px[t].iloc[-FORM_WIN:].notna().sum() >= FORM_WIN - 10]
    missing = sorted(set(tickers) - set(keep))
    if missing:
        warnings_out.append(f"{len(missing)} tickers sans historique suffisant : {', '.join(missing[:40])}")
    px = px[keep]
    universe = universe[universe.ticker.isin(keep)].reset_index(drop=True)
    fxdf = fxdf.reindex(px.index).ffill().bfill()
    px_eur = px.copy()
    for t in keep:
        c = ccy_of(t)
        if c != "EUR":
            px_eur[t] = px[t] / fxdf[c]
    lpx = np.log(px_eur)
    rets = lpx.diff()
    as_of = px.index[-1]
    fac = fac.reindex(px.index).ffill() if len(fac) else fac

    # --- blocks
    print("pairs...", file=sys.stderr)
    pairs, n_tested, n_coint = scan_pairs(lpx, universe)
    print("residuals...", file=sys.stderr)
    resid = scan_resid(rets, sector_baskets(rets, universe))
    print("value + macro...", file=sys.stderr)
    value, value_r2 = value_block(universe, fund)
    macro_stock, macro_sectors, regime, fkeys = macro_block(px_eur, fac, universe)
    labels = {k: l for k, l, *_ in FACTORS}
    if fac_sym.get("VSTOXX") == "^VIX":
        labels["VSTOXX"] = "VIX (proxy VSTOXX)"

    meta_u = universe.set_index("ticker")
    vol20 = rets.iloc[-20:].std()
    vol60 = rets.iloc[-60:].std()
    ret = lambda t, n: fnum(px[t].iloc[-1] / px[t].iloc[-1 - n] - 1, 4)

    def stock_card(t):
        f = fund.get(t, {})
        return dict(ticker=t, name=meta_u.loc[t, "name"], sector=meta_u.loc[t, "sector"],
                    country=meta_u.loc[t, "country"], currency=ccy_of(t),
                    price=fnum(px[t].iloc[-1], 4), ret_1w=ret(t, 5), ret_1m=ret(t, 21), ret_3m=ret(t, 63),
                    vol_ann=fnum(vol60[t] * math.sqrt(252), 4),
                    target=fnum(f.get("targetMeanPrice"), 2), n_analysts=f.get("numberOfAnalystOpinions"),
                    earnings_date=f.get("earnings_date"),
                    nordic=ccy_of(t) in NORDIC)

    def flags_for(t):
        fl = []
        r20 = px[t].iloc[-1] / px[t].iloc[-21] - 1
        if r20 < -0.15:
            fl.append("falling_knife")
        ed = fund.get(t, {}).get("earnings_date")
        if ed:
            try:
                dd = (pd.Timestamp(ed) - as_of).days
                if 0 <= dd <= 10:
                    fl.append("earnings_soon")
            except Exception:
                pass
        vz = value.get(t, {}).get("value_z")
        if vz is not None and vz > 0.75:
            fl.append("rich_on_fundamentals")
        if ccy_of(t) in NORDIC:
            fl.append("nordic_listing")
        return fl

    candidates = []
    # PAIR candidates
    for p in pairs:
        lng_t = p["A"] if p["long"] == "A" else p["B"]
        ref_t = p["B"] if p["long"] == "A" else p["A"]
        ou = p["ou"]
        lo = p["leg_ou"]
        lv = levels_from_gap(float(px[lng_t].iloc[-1]), lo["m"] - lo["last"], lo["s_eq"], float(vol20[lng_t]), ou["hl"])
        if lv["exp_pct"] is None or lv["exp_pct"] <= 0 or lv["exp_pct"] > MAX_EXP:
            continue
        az = abs(p["z"])
        tier = "SIGNAL" if (az >= Z_ENTRY and p["beta_stab"] <= 1.0) else "WATCH"
        strength = clip((az - 1.5) / 1.5, 0, 1)
        q = 0.4 * (1.0 if p["p"] < 0.01 else 0.6) + 0.2 * (1.0 if (p["p_long"] or 1) < 0.10 else 0.0) \
            + 0.2 * (1.0 if p["beta_stab"] < 0.3 else 0.5 if p["beta_stab"] < 0.6 else 0.0) \
            + 0.2 * (1.0 if (p["hurst"] or 1) < 0.45 else 0.5)
        bq = bt_quality(p["bt"])
        quality = q if bq is None else 0.5 * q + 0.5 * bq
        fl = flags_for(lng_t)
        if p["beta_stab"] > 0.6:
            fl.append("unstable_hedge_ratio")
        vz = value.get(lng_t, {}).get("value_z")
        mp = macro_stock.get(lng_t, {}).get("pctile")
        sc, comps = score_signal(strength, quality, vz, mp, lv["rr"], fl)
        # chart series: last 180d z (current params) + rebased legs
        n180 = 180
        zser = (p["spread"][-n180:] - ou["m"]) / ou["s_eq"]
        dates = [d.strftime("%Y-%m-%d") for d in px.index[-n180:]]
        la = px_eur[lng_t].iloc[-n180:]
        lb = px_eur[ref_t].iloc[-n180:]
        sign = 1 if p["long"] == "A" else -1
        candidates.append(dict(
            type="PAIR", tier=tier, sector=p["sector"], long_t=lng_t, ref_t=ref_t, score=sc, comps=comps, flags=fl,
            stats=dict(z=fnum(p["z"] if p["long"] == "A" else -p["z"], 3),  # long-leg view: <0 = cheap
                       coint_p=fnum(p["p"], 4), coint_p_2y=fnum(p["p_long"], 4), half_life=fnum(ou["hl"], 1),
                       hurst=fnum(p["hurst"], 3), hedge_ratio=fnum(p["beta"], 3), kalman_beta=fnum(p["kalman_beta"], 3),
                       beta_drift=fnum(p["beta_stab"], 3), corr=fnum(p["corr"], 3), sigma_eq=fnum(ou["s_eq"], 4)),
            bt={k: fnum(v, 4) for k, v in p["bt"].items()}, levels=lv,
            series=dict(dates=dates, z=[fnum(v * sign, 3) for v in zser],
                        long=[fnum(v, 2) for v in (la / la.iloc[0] * 100)],
                        ref=[fnum(v, 2) for v in (lb / lb.iloc[0] * 100)])))

    # RESID candidates
    for r in resid:
        s = r["s"]
        if s > -S_WATCH:
            continue
        t = r["ticker"]
        ou = r["ou"]
        gap = ou["m"] - ou["X"][-1]
        lv = levels_from_gap(float(px[t].iloc[-1]), gap, ou["s_eq"], float(vol20[t]), ou["hl"])
        if lv["exp_pct"] is None or lv["exp_pct"] <= 0 or lv["exp_pct"] > MAX_EXP:
            continue
        tier = "SIGNAL" if (s <= -S_ENTRY and ou["hl"] >= RESID_HL_MIN) else "WATCH"
        strength = clip((abs(s) - 1.0) / 1.5, 0, 1)
        q = 0.5 * clip((ou["kappa"] - KAPPA_MIN) / 30, 0, 1) + 0.5
        bq = bt_quality(r["bt"])
        quality = q if bq is None else 0.5 * q + 0.5 * bq
        fl = flags_for(t)
        vz = value.get(t, {}).get("value_z")
        mp = macro_stock.get(t, {}).get("pctile")
        sc, comps = score_signal(strength, quality, vz, mp, lv["rr"], fl)
        n = len(r["path"])
        dates = [d.strftime("%Y-%m-%d") for d in px.index[-n:]]
        sec = meta_u.loc[t, "sector"]
        peers = [p for p in universe[universe.sector == sec].ticker if p != t]
        bask = np.exp(rets[peers].iloc[-n:].mean(axis=1).fillna(0).cumsum()) * 100
        stk = px_eur[t].iloc[-n:]
        candidates.append(dict(
            type="RESID", tier=tier, sector=sec, long_t=t, ref_t=None, score=sc, comps=comps, flags=fl,
            stats=dict(z=fnum(s, 3), kappa=fnum(ou["kappa"], 2), half_life=fnum(ou["hl"], 1),
                       sector_beta=fnum(ou["beta"], 3), sigma_eq=fnum(ou["s_eq"], 4)),
            bt={k: fnum(v, 4) for k, v in r["bt"].items()}, levels=lv,
            series=dict(dates=dates, z=[fnum(v, 3) for v in r["path"]],
                        long=[fnum(v, 2) for v in (stk / stk.iloc[0] * 100)],
                        ref=[fnum(v, 2) for v in bask.values])))

    # confirmations (number of independent signals on the same long name) + dedupe
    by_long = {}
    for c in candidates:
        by_long.setdefault(c["long_t"], []).append(c)
    ideas = []
    for t, cs in by_long.items():
        cs.sort(key=lambda c: (c["tier"] == "SIGNAL", c["score"]), reverse=True)
        best = cs[0]
        conf = [dict(type=c["type"], ref=c["ref_t"], z=c["stats"]["z"], tier=c["tier"]) for c in cs[1:]]
        n_sig = sum(1 for c in cs if c["tier"] == "SIGNAL")
        best["score"] = min(100.0, best["score"] + 3 * min(max(n_sig - 1, 0), 3))
        best["confirmations"] = conf
        ideas.append(best)
    ideas.sort(key=lambda c: c["score"], reverse=True)
    sig = [c for c in ideas if c["tier"] == "SIGNAL"][:N_SIGNALS]
    wat = [c for c in ideas if c["tier"] == "WATCH"][:N_WATCH]

    week = iso_week(as_of.date())
    out_ideas = []
    for rank, c in enumerate(sig + wat, 1):
        t = c["long_t"]
        idea = dict(
            id=f"{week}-{c['type']}-{t}" + (f"-{c['ref_t']}" if c["ref_t"] else ""),
            rank=rank, tier=c["tier"], type=c["type"], sector=c["sector"], score=fnum(c["score"], 1),
            components=c["comps"], flags=c["flags"], long=stock_card(t),
            ref=(dict(ticker=c["ref_t"], name=meta_u.loc[c["ref_t"], "name"], price=fnum(px[c["ref_t"]].iloc[-1], 4),
                      currency=ccy_of(c["ref_t"]), value=value.get(c["ref_t"]))
                 if c["ref_t"] else dict(ticker=None, name=f"Panier {c['sector']} (équipondéré, hors titre)")),
            stats=c["stats"], backtest=c["bt"], levels=c["levels"], value=value.get(t),
            macro=macro_stock.get(t), confirmations=c.get("confirmations", []), series=c["series"])
        out_ideas.append(idea)

    # universe table
    best_by = {i["long"]["ticker"]: i for i in out_ideas}
    resid_by = {r["ticker"]: r["s"] for r in resid}
    urows = []
    for t in universe.ticker:
        card = stock_card(t)
        v = value.get(t, {})
        m = macro_stock.get(t, {})
        b = best_by.get(t)
        urows.append(dict(t=t, n=card["name"], sec=card["sector"], ctry=card["country"], ccy=card["currency"],
                          px=card["price"], r1w=card["ret_1w"], r1m=card["ret_1m"], r3m=card["ret_3m"],
                          vol=card["vol_ann"], fpe=v.get("fpe"), ev=v.get("ev_ebitda"), pb=v.get("pb"),
                          vz=v.get("value_z"), s=fnum(resid_by.get(t), 2), mac=m.get("pctile"),
                          sig=(b["tier"] + ":" + b["type"]) if b else None, sc=b["score"] if b else None,
                          ed=card["earnings_date"]))

    # sector table with returns + valuation medians
    sec_rows = []
    for srow in macro_sectors:
        names = [t for t in universe[universe.sector == srow["sector"]].ticker]
        eq = rets[names].mean(axis=1)
        srow.update(ret_1w=fnum(math.expm1(eq.iloc[-5:].sum()), 4), ret_1m=fnum(math.expm1(eq.iloc[-21:].sum()), 4),
                    ret_3m=fnum(math.expm1(eq.iloc[-63:].sum()), 4),
                    n_signals=sum(1 for i in out_ideas if i["sector"] == srow["sector"] and i["tier"] == "SIGNAL"))
        sec_rows.append(srow)

    # factor tiles
    ftiles = []
    for key, lab, _, tr, unit in FACTORS:
        if key not in fac:
            continue
        s = fac[key].dropna()
        rg = regime.get(key, {})
        last = float(s.iloc[-1])
        if tr == "diff":
            c1, c4 = (last - s.iloc[-6]) * 100, (last - s.iloc[-22]) * 100
        else:
            c1, c4 = (last / s.iloc[-6] - 1) * 100, (last / s.iloc[-22] - 1) * 100
        pct1y = float((s.iloc[-252:] < last).mean() * 100)
        ftiles.append(dict(key=key, label=labels.get(key, lab), symbol=fac_sym.get(key), unit=unit,
                           last=fnum(last, 4), chg_1w=fnum(c1, 2), chg_4w=fnum(c4, 2),
                           chg_unit="pb" if tr == "diff" else "%", z4=fnum(rg.get("z4"), 2),
                           z13=fnum(rg.get("z13"), 2), pctile_1y=fnum(pct1y, 1),
                           spark=[fnum(v, 4) for v in s.iloc[-60:].values]))

    track_rows, track_summary = track_history(os.path.join(args.out, "history"), px, as_of)

    out = dict(
        meta=dict(version=VERSION, generated_at=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                  as_of=str(as_of.date()), week=week, synthetic=bool(args.synthetic),
                  universe_n=len(tickers), priced_n=len(keep),
                  fundamentals_n=sum(1 for t in keep if fund.get(t, {}).get("forwardPE") or fund.get(t, {}).get("priceToBook")),
                  pairs_tested=n_tested, pairs_cointegrated=n_coint,
                  pair_signals=sum(1 for c in candidates if c["type"] == "PAIR" and c["tier"] == "SIGNAL"),
                  resid_signals=sum(1 for c in candidates if c["type"] == "RESID" and c["tier"] == "SIGNAL"),
                  n_signals=len(sig), n_watch=len(wat), value_r2=value_r2,
                  runtime_s=round(time.time() - t0, 1), warnings=warnings_out,
                  params=dict(form_win=FORM_WIN, resid_win=RESID_WIN, coint_p=COINT_P, z_entry=Z_ENTRY,
                              s_entry=S_ENTRY, kappa_min=round(KAPPA_MIN, 2), hl=[HL_MIN, HL_MAX], min_corr=MIN_CORR)),
        macro=dict(factors=ftiles, regime_text=regime_text(regime, labels)),
        sectors=sec_rows, ideas=out_ideas, universe=urows,
        track=dict(summary=track_summary, rows=track_rows))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "latest.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    if not args.synthetic:
        hd = os.path.join(args.out, "history")
        os.makedirs(hd, exist_ok=True)
        slim = dict(week=week, as_of=str(as_of.date()),
                    ideas=[{k: v for k, v in i.items() if k != "series"} for i in out_ideas])
        with open(os.path.join(hd, f"{week}.json"), "w") as f:
            json.dump(slim, f, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps(out["meta"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
