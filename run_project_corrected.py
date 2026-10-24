"""
================================================================================
PORTFOLIO OPTIMISATION & DIVERSIFICATION  —  entry-level, runnable
================================================================================
A clean, classic Markowitz project — no advanced ML, no random-matrix theory.
What this does (in order):
  0. Download 15 US large-caps (2015-2024), build monthly returns
  1. Foundations: return, volatility, covariance, correlation
  2. Two-asset diversification demo (the "free lunch" of low correlation)
  3. Efficient frontier (Markowitz mean-variance)
  4. Minimum-variance + max-Sharpe (tangency) portfolios + Sharpe ratio
  5. Backtest 5 strategies with expanding-window historical estimates
  6. Metrics + charts

Run:  python run_project_corrected.py
Outputs:  prints a results table, saves charts to ./charts/
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config ---
TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL",   # tech
    "JPM","V",                             # financials
    "JNJ","UNH",                           # healthcare
    "PG","KO","WMT",                       # consumer staples
    "XOM",                                 # energy
    "CAT",                                 # industrials
    "DIS",                                 # communications
]
START, END = "2015-01-01", "2024-12-31"
RF_ANNUAL = 0.02                           # risk-free rate (annual), simplified
BACKTEST_START = 36                        # months of history before first rebalance
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

print("=" * 74)
print("STEP 0 — DATA")
print("=" * 74)

px = yf.download(TICKERS, start=START, end=END,
                 progress=False, auto_adjust=True)["Close"]
if px.empty or set(px.columns) != set(TICKERS):
    raise RuntimeError(
        "Real market-data download is incomplete. Check network, Yahoo Finance "
        "availability, and ticker coverage; no synthetic fallback is permitted."
    )
px = px.reindex(columns=TICKERS)
if px.isna().any().any():
    raise RuntimeError("Missing daily prices detected; inspect data before running the backtest.")
print(f"Downloaded {px.shape[1]} tickers, {px.shape[0]} daily rows.")

monthly_px = px.resample("ME").last()
rets = monthly_px.pct_change().dropna()
T, N = rets.shape
print(f"Monthly returns: {T} months x {N} assets  "
      f"({rets.index[0].date()} -> {rets.index[-1].date()})")

# ==============================================================================
print("\n" + "=" * 74)
print("STEP 1 — FOUNDATIONS (return, risk, covariance, correlation)")
print("=" * 74)

mu_ann  = rets.mean() * 12                    # annualised expected return
vol_ann = rets.std() * np.sqrt(12)            # annualised volatility
Sigma   = rets.cov() * 12                     # annualised covariance
Corr    = rets.corr()

print(f"Annualised vol range:  {vol_ann.min():.1%}  ->  {vol_ann.max():.1%}")
print(f"Annualised mean range: {mu_ann.min():.1%}  ->  {mu_ann.max():.1%}")

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(Corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(N)); ax.set_xticklabels(Corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(N)); ax.set_yticklabels(Corr.columns, fontsize=7)
ax.set_title("Correlation matrix (monthly returns)")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(f"{CHART_DIR}/01_correlation.png", dpi=120); plt.close(fig)

# ==============================================================================
print("\n" + "=" * 74)
print("STEP 2 — TWO-ASSET DIVERSIFICATION DEMO")
print("=" * 74)

# pick two assets with moderate correlation to show the diversification benefit
a, b = "XOM", "JNJ"
rho_ab = Corr.loc[a, b]
print(f"Chosen pair: {a} & {b}  (correlation = {rho_ab:.2f})")

ws = np.linspace(0, 1, 101)
sig_a, sig_b = vol_ann[a], vol_ann[b]
port_vol = np.sqrt(ws**2 * sig_a**2 + (1-ws)**2 * sig_b**2
                   + 2*ws*(1-ws)*rho_ab*sig_a*sig_b)
w_min = (sig_b**2 - rho_ab*sig_a*sig_b) / (sig_a**2 + sig_b**2 - 2*rho_ab*sig_a*sig_b)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ws, port_vol, "b-", lw=2, label="portfolio volatility")
ax.axvline(w_min, color="r", ls="--", label=f"min-vol weight = {w_min:.0%} {a}")
ax.set_xlabel(f"weight in {a}"); ax.set_ylabel("annualised volatility")
ax.set_title(f"Diversification: {a} & {b} (corr = {rho_ab:.2f})")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{CHART_DIR}/02_diversification.png", dpi=120); plt.close(fig)

print(f"  {a} alone: {sig_a:.1%} vol;  {b} alone: {sig_b:.1%} vol;  "
      f"min-vol mix: {port_vol.min():.1%} vol")
print("  -> combining imperfectly-correlated assets lowers risk below either alone")

# ==============================================================================
print("\n" + "=" * 74)
print("STEP 2b — MACHINE LEARNING: k-means clustering for diversification")
print("=" * 74)

# Unsupervised ML: let the data discover which assets move together, instead of
# assuming sector labels. Standardise each return series so clustering reflects
# *correlation* (the pattern), not magnitude.
Xz = rets.T.values                                  # (N assets, T months)
Xz = (Xz - Xz.mean(axis=1, keepdims=True)) / Xz.std(axis=1, keepdims=True)
N_CLUSTERS = 4
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit(Xz)
cluster = pd.Series(kmeans.labels_, index=rets.columns)

print("k-means clusters (unsupervised): assets grouped by return behaviour")
for c in sorted(cluster.unique()):
    print(f"  Cluster {c}: {list(cluster[cluster == c].index)}")

# 2D visualisation via PCA, coloured by cluster
pca = PCA(n_components=2, random_state=0).fit_transform(Xz)
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(pca[:, 0], pca[:, 1], c=cluster.values, cmap="tab10", s=90)
for i, t in enumerate(rets.columns):
    ax.annotate(t, (pca[i, 0], pca[i, 1]), fontsize=8)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Assets in 2D (PCA), coloured by k-means cluster")
fig.colorbar(sc, ax=ax, label="cluster")
fig.tight_layout(); fig.savefig(f"{CHART_DIR}/05_kmeans_clusters.png", dpi=120); plt.close(fig)

# ==============================================================================
print("\n" + "=" * 74)
print("STEP 3 — EFFICIENT FRONTIER (Markowitz)")
print("=" * 74)

def min_variance(Sigma):
    """Minimum-variance portfolio, long-only, fully invested (scipy SLSQP)."""
    n = Sigma.shape[0]
    res = minimize(lambda w: w @ Sigma.values @ w, np.ones(n)/n, method="SLSQP",
                   bounds=[(0, 1)]*n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return pd.Series(res.x, index=Sigma.columns) if res.success else None

def max_sharpe(mu, Sigma, rf=RF_ANNUAL):
    """Tangency (max-Sharpe) portfolio: minimise negative Sharpe ratio."""
    n = Sigma.shape[0]
    res = minimize(lambda w: -(mu.values @ w - rf) / np.sqrt(w @ Sigma.values @ w),
                   np.ones(n)/n, method="SLSQP", bounds=[(0, 1)]*n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return pd.Series(res.x, index=Sigma.columns) if res.success else None

w_gmv  = min_variance(Sigma)
w_tang = max_sharpe(mu_ann, Sigma)

def top(w, k=5):
    return w.sort_values(ascending=False).head(k).round(3)

print("Minimum-variance top holdings:\n", top(w_gmv))
print("Max-Sharpe top holdings:\n", top(w_tang))

frontier = []
for target in np.linspace(mu_ann.min(), mu_ann.max(), 40):
    n = Sigma.shape[0]
    res = minimize(lambda w: w @ Sigma.values @ w, np.ones(n)/n, method="SLSQP",
                   bounds=[(0, 1)]*n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                {"type": "eq", "fun": lambda w: mu_ann.values @ w - target}])
    if res.success:
        frontier.append((np.sqrt(res.x @ Sigma.values @ res.x), target))
frontier = np.array(frontier)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(vol_ann, mu_ann, s=20, c="gray", label="assets")
ax.plot(frontier[:, 0], frontier[:, 1], "b-", lw=2, label="minimum-variance boundary")
ax.scatter(np.sqrt(w_gmv @ Sigma @ w_gmv), mu_ann @ w_gmv, c="g", s=90, marker="*", label="min-variance")
ax.scatter(np.sqrt(w_tang @ Sigma @ w_tang), mu_ann @ w_tang, c="r", s=90, marker="*", label="max-Sharpe")
ax.set_xlabel("annualised volatility"); ax.set_ylabel("annualised return")
ax.set_title("Mean-variance boundary (upper branch is efficient)"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{CHART_DIR}/03_efficient_frontier.png", dpi=120); plt.close(fig)

# Sharpe ratios of the two special portfolios
sharpe_gmv  = (mu_ann @ w_gmv  - RF_ANNUAL) / np.sqrt(w_gmv  @ Sigma @ w_gmv)
sharpe_tang = (mu_ann @ w_tang - RF_ANNUAL) / np.sqrt(w_tang @ Sigma @ w_tang)
print(f"Sharpe ratio: min-variance = {sharpe_gmv:.2f},  max-Sharpe = {sharpe_tang:.2f}")

# ==============================================================================
print("\n" + "=" * 74)
print("STEP 4 — BACKTEST 5 STRATEGIES (out-of-sample)")
print("=" * 74)

def backtest(weight_fn):
    eq = [1.0]
    prev_w = None
    turnovers = []
    for t in range(BACKTEST_START, T - 1):
        hist = rets.iloc[:t+1]
        w = weight_fn(hist)
        if w is None or w.isna().any():
            w = pd.Series(1.0/N, index=rets.columns)
        if prev_w is not None:
            r_previous = rets.iloc[t]
            previous_return = float(prev_w @ r_previous)
            drifted = prev_w * (1 + r_previous) / (1 + previous_return)
            turnovers.append((w - drifted).abs().sum())

        r_next = rets.iloc[t+1]
        rp = float(w @ r_next)
        eq.append(eq[-1] * (1 + rp))
        prev_w = w
    return np.array(eq), (np.mean(turnovers) if turnovers else 0.0)

def w_equal(hist):   return pd.Series(1.0/N, index=rets.columns)
def w_gmv_fn(hist):  return min_variance(hist.cov() * 12)
def w_tang_fn(hist): return max_sharpe(hist.mean() * 12, hist.cov() * 12)
def w_invvol(hist):
    iv = 1.0 / (hist.std() * np.sqrt(12))
    return iv / iv.sum()

def w_cluster(hist):
    """Refit K-means using historical returns only."""
    X_hist = hist.T.values
    X_hist = (
        X_hist - X_hist.mean(axis=1, keepdims=True)
    ) / X_hist.std(axis=1, keepdims=True)

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=0,
        n_init=10
    )

    labels = model.fit_predict(X_hist)
    historical_clusters = pd.Series(labels, index=hist.columns)

    w = pd.Series(0.0, index=hist.columns)

    for c in historical_clusters.unique():
        members = historical_clusters[historical_clusters == c].index
        w[members] = 1.0 / (N_CLUSTERS * len(members))

    return w

strategies = {
    "Equal-weight (1/N)":       w_equal,
    "Minimum-variance":         w_gmv_fn,
    "Max-Sharpe (tangency)":    w_tang_fn,
    "Inverse-volatility":       w_invvol,
    "Cluster-diversified (k-means)": w_cluster,
}

results = {}
for name, fn in strategies.items():
    eq, turnover = backtest(fn)
    r = pd.Series(eq).pct_change().dropna()
    ann_ret = (1 + r.mean()) ** 12 - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol
    mdd = (pd.Series(eq) / pd.Series(eq).cummax() - 1).min()
    cagr = float(eq[-1] ** (12 / (len(eq) - 1)) - 1)
    results[name] = dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                          mdd=mdd, turnover=turnover, cagr=cagr, eq=eq)

print(f"\n{'strategy':24s} {'ann.ret':>8s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>7s} {'turnover':>9s}")
print("-" * 74)
for name, m in results.items():
    print(f"{name:24s} {m['ann_ret']:7.1%} {m['ann_vol']:6.1%} {m['sharpe']:7.2f} "
          f"{m['mdd']:7.1%} {m['turnover']:8.2f}")

fig, ax = plt.subplots(figsize=(9, 5.5))
for name, m in results.items():
    ax.plot(m["eq"], label=name, lw=1.6)
ax.set_title("Cumulative return — 5 strategies (out-of-sample, monthly rebalance)")
ax.set_xlabel("month"); ax.set_ylabel("growth of $1")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{CHART_DIR}/04_cumulative_returns.png", dpi=120); plt.close(fig)

res_df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "eq"}
                       for k, v in results.items()}).T
res_df.to_csv(f"{CHART_DIR}/results.csv")
os.makedirs("results", exist_ok=True)
res_df.to_csv("results/performance_metrics.csv")

print("\nDone. Charts saved to ./charts/  (results.csv has the full table).")
