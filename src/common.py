"""
common.py  --  Shared, TensorFlow-FREE helpers (numpy / pandas only)
======================================================================
Data preparation follows the ORIGINAL legacy pipeline exactly:
  - scaler fitted on ALL rows (legacy protocol)
  - split_data: 20% test / 10% of remainder validation

NOTHING in this file contains typed-in results.
Every number is computed from data / checkpoints / saved files.

Usage:
    Set environment variables before importing:
        P3_DATA   -- path to APSLDC_15min_Clean.xlsx
        P3_CKPT   -- directory containing the 9 .keras checkpoint files
        P3_WORK   -- output directory (results, cache, figures, tables)
        P3_QUICK  -- set to "1" for a fast smoke-test (2 epochs, seed 42 only)
        P3_SEEDS  -- comma-separated seeds, default "42,43,44"
"""

import os
import json
import math
import warnings
import zipfile
import copy

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ paths
DRIVE_DIR = os.environ.get("P3_DRIVE", os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE  = os.environ.get("P3_DATA",  os.path.join(os.path.dirname(__file__), "..", "APSLDC_15min_Clean.xlsx"))
CKPT_DIR   = os.environ.get("P3_CKPT",  os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
WORK_DIR   = os.environ.get("P3_WORK",  os.path.join(os.path.dirname(__file__), "..", "results"))

D_RES    = os.path.join(WORK_DIR, "results")
D_CACHE  = os.path.join(WORK_DIR, "cache")
D_FIG    = os.path.join(WORK_DIR, "figures")
D_TAB    = os.path.join(WORK_DIR, "tables")
D_STRICT = os.path.join(WORK_DIR, "strict_ckpt")

for _d in (D_RES, D_CACHE, D_FIG, D_TAB, D_STRICT):
    os.makedirs(_d, exist_ok=True)

QUICK = os.environ.get("P3_QUICK", "0") == "1"
SEEDS = [42] if QUICK else [int(s) for s in os.environ.get("P3_SEEDS", "42,43,44").split(",")]
SEED  = 42

# ------------------------------------------------------------------ experiment constants
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
Q, MED, LO, HI = 7, 3, 0, 6
ALPHA = 0.10

TEST_RATIO, VAL_RATIO = 0.20, 0.10

ARCHS = ["LSTM_Q", "CNN_BiGRU_Q", "DTAQ"]
CONDS = ["C1", "C2", "C3"]
LB_OF = {"C1": 1, "C2": 48, "C3": 48}

# Training protocol as stated in the manuscript
PATIENCE   = {"LSTM_Q": 5, "CNN_BiGRU_Q": 5, "DTAQ": 15}
MAX_EPOCHS = {"LSTM_Q": 30, "CNN_BiGRU_Q": 60, "DTAQ": 60}
if QUICK:
    MAX_EPOCHS = {k: 2 for k in MAX_EPOCHS}

# C3 feature subsets:
# 1. C3_CHECKPOINT_FEATURES (TOP10_LEGACY): The exact 10 features used to train the 9 original checkpoints
#    (matches C3_LSTM_Q.keras, C3_CNN_BiGRU_Q.keras, C3_DTAQ.keras with shape [None, 48, 10]).
TOP10_LEGACY = [
    "Actual_MW", "solarenergy", "solarradiation", "hour", "TEMP",
    "dew", "humidity", "feelslike", "uvindex", "is_weekend"
]
C3_CHECKPOINT_FEATURES = TOP10_LEGACY

# 2. MANUSCRIPT_TABLE3_FEATURES: Top-10 features reported in Table 3 of Manuscript_Revision_Pass1
#    (derived from the revised permutation-importance evaluation of DTAQ C2 on training windows).
MANUSCRIPT_TABLE3_FEATURES = [
    "Actual_MW", "hour", "solarenergy", "solarradiation", "uvindex",
    "windspeed", "feelslike", "TEMP", "sealevelpressure", "icon_enc"
]


def ckpt_path(cond, arch):
    return os.path.join(CKPT_DIR, f"{cond}_{arch}.keras")


# ------------------------------------------------------------------ data loading
def load_frame():
    """
    Load and encode the APSLDC dataset exactly like the legacy notebook.
    Returns (df, ALL_FEATS, apsldc_col).

    - Renames 'Actuals in MW' -> 'Actual_MW'
    - Encodes categorical weather columns (conditions, icon, preciptype)
    - Derives cyclic day-of-week features (dow_sin, dow_cos)
    - Derives is_holiday from holiday_status
    - Returns 26 numeric features (assertion checked)
    """
    df = pd.read_excel(DATA_FILE, parse_dates=["Datetime"])
    df = df.set_index("Datetime").sort_index()
    df = df.rename(columns={"Actuals in MW": "Actual_MW"})

    # Categorical encodings (must match the original training encoding)
    cond_enc = {
        "Clear": 0, "Partially cloudy": 1, "Overcast": 2,
        "Rain, Partially cloudy": 3, "Rain, Overcast": 4, "Rain": 5
    }
    icon_enc = {
        "clear-day": 0, "clear-night": 0, "partly-cloudy-day": 1,
        "partly-cloudy-night": 1, "cloudy": 2, "fog": 3, "wind": 3, "rain": 4
    }
    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    if "preciptype" in df.columns:
        df["precip_binary"] = df["preciptype"].apply(
            lambda x: 0.0 if str(x).strip().lower() in {"nan", "none", "", "no precip", "clear"}
            else 1.0
        )
    if "conditions" in df.columns:
        df["conditions_enc"] = df["conditions"].apply(
            lambda x: cond_enc.get(str(x).strip(), 2)
        ).astype(float)
    if "icon" in df.columns:
        df["icon_enc"] = df["icon"].apply(
            lambda x: icon_enc.get(str(x).strip().lower(), 2)
        ).astype(float)
    if "DAY NAME" in df.columns:
        dn = df["DAY NAME"].map(dow_map).fillna(0).astype(float)
        df["dow_sin"] = np.sin(2 * np.pi * dn / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dn / 7)
    if "holiday_status" in df.columns:
        df["is_holiday"] = df["holiday_status"].apply(
            lambda x: 1.0 if str(x).strip().lower() in {"holiday", "1", "yes", "true"} else 0.0
        )

    raw_cat = {"preciptype", "conditions", "icon", "DAY NAME", "holiday_status"}
    ap = next(
        (c for c in df.columns if any(k in c.lower() for k in ["demand", "forecast", "estm", "apsldc"])),
        None
    )
    excl  = {"Actual_MW"} | raw_cat | ({ap} if ap else set())
    num   = df.select_dtypes(include=np.number).columns.tolist()
    feats = ["Actual_MW"] + [c for c in num if c not in excl]
    assert len(feats) == 26, f"expected 26 features, got {len(feats)}: {feats}"
    return df, feats, ap


def feature_groups(feats):
    """Categorise features into groups."""
    cal   = [f for f in ["hour", "dow", "is_weekend", "dow_sin", "dow_cos", "is_holiday"] if f in feats]
    enc_w = [f for f in ["precip_binary", "conditions_enc", "icon_enc"] if f in feats]
    meteo = [f for f in feats if f not in cal + enc_w + ["Actual_MW"]]
    return dict(demand=["Actual_MW"], calendar=cal, weather_numeric=meteo, weather_encoded=enc_w)


# ------------------------------------------------------------------ windowing
def legacy_bounds(n):
    """Legacy split_data on n windows -> (train_end, val_end)."""
    te = int(n * TEST_RATIO)
    tv = n - te
    vl = int(tv * VAL_RATIO)
    return tv - vl, tv


def make_windows(vals, lb):
    """
    vals: (rows, feats) scaled array.
    Window i uses rows i..i+lb-1; target is row i+lb (feature 0).

    Returns:
        X        -- (n, lb, feats) float32
        y        -- (n,) float32
        tgt_rows -- (n,) int64 target row indices
    """
    n       = len(vals) - lb
    idx     = np.arange(n)[:, None] + np.arange(lb)[None, :]
    X       = vals[idx].astype(np.float32)
    tgt_rows = np.arange(lb, len(vals))
    y       = vals[tgt_rows, 0].astype(np.float32)
    assert X.shape[0] == len(y) == len(tgt_rows)
    assert (idx.max(axis=1) < tgt_rows).all(), \
        "LEAK: an input row is not strictly earlier than its target"
    return X, y, tgt_rows


class Prep:
    """Container for prepared windows for one feature-set / look-back / protocol."""
    pass


def prepare(df, feats, lb, protocol="legacy", bounds_ts=None):
    """
    Prepare sequence windows.

    protocol="legacy":
        Scaler fitted on ALL rows; partitions = legacy split_data ->
        train / cal (=validation) / test.
    protocol="strict":
        Scaler fitted on rows strictly before dev_start;
        partitions assigned by TARGET timestamp using
        bounds_ts = (dev_start, cal_start, test_start).
    """
    from sklearn.preprocessing import StandardScaler

    vals_raw = df[feats].values.astype(np.float64)
    ts_all   = df.index.values
    sc       = StandardScaler()

    if protocol == "legacy":
        sc.fit(vals_raw)
    else:
        sc.fit(vals_raw[ts_all < np.datetime64(bounds_ts[0])])

    vals = sc.transform(vals_raw).astype(np.float32)
    X, y, rows = make_windows(vals, lb)
    ts = ts_all[rows]

    p = Prep()
    p.X, p.y, p.ts, p.rows = X, y, ts, rows
    p.feats, p.lb           = list(feats), lb
    p.mean0, p.scale0       = float(sc.mean_[0]), float(sc.scale_[0])
    p.scaler                = sc
    n = len(y)

    if protocol == "legacy":
        a, b = legacy_bounds(n)
        p.idx = dict(train=np.arange(0, a), cal=np.arange(a, b), test=np.arange(b, n))
    else:
        dev_s, cal_s, test_s = [np.datetime64(t) for t in bounds_ts]
        p.idx = dict(
            train=np.where(ts < dev_s)[0],
            dev=np.where((ts >= dev_s) & (ts < cal_s))[0],
            cal=np.where((ts >= cal_s) & (ts < test_s))[0],
            test=np.where(ts >= test_s)[0],
        )
    return p


def inv(p, y_scaled):
    """Inverse-transform scaled target values to MW."""
    return np.asarray(y_scaled, dtype=np.float64) * p.scale0 + p.mean0


def strict_bounds(df, feats):
    """
    Strict partition boundaries derived from the legacy LB=48 windows:
    test and calibration = identical windows to legacy;
    dev = same number of windows just before them.
    """
    n = len(df) - 48
    a, b  = legacy_bounds(n)
    ts48  = df.index.values[48:]
    return ts48[a - (b - a)], ts48[a], ts48[b]   # dev_start, cal_start, test_start


def to_ns(a):
    return np.asarray(a).astype("datetime64[ns]")


# ------------------------------------------------------------------ metrics
def point_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    return dict(
        RMSE=float(np.sqrt(np.mean(e ** 2))),
        MAE=float(np.mean(np.abs(e))),
        MAPE=float(np.mean(np.abs(e) / np.maximum(np.abs(y), 1.0)) * 100),
        R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))
    )


def pinball(y, q, tau):
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def pinball_avg(y, yq):
    return float(np.mean([pinball(y, yq[:, i], t) for i, t in enumerate(QUANTILES)]))


def covered(y, lo, hi):
    return (y >= lo) & (y <= hi)


def picp(y, lo, hi):
    return float(np.mean(covered(y, lo, hi)))


def winkler(y, lo, hi, alpha=ALPHA):
    return float(np.mean(
        (hi - lo)
        + (2 / alpha) * np.maximum(lo - y, 0)
        + (2 / alpha) * np.maximum(y - hi, 0)
    ))


def prob_metrics(y, yq, rng=None):
    lo, hi = yq[:, LO], yq[:, HI]
    rng = (y.max() - y.min()) if rng is None else rng
    return dict(
        PICP=picp(y, lo, hi),
        PINAW=float(np.mean(hi - lo) / rng),
        Width_MW=float(np.mean(hi - lo)),
        Winkler90=winkler(y, lo, hi),
        Pinball7=pinball_avg(y, yq),
        CRPS_approx=2 * pinball_avg(y, yq),
        Cov50=picp(y, yq[:, 2], yq[:, 4]),
        Cov80=picp(y, yq[:, 1], yq[:, 5])
    )


# ------------------------------------------------------------------ conformal
def cqr_scores(y, yq):
    return np.maximum(yq[:, LO] - y, y - yq[:, HI])


def conformal_level(n, alpha=ALPHA):
    return min(math.ceil((1 - alpha) * (n + 1)) / n, 1.0)


def cqr_qhat(scores, alpha=ALPHA):
    """
    CQR quantile estimator: np.quantile (linear) at ceil((1-a)(n+1))/n.
    Signed negative q_hat is allowed and can shrink an over-covering interval.
    """
    return float(np.quantile(scores, conformal_level(len(scores), alpha)))


def shift_outer(yq, qhat):
    """Apply CQR adjustment: widen (or narrow if qhat<0) the outer quantiles."""
    out = yq.copy()
    out[:, LO] -= qhat
    out[:, HI] += qhat
    return out


def prospective_conformal(y, yq, init_scores, W=2880, alpha=ALPHA, gamma=None, delay=1):
    """
    Prospective split-conformal on a time-ordered stream.
    - score of forecast t is usable only for forecasts t+delay (late labels)
    - W: window of most recent scores
    - gamma given -> Adaptive Conformal Inference (Gibbs & Candes 2021)
    Returns (lo_adj, hi_adj, qhat_t, alpha_t).
    """
    n  = len(y)
    lo, hi = yq[:, LO], yq[:, HI]
    s  = np.maximum(lo - y, y - hi)
    buf = np.asarray(init_scores, float)[-W:].tolist()
    qh = np.empty(n)
    al = np.empty(n)
    a_t = alpha

    for t in range(n):
        r = t - delay
        if r >= 0:
            buf.append(s[r])
            if len(buf) > W:
                buf.pop(0)
            if gamma is not None:
                err = 0.0 if (lo[r] - qh[r] <= y[r] <= hi[r] + qh[r]) else 1.0
                a_t = a_t + gamma * (alpha - err)
        arr = np.asarray(buf)
        lvl = 1.0 - a_t
        if lvl >= 1.0:
            q = arr.max()
        elif lvl <= 0.0:
            q = arr.min()
        else:
            q = np.quantile(arr, min(math.ceil(lvl * (len(arr) + 1)) / len(arr), 1.0))
        qh[t], al[t] = q, a_t

    return lo - qh, hi + qh, qh, al


def mondrian_qhat(scores_cal, groups_cal, groups_test, alpha=ALPHA, min_n=50):
    glob  = cqr_qhat(scores_cal, alpha)
    table = {}
    for g in np.unique(groups_cal):
        sc = scores_cal[groups_cal == g]
        table[g] = cqr_qhat(sc, alpha) if len(sc) >= min_n else glob
    return np.array([table.get(g, glob) for g in groups_test]), table


def season_of(ts):
    m = pd.DatetimeIndex(ts).month
    return np.where(np.isin(m, [3, 4, 5, 6]), "Summer",
                    np.where(np.isin(m, [7, 8, 9]), "Monsoon", "Winter"))


def is_peak(ts):
    return np.isin(pd.DatetimeIndex(ts).hour, [18, 19, 20, 21])


# ------------------------------------------------------------------ statistics
def auto_lag(n):
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def nw_lrv(d, lag):
    d0 = d - d.mean()
    n  = len(d)
    v  = float(d0 @ d0) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * float(d0[k:] @ d0[:-k]) / n
    return v


def dm_hac(y, p1, p2, lag=None, loss="se"):
    """
    Diebold-Mariano test with Newey-West (Bartlett) HAC variance.
    d = L(model1) - L(model2): negative => model1 better.
    """
    from scipy import stats
    e1, e2 = y - p1, y - p2
    d = (e1 ** 2 - e2 ** 2) if loss == "se" else (np.abs(e1) - np.abs(e2))
    n   = len(d)
    lag = auto_lag(n) if lag is None else lag
    se  = math.sqrt(max(nw_lrv(d, lag), 1e-12) / n)
    stat = d.mean() / se
    return dict(
        lag=lag, DM=float(stat),
        p=float(2 * (1 - stats.norm.cdf(abs(stat)))),
        mean_diff=float(d.mean()),
        CI_lo=float(d.mean() - 1.96 * se),
        CI_hi=float(d.mean() + 1.96 * se),
        n=n
    )


def dm_iid(y, p1, p2):
    from scipy import stats
    d = (y - p1) ** 2 - (y - p2) ** 2
    n = len(d)
    t = d.mean() / math.sqrt(np.var(d, ddof=1) / n)
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))


def holm(pvals):
    p     = np.asarray(pvals, float)
    order = np.argsort(p)
    m     = len(p)
    adj   = np.empty(m)
    run   = 0.0
    for rank, i in enumerate(order):
        run   = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def block_idx(n, block, rng):
    nb     = math.ceil(n / block)
    starts = rng.integers(0, n, nb)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def block_boot_ci(fn, arrays, block=96, B=2000, seed=SEED, q=(2.5, 97.5)):
    rng  = np.random.default_rng(seed)
    n    = len(arrays[0])
    vals = np.empty(B)
    for b in range(B):
        ii      = block_idx(n, block, rng)
        vals[b] = fn(*[a[ii] for a in arrays])
    return float(np.percentile(vals, q[0])), float(np.percentile(vals, q[1]))


def acf(x, nlags=192):
    x   = np.asarray(x, float) - np.mean(x)
    den = float(x @ x)
    return np.array([float(x[k:] @ x[:len(x) - k]) / den for k in range(nlags + 1)])


# ------------------------------------------------------------------ saved predictions
def save_preds(path, **kw):
    np.savez_compressed(
        path,
        **{k: (to_ns(v) if k.startswith("ts") else np.asarray(v)) for k, v in kw.items()}
    )


def load_preds(tag):
    z = np.load(os.path.join(D_CACHE, f"{tag}.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def lagged_lookup(series_ts, series_val, query_ts, delta):
    """Value of the series at (query_ts - delta); NaN where that timestamp does not exist."""
    s = pd.Series(series_val, index=pd.DatetimeIndex(series_ts))
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(pd.DatetimeIndex(query_ts) - delta).values


# ------------------------------------------------------------------ figures / tables
def _font():
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for c in ["Times New Roman", "Liberation Serif", "DejaVu Serif"]:
        if c in have:
            return c
    return "serif"


def new_fig(width=3.5, height=2.7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [_font()], "font.size": 10,
        "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.grid": True,
        "grid.alpha": 0.3, "grid.linestyle": "--"
    })
    return plt.subplots(figsize=(width, height))


def save_fig(fig, name, source=None, caption=""):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(
        os.path.join(D_FIG, f"{name}.png"),
        dpi=600, bbox_inches="tight", edgecolor="none"
    )
    plt.close(fig)
    with open(os.path.join(D_FIG, f"{name}.json"), "w") as f:
        json.dump(
            dict(file=f"{name}.png", source=source, caption_suggestion=caption, dpi=600),
            f, indent=1
        )
    print("  figure ->", os.path.join(D_FIG, f"{name}.png"))


def _cell_border(cell, **kw):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    tcPr = cell._tc.get_or_add_tcPr()
    b    = OxmlElement("w:tcBorders")
    for edge in ("top", "bottom"):
        if edge in kw:
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(kw[edge]))
            e.set(qn("w:color"), "000000")
            b.append(e)
    tcPr.append(b)


def write_table(name, df, caption="", notes=""):
    """Save a results table as CSV + DOCX (Times New Roman 10 pt, horizontal rules only)."""
    from docx import Document
    from docx.shared import Pt
    from docx.oxml import OxmlElement

    df.to_csv(os.path.join(D_TAB, f"{name}.csv"), index=False)
    doc = Document()
    if caption:
        r = doc.add_paragraph().add_run(caption)
        r.font.size = Pt(10); r.font.name = "Times New Roman"; r.bold = True
    t = doc.add_table(rows=1, cols=len(df.columns))
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(c))
        run.font.size = Pt(10); run.font.name = "Times New Roman"; run.bold = True
        _cell_border(cell, top=12, bottom=6)
    for i, row in enumerate(df.itertuples(index=False)):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            txt = f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
            cells[j].text = ""
            run = cells[j].paragraphs[0].add_run(txt)
            run.font.size = Pt(10); run.font.name = "Times New Roman"
            if i == len(df) - 1:
                _cell_border(cells[j], bottom=12)
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    if notes:
        r = doc.add_paragraph().add_run(notes)
        r.font.size = Pt(10); r.font.name = "Times New Roman"
    doc.save(os.path.join(D_TAB, f"{name}.docx"))
    print("  table  ->", os.path.join(D_TAB, f"{name}.csv|docx"))


def need(path, hint):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. {hint}")
    return path


# ------------------------------------------------------------------ checkpoint config (no TensorFlow needed)
def read_cfg(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("config.json"))


def conv_audit(path):
    """
    Receptive field read straight from the checkpoint config:
    RF = 1 + sum (k-1)*d over causal convolutions.
    """
    rows = []
    for l in read_cfg(path)["config"].get("layers", []):
        if l["class_name"] in ("Conv1D", "Conv1d"):
            c = l["config"]
            k = c["kernel_size"][0] if isinstance(c["kernel_size"], (list, tuple)) else c["kernel_size"]
            d = c["dilation_rate"][0] if isinstance(c["dilation_rate"], (list, tuple)) else c["dilation_rate"]
            rows.append((c["name"], k, d, c["padding"]))
    return rows, 1 + sum((k - 1) * d for _, k, d, p in rows if p == "causal")
