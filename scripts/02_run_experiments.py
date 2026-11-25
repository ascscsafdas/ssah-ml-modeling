#!/usr/bin/env python3
"""
Step 3: Leakage-free evaluation of 11 models (4 physical/linear baselines,
6 core ML models, and 1 physics-residual hybrid model) across 4 validation regimes:
  V1: Repeated Nested 80:20 Split (10 seeds: 42..51) on F5 & F6_area
  V2: Leave-One-Date-Out CV (10 experimental days) on F5 & F6_area
  V3: Leave-One-Source-Out CV (5 sources) on F5, F6_area & F5_flux
  V4: Leave-One-Geometry-Out CV (LOGO_N & LOGO_H) on F5 & F6_area
Plus paired statistical tests and single-thread timing/memory benchmarks.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import pickle
import platform
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "ssah_dataset_clean.csv")
OUT_DIR = os.path.join(ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

CP_AIR = 1005.0
DEFAULT_AC = 1.93 * 0.93  # 1.7949 m^2

FEAT_SETS = {
    "F5": ["Tin_C", "H_m", "N_baffles", "m_dot_kg_s", "I_W_m2"],
    "F6_area": ["Tin_C", "H_m", "N_baffles", "m_dot_kg_s", "I_W_m2", "A_c_m2"],
    "F5_flux": ["Tin_C", "H_m", "N_baffles", "G_kg_m2_s", "I_W_m2"],
}


def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Custom Physical Baselines & Neural Estimators (strictly sklearn-compatible)
# ---------------------------------------------------------------------------

class PersistenceModel(BaseEstimator, RegressorMixin):
    def __init__(self, tin_idx=0):
        self.tin_idx = tin_idx

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X)[:, self.tin_idx].astype(float)


class LinearDeltaTModel(BaseEstimator, RegressorMixin):
    def __init__(self, tin_idx=0):
        self.tin_idx = tin_idx

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        dt = y - X[:, self.tin_idx]
        self.pipe_ = Pipeline([("scaler", MinMaxScaler()), ("lr", LinearRegression())])
        self.pipe_.fit(X, dt)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.tin_idx] + self.pipe_.predict(X)


def _extract_thermal_vars(X, feature_names):
    idx = {name: i for i, name in enumerate(feature_names)}
    tin = X[:, idx["Tin_C"]]
    h = X[:, idx["H_m"]]
    n_baf = X[:, idx["N_baffles"]]
    irrad = X[:, idx["I_W_m2"]]
    if "G_kg_m2_s" in idx:
        g_flux = X[:, idx["G_kg_m2_s"]]
        a_c = np.full(len(X), DEFAULT_AC, dtype=float)
        m_dot = g_flux * a_c
    else:
        m_dot = X[:, idx["m_dot_kg_s"]]
        a_c = X[:, idx["A_c_m2"]] if "A_c_m2" in idx else np.full(len(X), DEFAULT_AC, dtype=float)
    return tin, h, n_baf, m_dot, irrad, a_c


class SimpleEnergyBalanceModel(BaseEstimator, RegressorMixin):
    """Tout = Tin + eta_bar * I * A_c / (m_dot * cp)"""

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        tin, _, _, m_dot, irrad, a_c = _extract_thermal_vars(X, self.feature_names)
        q_solar_over_mcp = (irrad * a_c) / np.maximum(m_dot * CP_AIR, 1e-6)
        dt = y - tin
        denom = np.sum(q_solar_over_mcp ** 2)
        eta = float(np.sum(q_solar_over_mcp * dt) / denom) if denom > 0 else 0.5
        self.eta_bar_ = float(np.clip(eta, 0.05, 0.99))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        tin, _, _, m_dot, irrad, a_c = _extract_thermal_vars(X, self.feature_names)
        q_solar_over_mcp = (irrad * a_c) / np.maximum(m_dot * CP_AIR, 1e-6)
        return tin + self.eta_bar_ * q_solar_over_mcp


class HottelWhillierBlissModel(BaseEstimator, RegressorMixin):
    """
    Physically informed HWB collector baseline:
      Tout = Tin + (I * A_c / (m_dot * cp)) * F_R * tau_alpha - F_R * U_L * A_c * (Tin - T_ref) / (m_dot * cp)
    where F_R = (m_dot * cp / (A_c * U_L)) * (1 - exp(-A_c * U_L * F_prime / (m_dot * cp)))
    and F_prime = clip(a0 + a1 * (H - 0.04) + a2 * (N - 8), 0.15, 1.5)
    """

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def _forward(self, params, X):
        tau_alpha, u_l, a0, a1, a2, t_ref = params
        tin, h, n_baf, m_dot, irrad, a_c = _extract_thermal_vars(X, self.feature_names)
        mcp = np.maximum(m_dot * CP_AIR, 1e-4)
        f_prime = np.clip(a0 + a1 * (h - 0.04) + a2 * (n_baf - 8.0), 0.15, 1.4)
        ntu = np.clip(a_c * u_l * f_prime / mcp, 1e-5, 20.0)
        f_r = (mcp / np.maximum(a_c * u_l, 1e-5)) * (1.0 - np.exp(-ntu))
        dt = (irrad * a_c / mcp) * f_r * tau_alpha - (f_r * u_l * a_c / mcp) * (tin - t_ref)
        return tin + dt

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        t_ref0 = float(np.mean(X[:, self.feature_names.index("Tin_C")]))
        p0 = np.array([0.78, 8.0, 0.85, -1.0, 0.01, t_ref0], dtype=float)
        lb = np.array([0.30, 0.5, 0.25, -15.0, -0.25, -20.0], dtype=float)
        ub = np.array([0.98, 40.0, 1.25, 15.0, 0.25, 40.0], dtype=float)
        res = least_squares(
            lambda p: self._forward(p, X) - y,
            p0,
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=400,
        )
        self.params_ = res.x
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._forward(self.params_, X)


class _CNN1DNet(nn.Module):
    def __init__(self, in_len, channels=16, num_layers=2, kernel_size=2):
        super().__init__()
        convs = []
        in_ch = 1
        cur_len = in_len
        for _ in range(num_layers):
            pad = 1 if (kernel_size == 3 or cur_len < kernel_size) else 0
            convs.append(nn.Conv1d(in_ch, channels, kernel_size=kernel_size, padding=pad))
            convs.append(nn.ReLU())
            cur_len = cur_len + 2 * pad - kernel_size + 1
            in_ch = channels
        self.conv = nn.Sequential(*convs)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * cur_len, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)


class PyTorch1DCNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, channels=16, num_layers=2, kernel_size=2, lr=1.5e-2, weight_decay=1e-4, max_epochs=260, patience=6, seed=42):
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def fit(self, X, y):
        torch.set_num_threads(1)
        set_all_seeds(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.scaler_ = MinMaxScaler()
        Xs = self.scaler_.fit_transform(X).astype(np.float32)

        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(max(np.std(y), 1e-6))
        ys_norm = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(Xs, ys_norm, test_size=0.15, random_state=self.seed)
        X_tr_t = torch.from_numpy(X_tr)
        y_tr_t = torch.from_numpy(y_tr)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        self.model_ = _CNN1DNet(Xs.shape[1], self.channels, self.num_layers, self.kernel_size)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
        wait = 0
        for ep in range(self.max_epochs):
            self.model_.train()
            opt.zero_grad()
            loss = loss_fn(self.model_(X_tr_t), y_tr_t)
            loss.backward()
            opt.step()
            sched.step()

            if ep % 5 == 0 or ep == self.max_epochs - 1:
                self.model_.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(self.model_(X_val_t), y_val_t).item())
                if vloss < best_loss - 1e-5:
                    best_loss = vloss
                    best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        with torch.no_grad():
            pred_norm = self.model_(torch.from_numpy(Xs)).numpy().astype(float)
        return pred_norm * self.y_std_ + self.y_mean_


# ---------------------------------------------------------------------------
# Comprehensive Metrics & Statistical Comparison
# ---------------------------------------------------------------------------

def compute_all_metrics(y_true, y_pred, tin, m_dot, irrad, a_c):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    tin = np.asarray(tin, dtype=float)
    m_dot = np.asarray(m_dot, dtype=float)
    irrad = np.asarray(irrad, dtype=float)
    a_c = np.asarray(a_c, dtype=float)

    err = y_pred - y_true
    abs_err = np.abs(err)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.std(y_true) > 1e-9 else float("nan")
    max_ae = float(np.max(abs_err))
    p95_ae = float(np.percentile(abs_err, 95))

    # Relative deviations (C, K, and dT)
    rd_c = abs_err / np.maximum(np.abs(y_true), 1e-6) * 100.0
    rd_k = abs_err / (y_true + 273.15) * 100.0

    dt_true = y_true - tin
    dt_pred = y_pred - tin
    r2_dt = float(r2_score(dt_true, dt_pred)) if len(dt_true) > 1 and np.std(dt_true) > 1e-9 else float("nan")
    rd_dt = np.abs(dt_pred - dt_true) / np.maximum(np.abs(dt_true), 1e-6) * 100.0

    eta_true = m_dot * CP_AIR * dt_true / (irrad * a_c)
    eta_pred = m_dot * CP_AIR * dt_pred / (irrad * a_c)
    rmse_eta = float(np.sqrt(mean_squared_error(eta_true, eta_pred)))
    mae_eta = float(mean_absolute_error(eta_true, eta_pred))
    r2_eta = float(r2_score(eta_true, eta_pred)) if len(eta_true) > 1 and np.std(eta_true) > 1e-9 else float("nan")

    return {
        "n": int(len(y_true)),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
        "MaxAE": max_ae,
        "P95_AE": p95_ae,
        "MARD_C_pct": float(np.mean(rd_c)),
        "MARD_K_pct": float(np.mean(rd_k)),
        "N_gt10_C": int(np.sum(rd_c > 10.0)),
        "Pct_gt10_C": float(np.mean(rd_c > 10.0) * 100.0),
        "N_gt20_C": int(np.sum(rd_c > 20.0)),
        "Pct_gt20_C": float(np.mean(rd_c > 20.0) * 100.0),
        "R2_dT": r2_dt,
        "RMSE_dT": rmse,  # identical to RMSE in Tout since Tin cancels in (y_pred - Tin) - (y_true - Tin)
        "MAE_dT": mae,
        "MARD_dT_pct": float(np.mean(rd_dt)),
        "N_gt10_dT": int(np.sum(rd_dt > 10.0)),
        "Pct_gt10_dT": float(np.mean(rd_dt > 10.0) * 100.0),
        "N_gt20_dT": int(np.sum(rd_dt > 20.0)),
        "Pct_gt20_dT": float(np.mean(rd_dt > 20.0) * 100.0),
        "R2_eta": r2_eta,
        "RMSE_eta": rmse_eta,
        "MAE_eta": mae_eta,
    }


def paired_tests(y_true, pred_a, pred_b, n_boot=1000, seed=42):
    """Compare model A vs model B: delta = metric(A) - metric(B)."""
    y_true = np.asarray(y_true)
    ae_a = np.abs(np.asarray(pred_a) - y_true)
    ae_b = np.abs(np.asarray(pred_b) - y_true)
    se_a = (np.asarray(pred_a) - y_true) ** 2
    se_b = (np.asarray(pred_b) - y_true) ** 2

    diff_ae = ae_a - ae_b
    if np.allclose(diff_ae, 0):
        w_p, t_ae_p, t_se_p = 1.0, 1.0, 1.0
    else:
        _, w_p = stats.wilcoxon(ae_a, ae_b)
        _, t_ae_p = stats.ttest_rel(ae_a, ae_b)
        _, t_se_p = stats.ttest_rel(se_a, se_b)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    d_rmse, d_mae, d_r2 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        d_rmse.append(np.sqrt(np.mean((pa - yt) ** 2)) - np.sqrt(np.mean((pb - yt) ** 2)))
        d_mae.append(np.mean(np.abs(pa - yt)) - np.mean(np.abs(pb - yt)))
        if np.std(yt) > 1e-9:
            d_r2.append(r2_score(yt, pa) - r2_score(yt, pb))
    return {
        "wilcoxon_abs_err_p": float(w_p),
        "ttest_abs_err_p": float(t_ae_p),
        "ttest_sq_err_p": float(t_se_p),
        "delta_RMSE_mean": float(np.mean(d_rmse)),
        "delta_RMSE_95CI": [float(np.percentile(d_rmse, 2.5)), float(np.percentile(d_rmse, 97.5))],
        "delta_MAE_mean": float(np.mean(d_mae)),
        "delta_MAE_95CI": [float(np.percentile(d_mae, 2.5)), float(np.percentile(d_mae, 97.5))],
        "delta_R2_mean": float(np.mean(d_r2)) if d_r2 else float("nan"),
        "delta_R2_95CI": [float(np.percentile(d_r2, 2.5)), float(np.percentile(d_r2, 97.5))] if d_r2 else [float("nan"), float("nan")],
    }


# ---------------------------------------------------------------------------
# Model Tuning & Fitting Helper (Strictly inside training fold)
# ---------------------------------------------------------------------------

SEARCH_SPACES = {
    "Ridge": {"model__alpha": [float(x) for x in np.logspace(-4, 3, 29)]},
    "SVR": {
        "model__C": [1, 10, 50, 100, 300, 1000],
        "model__gamma": ["scale", 0.05, 0.1, 0.5, 1.0, 2.0],
        "model__epsilon": [0.05, 0.1, 0.5, 1.0],
    },
    "KNN": {
        "model__n_neighbors": [2, 3, 4, 5, 7, 9, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200, 350],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__reg_lambda": [1.0, 5.0],
    },
    "ANN": {
        "model__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
        "model__alpha": [1e-3, 1e-2, 1e-1],
        "model__solver": ["lbfgs"],
    },
    "1D-CNN": {
        "channels": [8, 16],
        "num_layers": [2],
        "kernel_size": [2, 3],
        "lr": [1.5e-2],
    },
}


def fit_and_predict_all(X_train, y_train, X_test, feature_names, seed=42, n_jobs=1):
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    tin_idx = feature_names.index("Tin_C")
    preds = {}
    best_params = {}
    timings = {}

    def _bench_predict(est, X_t):
        t0 = time.perf_counter()
        for _ in range(50):
            p = est.predict(X_t)
        dt = (time.perf_counter() - t0) / 50.0
        size_kb = len(pickle.dumps(est)) / 1024.0
        return p, dt * 1000.0, (dt / len(X_t)) * 1e6, size_kb

    # 1. Baselines
    for name, est in [
        ("Persistence", PersistenceModel(tin_idx=tin_idx)),
        ("Linear_dT", LinearDeltaTModel(tin_idx=tin_idx)),
        ("EB_Simple", SimpleEnergyBalanceModel(feature_names=feature_names)),
        ("EB_HWB", HottelWhillierBlissModel(feature_names=feature_names)),
    ]:
        t0 = time.perf_counter()
        est.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        p, b_ms, ps_us, sz_kb = _bench_predict(est, X_test)
        preds[name] = p
        best_params[name] = getattr(est, "params_", getattr(est, "eta_bar_", "fixed"))
        if isinstance(best_params[name], np.ndarray):
            best_params[name] = [round(float(v), 5) for v in best_params[name]]
        timings[name] = {
            "tune_time_s": 0.0,
            "fit_time_s": fit_s,
            "infer_batch_ms": b_ms,
            "infer_per_sample_us": ps_us,
            "model_size_kb": sz_kb,
        }

    # 2. Ridge
    pipe_ridge = Pipeline([("scaler", MinMaxScaler()), ("model", Ridge())])
    gs_ridge = GridSearchCV(pipe_ridge, SEARCH_SPACES["Ridge"], cv=inner_cv, scoring="r2", n_jobs=n_jobs, refit=False)
    t0 = time.perf_counter()
    gs_ridge.fit(X_train, y_train)
    tune_s = time.perf_counter() - t0
    est_ridge = Pipeline([("scaler", MinMaxScaler()), ("model", Ridge(alpha=gs_ridge.best_params_["model__alpha"]))])
    t1 = time.perf_counter()
    est_ridge.fit(X_train, y_train)
    fit_s = time.perf_counter() - t1
    p, b_ms, ps_us, sz_kb = _bench_predict(est_ridge, X_test)
    preds["Ridge"] = p
    best_params["Ridge"] = gs_ridge.best_params_
    timings["Ridge"] = {"tune_time_s": tune_s, "fit_time_s": fit_s, "infer_batch_ms": b_ms, "infer_per_sample_us": ps_us, "model_size_kb": sz_kb}

    # 3. SVR
    pipe_svr = Pipeline([("scaler", MinMaxScaler()), ("model", SVR(kernel="rbf"))])
    gs_svr = GridSearchCV(pipe_svr, SEARCH_SPACES["SVR"], cv=inner_cv, scoring="r2", n_jobs=n_jobs, refit=False)
    t0 = time.perf_counter()
    gs_svr.fit(X_train, y_train)
    tune_s = time.perf_counter() - t0
    bp = gs_svr.best_params_
    est_svr = Pipeline([("scaler", MinMaxScaler()), ("model", SVR(kernel="rbf", C=bp["model__C"], gamma=bp["model__gamma"], epsilon=bp["model__epsilon"]))])
    t1 = time.perf_counter()
    est_svr.fit(X_train, y_train)
    fit_s = time.perf_counter() - t1
    p, b_ms, ps_us, sz_kb = _bench_predict(est_svr, X_test)
    preds["SVR"] = p
    best_params["SVR"] = bp
    timings["SVR"] = {"tune_time_s": tune_s, "fit_time_s": fit_s, "infer_batch_ms": b_ms, "infer_per_sample_us": ps_us, "model_size_kb": sz_kb}

    # 4. KNN
    pipe_knn = Pipeline([("scaler", MinMaxScaler()), ("model", KNeighborsRegressor())])
    gs_knn = GridSearchCV(pipe_knn, SEARCH_SPACES["KNN"], cv=inner_cv, scoring="r2", n_jobs=n_jobs, refit=False)
    t0 = time.perf_counter()
    gs_knn.fit(X_train, y_train)
    tune_s = time.perf_counter() - t0
    bp = gs_knn.best_params_
    est_knn = Pipeline([("scaler", MinMaxScaler()), ("model", KNeighborsRegressor(n_neighbors=bp["model__n_neighbors"], weights=bp["model__weights"], p=bp["model__p"]))])
    t1 = time.perf_counter()
    est_knn.fit(X_train, y_train)
    fit_s = time.perf_counter() - t1
    p, b_ms, ps_us, sz_kb = _bench_predict(est_knn, X_test)
    preds["KNN"] = p
    best_params["KNN"] = bp
    timings["KNN"] = {"tune_time_s": tune_s, "fit_time_s": fit_s, "infer_batch_ms": b_ms, "infer_per_sample_us": ps_us, "model_size_kb": sz_kb}

    # 5. XGBoost
    pipe_xgb = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1, tree_method="exact")),
    ])
    rs_xgb = RandomizedSearchCV(
        pipe_xgb,
        SEARCH_SPACES["XGBoost"],
        n_iter=24,
        cv=inner_cv,
        scoring="r2",
        random_state=seed,
        n_jobs=n_jobs,
        refit=False,
    )
    t0 = time.perf_counter()
    rs_xgb.fit(X_train, y_train)
    tune_s = time.perf_counter() - t0
    bp = rs_xgb.best_params_
    est_xgb = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            tree_method="exact",
            n_estimators=bp["model__n_estimators"],
            max_depth=bp["model__max_depth"],
            learning_rate=bp["model__learning_rate"],
            subsample=bp["model__subsample"],
            colsample_bytree=bp["model__colsample_bytree"],
            reg_lambda=bp["model__reg_lambda"],
        )),
    ])
    t1 = time.perf_counter()
    est_xgb.fit(X_train, y_train)
    fit_s = time.perf_counter() - t1
    p, b_ms, ps_us, sz_kb = _bench_predict(est_xgb, X_test)
    preds["XGBoost"] = p
    best_params["XGBoost"] = bp
    timings["XGBoost"] = {"tune_time_s": tune_s, "fit_time_s": fit_s, "infer_batch_ms": b_ms, "infer_per_sample_us": ps_us, "model_size_kb": sz_kb}

    # 6. ANN (sklearn MLPRegressor with L-BFGS)
    pipe_ann = Pipeline([("scaler", MinMaxScaler()), ("model", MLPRegressor(activation="relu", max_iter=600, random_state=seed))])
    gs_ann = GridSearchCV(pipe_ann, SEARCH_SPACES["ANN"], cv=inner_cv, scoring="r2", n_jobs=n_jobs, refit=False)
    t0 = time.perf_counter()
    gs_ann.fit(X_train, y_train)
    tune_s = time.perf_counter() - t0
    bp = gs_ann.best_params_
    est_ann = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=bp["model__hidden_layer_sizes"],
            alpha=bp["model__alpha"],
            solver=bp["model__solver"],
            activation="relu",
            max_iter=600,
            random_state=seed,
        )),
    ])
    t1 = time.perf_counter()
    est_ann.fit(X_train, y_train)
    fit_s = time.perf_counter() - t1
    p, b_ms, ps_us, sz_kb = _bench_predict(est_ann, X_test)
    preds["ANN"] = p
    best_params["ANN"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in bp.items()}
    timings["ANN"] = {"tune_time_s": tune_s, "fit_time_s": fit_s, "infer_batch_ms": b_ms, "infer_per_sample_us": ps_us, "model_size_kb": sz_kb}

    # 7. 1D-CNN (target-standardized PyTorch Conv1D)
    gs_cnn = GridSearchCV(
        PyTorch1DCNNRegressor(max_epochs=260, patience=6, seed=seed),
        SEARCH_SPACES["1D-CNN"],
        cv=inner_cv,
        scoring="r2",
        n_jobs=n_jobs,
        refit=False,
    )
    t0 = time.perf_counter()
    gs_cnn.fit(X_train, y_train)
    tune_s = time.perf_counter() - t0
    bp = gs_cnn.best_params_
    est_cnn = PyTorch1DCNNRegressor(
        channels=bp["channels"], num_layers=bp["num_layers"], kernel_size=bp["kernel_size"], lr=bp["lr"], max_epochs=260, patience=6, seed=seed
    )
    t1 = time.perf_counter()
    est_cnn.fit(X_train, y_train)
    fit_s = time.perf_counter() - t1
    p, b_ms, ps_us, sz_kb = _bench_predict(est_cnn, X_test)
    preds["1D-CNN"] = p
    best_params["1D-CNN"] = bp
    timings["1D-CNN"] = {"tune_time_s": tune_s, "fit_time_s": fit_s, "infer_batch_ms": b_ms, "infer_per_sample_us": ps_us, "model_size_kb": sz_kb}

    # 8. Hybrid HWB + XGBoost residual
    t_h0 = time.perf_counter()
    hwb_base = HottelWhillierBlissModel(feature_names=feature_names)
    hwb_base.fit(X_train, y_train)
    res_train = y_train - hwb_base.predict(X_train)
    rs_res = RandomizedSearchCV(
        Pipeline([("scaler", MinMaxScaler()), ("model", XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1, tree_method="exact"))]),
        SEARCH_SPACES["XGBoost"],
        n_iter=20,
        cv=inner_cv,
        scoring="r2",
        random_state=seed,
        n_jobs=n_jobs,
        refit=False,
    )
    rs_res.fit(X_train, res_train)
    tune_s_hyb = time.perf_counter() - t_h0
    bp_hyb = rs_res.best_params_
    t_h1 = time.perf_counter()
    hwb_final = HottelWhillierBlissModel(feature_names=feature_names).fit(X_train, y_train)
    res_final = y_train - hwb_final.predict(X_train)
    xgb_res = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            tree_method="exact",
            n_estimators=bp_hyb["model__n_estimators"],
            max_depth=bp_hyb["model__max_depth"],
            learning_rate=bp_hyb["model__learning_rate"],
            subsample=bp_hyb["model__subsample"],
            colsample_bytree=bp_hyb["model__colsample_bytree"],
            reg_lambda=bp_hyb["model__reg_lambda"],
        )),
    ]).fit(X_train, res_final)
    fit_s_hyb = time.perf_counter() - t_h1

    t_p0 = time.perf_counter()
    for _ in range(50):
        p_hyb = hwb_final.predict(X_test) + xgb_res.predict(X_test)
    dt_hyb = (time.perf_counter() - t_p0) / 50.0
    sz_hyb = (len(pickle.dumps(hwb_final)) + len(pickle.dumps(xgb_res))) / 1024.0
    preds["Hybrid_HWB_XGB"] = p_hyb
    best_params["Hybrid_HWB_XGB"] = bp_hyb
    timings["Hybrid_HWB_XGB"] = {
        "tune_time_s": tune_s_hyb,
        "fit_time_s": fit_s_hyb,
        "infer_batch_ms": dt_hyb * 1000.0,
        "infer_per_sample_us": (dt_hyb / len(X_test)) * 1e6,
        "model_size_kb": sz_hyb,
    }

    return preds, best_params, timings


def aggregate_metrics(metric_list):
    keys = [k for k in metric_list[0].keys() if k != "n"]
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_list], dtype=float)
        agg[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }
    return agg


def run_v1_repeated(df, feat_key="F5", n_jobs=1):
    fnames = FEAT_SETS[feat_key]
    X = df[fnames].values
    y = df["Tout_C"].values
    seeds = list(range(42, 52))

    per_model_runs = {}
    per_model_timings = {}
    ref_preds = None
    ref_test_df = None
    ref_best_params = None
    pooled_y = []
    pooled_preds = {}

    for s in seeds:
        t_seed = time.perf_counter()
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=s)
        sub_te = df.iloc[idx_te]
        preds, best_params, timings = fit_and_predict_all(X[idx_tr], y[idx_tr], X[idx_te], fnames, seed=s, n_jobs=n_jobs)
        print(f"  [V1 {feat_key}] seed={s} done in {time.perf_counter()-t_seed:.1f}s", flush=True)
        pooled_y.append(y[idx_te])
        for mname, p in preds.items():
            met = compute_all_metrics(
                y[idx_te], p, sub_te["Tin_C"].values, sub_te["m_dot_kg_s"].values, sub_te["I_W_m2"].values, sub_te["A_c_m2"].values
            )
            per_model_runs.setdefault(mname, []).append(met)
            per_model_timings.setdefault(mname, []).append(timings[mname])
            pooled_preds.setdefault(mname, []).append(p)
        if s == 42:
            ref_preds = preds
            ref_test_df = sub_te.copy()
            ref_best_params = best_params

    summary = {m: aggregate_metrics(runs) for m, runs in per_model_runs.items()}
    timing_summary = {
        m: {
            k: {"mean": float(np.mean([t[k] for t in tlist])), "std": float(np.std([t[k] for t in tlist], ddof=1))}
            for k in tlist[0].keys()
        }
        for m, tlist in per_model_timings.items()
    }

    # Reference split metrics + paired statistical tests (seed=42 and pooled 10 repeats)
    ref_metrics = {
        m: compute_all_metrics(
            ref_test_df["Tout_C"].values, p, ref_test_df["Tin_C"].values, ref_test_df["m_dot_kg_s"].values, ref_test_df["I_W_m2"].values, ref_test_df["A_c_m2"].values
        )
        for m, p in ref_preds.items()
    }
    pooled_y_arr = np.concatenate(pooled_y)
    paired_ref = {}
    paired_pooled = {}
    for other in ["KNN", "SVR", "ANN", "1D-CNN", "Ridge", "EB_HWB", "Hybrid_HWB_XGB"]:
        paired_ref[f"XGBoost_vs_{other}"] = paired_tests(ref_test_df["Tout_C"].values, ref_preds["XGBoost"], ref_preds[other])
        paired_pooled[f"XGBoost_vs_{other}"] = paired_tests(pooled_y_arr, np.concatenate(pooled_preds["XGBoost"]), np.concatenate(pooled_preds[other]))

    # Save reference test predictions CSV for plotting and local sparsity slicing
    ref_out = ref_test_df[["sample_id", "source_id", "campaign_id", "Tin_C", "H_m", "N_baffles", "m_dot_kg_s", "I_W_m2", "A_c_m2", "Tout_C", "dT_C", "eta_abs"]].copy()
    for m, p in ref_preds.items():
        ref_out[f"pred_{m}"] = p
    ref_out.to_csv(os.path.join(OUT_DIR, f"v1_ref_predictions_{feat_key}.csv"), index=False)

    return {
        "feature_set": feat_key,
        "features": fnames,
        "n_train": 624,
        "n_test": 156,
        "seeds": seeds,
        "summary_10repeats": summary,
        "timing_10repeats": timing_summary,
        "ref_seed42_metrics": ref_metrics,
        "ref_seed42_best_params": ref_best_params,
        "paired_tests_seed42": paired_ref,
        "paired_tests_pooled10": paired_pooled,
    }


def run_group_cv(df, group_col, holdout_values, feat_key, tag, n_jobs=8):
    fnames = FEAT_SETS[feat_key]
    X = df[fnames].values
    y = df["Tout_C"].values

    oof_indices = []
    oof_preds = {}
    per_fold = {}

    for gval in holdout_values:
        mask_te = (df[group_col].values == gval)
        mask_tr = ~mask_te
        if np.sum(mask_te) == 0:
            continue
        preds, best_params, _ = fit_and_predict_all(X[mask_tr], y[mask_tr], X[mask_te], fnames, seed=42, n_jobs=n_jobs)
        print(f"  [{tag} {feat_key}] fold={gval} (n_test={int(np.sum(mask_te))}) done", flush=True)
        sub_te = df[mask_te]
        oof_indices.append(np.where(mask_te)[0])
        fold_res = {"n_test": int(np.sum(mask_te)), "n_train": int(np.sum(mask_tr)), "metrics": {}}
        for m, p in preds.items():
            oof_preds.setdefault(m, []).append(p)
            fold_res["metrics"][m] = compute_all_metrics(
                y[mask_te], p, sub_te["Tin_C"].values, sub_te["m_dot_kg_s"].values, sub_te["I_W_m2"].values, sub_te["A_c_m2"].values
            )
        per_fold[str(gval)] = fold_res

    all_idx = np.concatenate(oof_indices)
    sub_all = df.iloc[all_idx]
    oof_metrics = {}
    oof_df = sub_all[["sample_id", "source_id", "campaign_id", "Tin_C", "H_m", "N_baffles", "m_dot_kg_s", "I_W_m2", "A_c_m2", "Tout_C", "dT_C", "eta_abs"]].copy()
    for m, plist in oof_preds.items():
        p_all = np.concatenate(plist)
        oof_df[f"pred_{m}"] = p_all
        oof_metrics[m] = compute_all_metrics(
            sub_all["Tout_C"].values, p_all, sub_all["Tin_C"].values, sub_all["m_dot_kg_s"].values, sub_all["I_W_m2"].values, sub_all["A_c_m2"].values
        )
    oof_df.to_csv(os.path.join(OUT_DIR, f"{tag}_{feat_key}_oof.csv"), index=False)

    paired = {}
    if len(all_idx) >= 20:
        for other in ["KNN", "SVR", "ANN", "1D-CNN", "Ridge", "EB_HWB", "Hybrid_HWB_XGB"]:
            paired[f"XGBoost_vs_{other}"] = paired_tests(
                sub_all["Tout_C"].values, oof_df["pred_XGBoost"].values, oof_df[f"pred_{other}"].values
            )

    return {
        "tag": tag,
        "feature_set": feat_key,
        "per_fold": per_fold,
        "oof_pooled_metrics": oof_metrics,
        "paired_oof": paired,
    }


def main():
    t_start = time.time()
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded clean dataset: {len(df)} rows.")

    env_info = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "n_threads": 1,
    }
    import sklearn, scipy, xgboost, shap
    env_info.update({
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
    })

    results = {"environment": env_info, "search_spaces": SEARCH_SPACES}

    print("=== Running V1: Repeated Nested 80:20 Split (F5, single-thread timing) ===", flush=True)
    results["V1_F5"] = run_v1_repeated(df, "F5", n_jobs=1)
    print("=== Running V1: Repeated Nested 80:20 Split (F6_area, n_jobs=8) ===", flush=True)
    results["V1_F6_area"] = run_v1_repeated(df, "F6_area", n_jobs=8)

    exp_days = list(df[df["source_id"] == "Exp_Authors"]["campaign_id"].unique())
    print(f"=== Running V2: Leave-One-Date-Out CV ({len(exp_days)} days, F5 & F6_area) ===", flush=True)
    results["V2_LODO_F5"] = run_group_cv(df, "campaign_id", exp_days, "F5", "v2_lodo", n_jobs=8)
    results["V2_LODO_F6_area"] = run_group_cv(df, "campaign_id", exp_days, "F6_area", "v2_lodo", n_jobs=8)

    sources = ["Exp_Authors", "Jia2019", "Jia2021", "Amara2023", "BenAmara2024"]
    print("=== Running V3: Leave-One-Source-Out CV (F5, F6_area, F5_flux) ===", flush=True)
    results["V3_LOSO_F5"] = run_group_cv(df, "source_id", sources, "F5", "v3_loso", n_jobs=8)
    results["V3_LOSO_F6_area"] = run_group_cv(df, "source_id", sources, "F6_area", "v3_loso", n_jobs=8)
    results["V3_LOSO_F5_flux"] = run_group_cv(df, "source_id", sources, "F5_flux", "v3_loso", n_jobs=8)

    print("=== Running V4: Leave-One-Geometry-Out CV (LOGO_N & LOGO_H) ===", flush=True)
    results["V4_LOGO_N_F5"] = run_group_cv(df, "N_baffles", [1, 2, 4, 6], "F5", "v4_logo_n", n_jobs=8)
    results["V4_LOGO_N_F6_area"] = run_group_cv(df, "N_baffles", [1, 2, 4, 6], "F6_area", "v4_logo_n", n_jobs=8)
    results["V4_LOGO_H_F5"] = run_group_cv(df, "H_m", [0.042, 0.05, 0.07, 0.09, 0.11, 0.13], "F5", "v4_logo_h", n_jobs=8)
    results["V4_LOGO_H_F6_area"] = run_group_cv(df, "H_m", [0.042, 0.05, 0.07, 0.09, 0.11, 0.13], "F6_area", "v4_logo_h", n_jobs=8)

    results["total_elapsed_s"] = round(time.time() - t_start, 2)
    out_json = os.path.join(OUT_DIR, "experiment_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also emit concise CSV summary tables for quick inspection
    rows = []
    for m, stats_d in results["V1_F5"]["summary_10repeats"].items():
        t_d = results["V1_F5"]["timing_10repeats"][m]
        rows.append({
            "Model": m,
            "V1_R2_mean": round(stats_d["R2"]["mean"], 4),
            "V1_R2_std": round(stats_d["R2"]["std"], 4),
            "V1_RMSE_mean": round(stats_d["RMSE"]["mean"], 4),
            "V1_RMSE_std": round(stats_d["RMSE"]["std"], 4),
            "V1_MAE_mean": round(stats_d["MAE"]["mean"], 4),
            "V1_MARD_C_pct": round(stats_d["MARD_C_pct"]["mean"], 2),
            "V1_MARD_dT_pct": round(stats_d["MARD_dT_pct"]["mean"], 2),
            "LODO_R2": round(results["V2_LODO_F5"]["oof_pooled_metrics"][m]["R2"], 4),
            "LODO_RMSE": round(results["V2_LODO_F5"]["oof_pooled_metrics"][m]["RMSE"], 4),
            "LOSO_F5_RMSE": round(results["V3_LOSO_F5"]["oof_pooled_metrics"][m]["RMSE"], 4),
            "LOSO_F6_RMSE": round(results["V3_LOSO_F6_area"]["oof_pooled_metrics"][m]["RMSE"], 4),
            "LOSO_Flux_RMSE": round(results["V3_LOSO_F5_flux"]["oof_pooled_metrics"][m]["RMSE"], 4),
            "LOGO_N_RMSE": round(results["V4_LOGO_N_F5"]["oof_pooled_metrics"][m]["RMSE"], 4),
            "LOGO_H_F6_RMSE": round(results["V4_LOGO_H_F6_area"]["oof_pooled_metrics"][m]["RMSE"], 4),
            "Tune_s": round(t_d["tune_time_s"]["mean"], 3),
            "Fit_s": round(t_d["fit_time_s"]["mean"], 4),
            "Infer_us_per_sample": round(t_d["infer_per_sample_us"]["mean"], 2),
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "benchmark_summary.csv"), index=False)
    print(f"All experiments finished in {results['total_elapsed_s']} s. Saved {out_json}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
