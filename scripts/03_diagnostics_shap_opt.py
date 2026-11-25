#!/usr/bin/env python3
"""
Step 4: Deep Diagnostics, Physical Interpretability & Engineering Optimization:
  1. TreeSHAP + Permutation Importance across XGBoost/KNN/SVR/ANN
  2. Solar incidence angle (theta) exclusion & quantitative error analysis (n=693)
  3. Measurement uncertainty propagation (Kline-McClintock analytical + M=200 Monte Carlo)
  4. 1D-CNN exhaustive architecture (18 configs) & feature-order permutation ablation
  5. Sparse-region slicing, extreme-tail holdouts & out-of-range physical feasibility sweeps
  6. Demonstrative multi-parameter Pareto optimization with 100-bootstrap XGBoost 95% CIs
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import importlib.util
import json
import time
import warnings

import numpy as np
import pandas as pd
import shap
import torch
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "ssah_dataset_clean.csv")
OUT_DIR = os.path.join(ROOT, "results")
EXP_JSON = os.path.join(OUT_DIR, "experiment_results.json")

# Import reusable estimators from 02_run_experiments.py
spec = importlib.util.spec_from_file_location("exp_mod", os.path.join(ROOT, "scripts", "02_run_experiments.py"))
exp_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp_mod)

CP_AIR = 1005.0
DEFAULT_AC = 1.93 * 0.93


def build_xgb_from_bp(bp, seed=42):
    return XGBRegressor(
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
    )


def run_shap_and_importance(df, exp_res):
    print("  [1/6] Running SHAP & permutation importance...")
    f5 = exp_mod.FEAT_SETS["F5"]
    f6 = exp_mod.FEAT_SETS["F6_area"]
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=42)

    # F5 SHAP on reference split & 5-fold OOF
    bp_xgb5 = exp_res["V1_F5"]["ref_seed42_best_params"]["XGBoost"]
    scaler5 = MinMaxScaler().fit(df[f5].values[idx_tr])
    xgb5 = build_xgb_from_bp(bp_xgb5, seed=42).fit(scaler5.transform(df[f5].values[idx_tr]), df["Tout_C"].values[idx_tr])

    # Compute TreeSHAP in unscaled interpretable feature space by fitting equivalent tree on raw features
    # (scale-invariant for axis-aligned trees!)
    xgb5_raw = build_xgb_from_bp(bp_xgb5, seed=42).fit(df[f5].values[idx_tr], df["Tout_C"].values[idx_tr])
    explainer5 = shap.TreeExplainer(xgb5_raw)
    shap_te5 = explainer5.shap_values(df[f5].values[idx_te])

    # Full 5-fold OOF TreeSHAP across all 780 points
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_shap5 = np.zeros((len(df), len(f5)))
    for tr, te in kf.split(df):
        m = build_xgb_from_bp(bp_xgb5, seed=42).fit(df[f5].values[tr], df["Tout_C"].values[tr])
        oof_shap5[te] = shap.TreeExplainer(m).shap_values(df[f5].values[te])

    mean_abs_shap5 = {f: float(np.mean(np.abs(oof_shap5[:, i]))) for i, f in enumerate(f5)}
    pct_shap5 = {f: round(100.0 * v / sum(mean_abs_shap5.values()), 2) for f, v in mean_abs_shap5.items()}

    # F6_area SHAP
    bp_xgb6 = exp_res["V1_F6_area"]["ref_seed42_best_params"]["XGBoost"]
    oof_shap6 = np.zeros((len(df), len(f6)))
    for tr, te in kf.split(df):
        m = build_xgb_from_bp(bp_xgb6, seed=42).fit(df[f6].values[tr], df["Tout_C"].values[tr])
        oof_shap6[te] = shap.TreeExplainer(m).shap_values(df[f6].values[te])
    mean_abs_shap6 = {f: float(np.mean(np.abs(oof_shap6[:, i]))) for i, f in enumerate(f6)}
    pct_shap6 = {f: round(100.0 * v / sum(mean_abs_shap6.values()), 2) for f, v in mean_abs_shap6.items()}

    # Cross-model permutation importance on V1 reference test set (F5)
    bp_knn = exp_res["V1_F5"]["ref_seed42_best_params"]["KNN"]
    bp_svr = exp_res["V1_F5"]["ref_seed42_best_params"]["SVR"]
    bp_ann = exp_res["V1_F5"]["ref_seed42_best_params"]["ANN"]

    X_tr, X_te = df[f5].values[idx_tr], df[f5].values[idx_te]
    y_tr, y_te = df["Tout_C"].values[idx_tr], df["Tout_C"].values[idx_te]

    models_pi = {
        "XGBoost": Pipeline([("scaler", MinMaxScaler()), ("model", build_xgb_from_bp(bp_xgb5, 42))]).fit(X_tr, y_tr),
        "KNN": Pipeline([("scaler", MinMaxScaler()), ("model", KNeighborsRegressor(n_neighbors=bp_knn["model__n_neighbors"], weights=bp_knn["model__weights"], p=bp_knn["model__p"]))]).fit(X_tr, y_tr),
        "SVR": Pipeline([("scaler", MinMaxScaler()), ("model", SVR(kernel="rbf", C=bp_svr["model__C"], gamma=bp_svr["model__gamma"], epsilon=bp_svr["model__epsilon"]))]).fit(X_tr, y_tr),
        "ANN": Pipeline([("scaler", MinMaxScaler()), ("model", MLPRegressor(hidden_layer_sizes=tuple(bp_ann["model__hidden_layer_sizes"]), alpha=bp_ann["model__alpha"], solver=bp_ann["model__solver"], activation="relu", max_iter=600, random_state=42))]).fit(X_tr, y_tr),
    }
    perm_res = {}
    for mname, est in models_pi.items():
        pi = permutation_importance(est, X_te, y_te, n_repeats=20, random_state=42, scoring="neg_root_mean_squared_error")
        perm_res[mname] = {
            f: {"mean_delta_RMSE": float(pi.importances_mean[i]), "std": float(pi.importances_std[i])}
            for i, f in enumerate(f5)
        }

    np.savez(
        os.path.join(OUT_DIR, "shap_arrays.npz"),
        X_all_f5=df[f5].values,
        oof_shap5=oof_shap5,
        X_te_f5=X_te,
        shap_te5=shap_te5,
        X_all_f6=df[f6].values,
        oof_shap6=oof_shap6,
    )
    return {
        "F5_mean_abs_SHAP_oof": mean_abs_shap5,
        "F5_pct_SHAP_oof": pct_shap5,
        "F6_mean_abs_SHAP_oof": mean_abs_shap6,
        "F6_pct_SHAP_oof": pct_shap6,
        "permutation_importance_v1_ref": perm_res,
    }


def run_incidence_angle_study(df, exp_res):
    print("  [2/6] Running solar incidence angle exclusion & error analysis (n=693)...")
    sub = df[(df["is_experimental"] == 1) & (df["theta_deg"].notna())].copy().reset_index(drop=True)
    sub["cos_theta"] = np.cos(np.deg2rad(sub["theta_deg"]))
    sub["iam_term"] = 1.0 / np.maximum(sub["cos_theta"], 0.1) - 1.0

    feat_sets = {
        "No_theta": ["Tin_C", "m_dot_kg_s", "I_W_m2"],
        "With_theta": ["Tin_C", "m_dot_kg_s", "I_W_m2", "theta_deg"],
        "With_IAM": ["Tin_C", "m_dot_kg_s", "I_W_m2", "cos_theta", "iam_term"],
    }

    bp_xgb = exp_res["V1_F5"]["ref_seed42_best_params"]["XGBoost"]
    bp_knn = exp_res["V1_F5"]["ref_seed42_best_params"]["KNN"]
    bp_svr = exp_res["V1_F5"]["ref_seed42_best_params"]["SVR"]

    def make_est(mname):
        if mname == "XGBoost":
            return Pipeline([("s", MinMaxScaler()), ("m", build_xgb_from_bp(bp_xgb, 42))])
        if mname == "KNN":
            return Pipeline([("s", MinMaxScaler()), ("m", KNeighborsRegressor(n_neighbors=bp_knn["model__n_neighbors"], weights=bp_knn["model__weights"], p=bp_knn["model__p"]))])
        if mname == "SVR":
            return Pipeline([("s", MinMaxScaler()), ("m", SVR(kernel="rbf", C=bp_svr["model__C"], gamma=bp_svr["model__gamma"], epsilon=bp_svr["model__epsilon"]))])

    # 1. 10-repeat 80:20 CV
    random_res = {}
    for fk, cols in feat_sets.items():
        random_res[fk] = {}
        X = sub[cols].values
        y = sub["Tout_C"].values
        for mname in ["XGBoost", "KNN", "SVR"]:
            rmses, maes, r2s = [], [], []
            for s in range(42, 52):
                tr, te = train_test_split(np.arange(len(sub)), test_size=0.20, random_state=s)
                est = make_est(mname).fit(X[tr], y[tr])
                p = est.predict(X[te])
                rmses.append(np.sqrt(mean_squared_error(y[te], p)))
                maes.append(mean_absolute_error(y[te], p))
                r2s.append(r2_score(y[te], p))
            random_res[fk][mname] = {
                "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s, ddof=1)),
                "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses, ddof=1)),
                "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes, ddof=1)),
            }

    # 2. LODO-CV across 10 days
    days = sub["campaign_id"].unique()
    lodo_res = {}
    oof_preds = {}
    for fk, cols in feat_sets.items():
        lodo_res[fk] = {}
        oof_preds[fk] = {}
        X = sub[cols].values
        y = sub["Tout_C"].values
        for mname in ["XGBoost", "KNN", "SVR"]:
            p_all = np.zeros(len(sub))
            for d in days:
                te = (sub["campaign_id"].values == d)
                tr = ~te
                est = make_est(mname).fit(X[tr], y[tr])
                p_all[te] = est.predict(X[te])
            oof_preds[fk][mname] = p_all
            lodo_res[fk][mname] = {
                "R2": float(r2_score(y, p_all)),
                "RMSE": float(np.sqrt(mean_squared_error(y, p_all))),
                "MAE": float(mean_absolute_error(y, p_all)),
            }

    # Residual correlation & binned error vs theta for XGBoost LODO No_theta vs With_theta
    res_no = oof_preds["No_theta"]["XGBoost"] - sub["Tout_C"].values
    res_with = oof_preds["With_theta"]["XGBoost"] - sub["Tout_C"].values
    pearson_r, pearson_p = stats.pearsonr(sub["theta_deg"].values, res_no)
    spearman_r, spearman_p = stats.spearmanr(sub["theta_deg"].values, res_no)
    abs_pearson_r, abs_pearson_p = stats.pearsonr(sub["theta_deg"].values, np.abs(res_no))

    bins = [(0, 20), (20, 35), (35, 45), (45, 65)]
    binned = []
    for lo, hi in bins:
        m = (sub["theta_deg"].values >= lo) & (sub["theta_deg"].values < hi)
        binned.append({
            "bin_deg": f"[{lo}, {hi})",
            "n": int(np.sum(m)),
            "RMSE_no_theta": float(np.sqrt(np.mean(res_no[m] ** 2))),
            "MAE_no_theta": float(np.mean(np.abs(res_no[m]))),
            "Bias_no_theta": float(np.mean(res_no[m])),
            "RMSE_with_theta": float(np.sqrt(np.mean(res_with[m] ** 2))),
            "MAE_with_theta": float(np.mean(np.abs(res_with[m]))),
            "Bias_with_theta": float(np.mean(res_with[m])),
        })

    # Save array for Figure 9
    sub_out = sub[["sample_id", "campaign_id", "theta_deg", "I_W_m2", "Tout_C"]].copy()
    sub_out["res_xgb_no_theta"] = res_no
    sub_out["res_xgb_with_theta"] = res_with
    sub_out.to_csv(os.path.join(OUT_DIR, "incidence_angle_lodo_residuals.csv"), index=False)

    return {
        "n_aligned_exp": int(len(sub)),
        "theta_min": float(sub["theta_deg"].min()),
        "theta_max": float(sub["theta_deg"].max()),
        "random_80_20_10repeats": random_res,
        "lodo_10days": lodo_res,
        "lodo_paired_No_vs_With_theta_XGB": exp_mod.paired_tests(sub["Tout_C"].values, oof_preds["No_theta"]["XGBoost"], oof_preds["With_theta"]["XGBoost"]),
        "residual_vs_theta_pearson": [float(pearson_r), float(pearson_p)],
        "residual_vs_theta_spearman": [float(spearman_r), float(spearman_p)],
        "abs_residual_vs_theta_pearson": [float(abs_pearson_r), float(abs_pearson_p)],
        "binned_errors_lodo_xgb": binned,
    }


def run_uncertainty_analysis(df, exp_res):
    print("  [3/6] Running Kline-McClintock & Monte Carlo uncertainty propagation...")
    exp_df = df[df["is_experimental"] == 1].copy()

    # Analytical Kline-McClintock over all 694 experimental points
    scenarios = [
        {"name": "Nominal (u_T=0.5C, u_m=3%, u_I=2%)", "u_T": 0.5, "rel_m": 0.03, "rel_I": 0.02},
        {"name": "Conservative (u_T=1.0C, u_m=5%, u_I=2%)", "u_T": 1.0, "rel_m": 0.05, "rel_I": 0.02},
    ]
    km_res = {}
    for sc in scenarios:
        u_dT = np.sqrt(2.0) * sc["u_T"]
        rel_dT = u_dT / exp_df["dT_C"].values
        rel_eta = np.sqrt(sc["rel_m"] ** 2 + rel_dT ** 2 + sc["rel_I"] ** 2)
        abs_eta = rel_eta * exp_df["eta_abs"].values
        km_res[sc["name"]] = {
            "u_dT_C": round(float(u_dT), 4),
            "rel_u_dT_mean_pct": round(float(np.mean(rel_dT) * 100.0), 2),
            "rel_u_dT_max_pct": round(float(np.max(rel_dT) * 100.0), 2),
            "rel_u_eta_mean_pct": round(float(np.mean(rel_eta) * 100.0), 2),
            "rel_u_eta_median_pct": round(float(np.median(rel_eta) * 100.0), 2),
            "rel_u_eta_min_pct": round(float(np.min(rel_eta) * 100.0), 2),
            "rel_u_eta_max_pct": round(float(np.max(rel_eta) * 100.0), 2),
            "abs_u_eta_mean": round(float(np.mean(abs_eta)), 4),
        }

    # Monte Carlo propagation (M = 200) on V1 reference split (F5)
    f5 = exp_mod.FEAT_SETS["F5"]
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=42)
    X_tr, X_te = df[f5].values[idx_tr], df[f5].values[idx_te]
    y_tr, y_te = df["Tout_C"].values[idx_tr], df["Tout_C"].values[idx_te]
    is_exp_tr = df["is_experimental"].values[idx_tr] == 1
    is_exp_te = df["is_experimental"].values[idx_te] == 1

    bp_xgb = exp_res["V1_F5"]["ref_seed42_best_params"]["XGBoost"]
    bp_knn = exp_res["V1_F5"]["ref_seed42_best_params"]["KNN"]
    bp_svr = exp_res["V1_F5"]["ref_seed42_best_params"]["SVR"]
    bp_rid = exp_res["V1_F5"]["ref_seed42_best_params"]["Ridge"]
    bp_ann = exp_res["V1_F5"]["ref_seed42_best_params"]["ANN"]

    def fit_models(Xt, yt):
        return {
            "XGBoost": Pipeline([("s", MinMaxScaler()), ("m", build_xgb_from_bp(bp_xgb, 42))]).fit(Xt, yt),
            "KNN": Pipeline([("s", MinMaxScaler()), ("m", KNeighborsRegressor(n_neighbors=bp_knn["model__n_neighbors"], weights=bp_knn["model__weights"], p=bp_knn["model__p"]))]).fit(Xt, yt),
            "SVR": Pipeline([("s", MinMaxScaler()), ("m", SVR(kernel="rbf", C=bp_svr["model__C"], gamma=bp_svr["model__gamma"], epsilon=bp_svr["model__epsilon"]))]).fit(Xt, yt),
            "Ridge": Pipeline([("s", MinMaxScaler()), ("m", Ridge(alpha=bp_rid["model__alpha"]))]).fit(Xt, yt),
            "ANN": Pipeline([("s", MinMaxScaler()), ("m", MLPRegressor(hidden_layer_sizes=tuple(bp_ann["model__hidden_layer_sizes"]), alpha=bp_ann["model__alpha"], solver=bp_ann["model__solver"], activation="relu", max_iter=600, random_state=42))]).fit(Xt, yt),
            "EB_HWB": exp_mod.HottelWhillierBlissModel(feature_names=f5).fit(Xt, yt),
        }

    clean_models = fit_models(X_tr, y_tr)
    base_rmse = {m: float(np.sqrt(mean_squared_error(y_te, est.predict(X_te)))) for m, est in clean_models.items()}

    rng = np.random.default_rng(42)
    M_MC = 200
    u_T, rel_m, rel_I = 0.5, 0.03, 0.02

    # (a) Inference-time noise on experimental test inputs
    inf_preds = {m: np.zeros((M_MC, len(X_te))) for m in clean_models}
    inf_rmses = {m: [] for m in clean_models}
    for j in range(M_MC):
        Xn = X_te.copy()
        Xn[is_exp_te, 0] += rng.normal(0.0, u_T, size=np.sum(is_exp_te))
        Xn[is_exp_te, 3] *= (1.0 + rng.normal(0.0, rel_m, size=np.sum(is_exp_te)))
        Xn[is_exp_te, 4] *= (1.0 + rng.normal(0.0, rel_I, size=np.sum(is_exp_te)))
        for m, est in clean_models.items():
            p = est.predict(Xn)
            inf_preds[m][j] = p
            inf_rmses[m].append(np.sqrt(mean_squared_error(y_te, p)))

    # (b) End-to-end training + testing noise (M = 40 retrains for speed & precision)
    M_E2E = 40
    e2e_rmses = {m: [] for m in clean_models}
    for j in range(M_E2E):
        Xtr_n = X_tr.copy()
        ytr_n = y_tr.copy()
        Xtr_n[is_exp_tr, 0] += rng.normal(0.0, u_T, size=np.sum(is_exp_tr))
        Xtr_n[is_exp_tr, 3] *= (1.0 + rng.normal(0.0, rel_m, size=np.sum(is_exp_tr)))
        Xtr_n[is_exp_tr, 4] *= (1.0 + rng.normal(0.0, rel_I, size=np.sum(is_exp_tr)))
        ytr_n[is_exp_tr] += rng.normal(0.0, u_T, size=np.sum(is_exp_tr))

        Xte_n = X_te.copy()
        yte_n = y_te.copy()
        Xte_n[is_exp_te, 0] += rng.normal(0.0, u_T, size=np.sum(is_exp_te))
        Xte_n[is_exp_te, 3] *= (1.0 + rng.normal(0.0, rel_m, size=np.sum(is_exp_te)))
        Xte_n[is_exp_te, 4] *= (1.0 + rng.normal(0.0, rel_I, size=np.sum(is_exp_te)))
        yte_n[is_exp_te] += rng.normal(0.0, u_T, size=np.sum(is_exp_te))

        noisy_models = fit_models(Xtr_n, ytr_n)
        for m, est in noisy_models.items():
            e2e_rmses[m].append(np.sqrt(mean_squared_error(yte_n, est.predict(Xte_n))))

    mc_summary = {}
    for m in clean_models:
        per_sample_std = np.std(inf_preds[m], axis=0)
        mc_summary[m] = {
            "baseline_RMSE_C": round(base_rmse[m], 4),
            "inf_propagated_sigma_mean_C": round(float(np.mean(per_sample_std[is_exp_te])), 4),
            "inf_propagated_sigma_p95_C": round(float(np.percentile(per_sample_std[is_exp_te], 95)), 4),
            "inf_noisy_RMSE_mean_C": round(float(np.mean(inf_rmses[m])), 4),
            "inf_noisy_RMSE_std_C": round(float(np.std(inf_rmses[m], ddof=1)), 4),
            "e2e_noisy_RMSE_mean_C": round(float(np.mean(e2e_rmses[m])), 4),
            "e2e_noisy_RMSE_std_C": round(float(np.std(e2e_rmses[m], ddof=1)), 4),
            "e2e_RMSE_delta_C": round(float(np.mean(e2e_rmses[m]) - base_rmse[m]), 4),
        }

    return {"kline_mcclintock": km_res, "monte_carlo_propagation": mc_summary}


def run_cnn_ablation(df):
    print("  [4/6] Running 1D-CNN exhaustive architecture & feature-order ablation...")
    f5 = exp_mod.FEAT_SETS["F5"]
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=42)
    X_tr, X_te = df[f5].values[idx_tr], df[f5].values[idx_te]
    y_tr, y_te = df["Tout_C"].values[idx_tr], df["Tout_C"].values[idx_te]

    arch_results = []
    best_cfg = None
    best_rmse = 1e9
    for ch in [16, 32, 64]:
        for nl in [1, 2, 3]:
            for ks in [2, 3]:
                est = exp_mod.PyTorch1DCNNRegressor(channels=ch, num_layers=nl, kernel_size=ks, lr=1.5e-2, max_epochs=260, patience=6, seed=42)
                est.fit(X_tr, y_tr)
                p = est.predict(X_te)
                rmse = float(np.sqrt(mean_squared_error(y_te, p)))
                mae = float(mean_absolute_error(y_te, p))
                r2 = float(r2_score(y_te, p))
                row = {"channels": ch, "num_layers": nl, "kernel_size": ks, "R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}
                arch_results.append(row)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_cfg = (ch, nl, ks)

    # Feature-order permutation test with winning 1D-CNN config (across 5 seeds 42..46)
    orders = {
        "OrderA_Original [Tin,H,N,m,I]": ["Tin_C", "H_m", "N_baffles", "m_dot_kg_s", "I_W_m2"],
        "OrderB_OpThenGeom [I,Tin,m,H,N]": ["I_W_m2", "Tin_C", "m_dot_kg_s", "H_m", "N_baffles"],
        "OrderC_Interleaved [N,I,H,Tin,m]": ["N_baffles", "I_W_m2", "H_m", "Tin_C", "m_dot_kg_s"],
    }
    order_results = {}
    for oname, cols in orders.items():
        r2s, rmses, maes = [], [], []
        for s in range(42, 47):
            tr, te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=s)
            est = exp_mod.PyTorch1DCNNRegressor(channels=best_cfg[0], num_layers=best_cfg[1], kernel_size=best_cfg[2], lr=1.5e-2, max_epochs=260, patience=6, seed=s)
            est.fit(df[cols].values[tr], df["Tout_C"].values[tr])
            p = est.predict(df[cols].values[te])
            r2s.append(r2_score(df["Tout_C"].values[te], p))
            rmses.append(np.sqrt(mean_squared_error(df["Tout_C"].values[te], p)))
            maes.append(mean_absolute_error(df["Tout_C"].values[te], p))
        order_results[oname] = {
            "R2_mean": round(float(np.mean(r2s)), 4), "R2_std": round(float(np.std(r2s, ddof=1)), 4),
            "RMSE_mean": round(float(np.mean(rmses)), 4), "RMSE_std": round(float(np.std(rmses, ddof=1)), 4),
            "MAE_mean": round(float(np.mean(maes)), 4), "MAE_std": round(float(np.std(maes, ddof=1)), 4),
        }

    return {"architecture_grid_18": arch_results, "best_arch": best_cfg, "feature_order_ablation": order_results}


def run_sparsity_and_extrapolation(df, exp_res):
    print("  [5/6] Running sparse-region slicing, tail holdouts & out-of-range stress tests...")
    f5 = exp_mod.FEAT_SETS["F5"]
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=42)
    X_tr, X_te = df[f5].values[idx_tr], df[f5].values[idx_te]
    y_tr, y_te = df["Tout_C"].values[idx_tr], df["Tout_C"].values[idx_te]

    # 1. Local sparsity slicing on V1 reference test set (using normalized 5-NN distance to training set)
    sc = MinMaxScaler().fit(X_tr)
    nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(X_tr))
    dists, _ = nn.kneighbors(sc.transform(X_te))
    d5 = dists.mean(axis=1)
    thresh = np.percentile(d5, 85)
    sparse_mask = d5 >= thresh
    dense_mask = ~sparse_mask

    ref_df = pd.read_csv(os.path.join(OUT_DIR, "v1_ref_predictions_F5.csv"))
    sparsity_slice = {"n_dense": int(np.sum(dense_mask)), "n_sparse": int(np.sum(sparse_mask)), "d5_p85_thresh": round(float(thresh), 4)}
    for m in ["XGBoost", "KNN", "SVR", "ANN", "1D-CNN", "Ridge", "EB_HWB", "Hybrid_HWB_XGB"]:
        p = ref_df[f"pred_{m}"].values
        sparsity_slice[m] = {
            "dense_RMSE": round(float(np.sqrt(mean_squared_error(y_te[dense_mask], p[dense_mask]))), 4),
            "dense_MAE": round(float(mean_absolute_error(y_te[dense_mask], p[dense_mask])), 4),
            "sparse_RMSE": round(float(np.sqrt(mean_squared_error(y_te[sparse_mask], p[sparse_mask]))), 4),
            "sparse_MAE": round(float(mean_absolute_error(y_te[sparse_mask], p[sparse_mask])), 4),
            "sparse_MaxAE": round(float(np.max(np.abs(y_te[sparse_mask] - p[sparse_mask]))), 4),
        }

    # 2. Extreme-tail holdout tests (real held-out data beyond training parameter bounds)
    bp_xgb = exp_res["V1_F5"]["ref_seed42_best_params"]["XGBoost"]
    bp_knn = exp_res["V1_F5"]["ref_seed42_best_params"]["KNN"]
    bp_svr = exp_res["V1_F5"]["ref_seed42_best_params"]["SVR"]
    bp_ann = exp_res["V1_F5"]["ref_seed42_best_params"]["ANN"]

    tail_defs = {
        "Low_Irradiance_bottom15pct (I<=465)": df["I_W_m2"].values <= np.percentile(df["I_W_m2"].values, 15),
        "High_Irradiance_top15pct (I>=936)": df["I_W_m2"].values >= np.percentile(df["I_W_m2"].values, 85),
        "Low_Flow_bottom15pct (m<=0.014655)": df["m_dot_kg_s"].values <= np.percentile(df["m_dot_kg_s"].values, 15),
        "High_Flow_top15pct (m>=0.02499)": df["m_dot_kg_s"].values >= np.percentile(df["m_dot_kg_s"].values, 85),
        "Subzero_Tin (Tin<0C)": df["Tin_C"].values < 0.0,
        "Warm_Tin (Tin>20C)": df["Tin_C"].values > 20.0,
    }
    tail_results = {}
    for tname, te_mask in tail_defs.items():
        tr_mask = ~te_mask
        Xtr_t, ytr_t = df[f5].values[tr_mask], df["Tout_C"].values[tr_mask]
        Xte_t, yte_t = df[f5].values[te_mask], df["Tout_C"].values[te_mask]

        hwb = exp_mod.HottelWhillierBlissModel(feature_names=f5).fit(Xtr_t, ytr_t)
        res_tr = ytr_t - hwb.predict(Xtr_t)
        hyb_xgb = Pipeline([("s", MinMaxScaler()), ("m", build_xgb_from_bp(bp_xgb, 42))]).fit(Xtr_t, res_tr)

        ests = {
            "XGBoost": Pipeline([("s", MinMaxScaler()), ("m", build_xgb_from_bp(bp_xgb, 42))]).fit(Xtr_t, ytr_t),
            "KNN": Pipeline([("s", MinMaxScaler()), ("m", KNeighborsRegressor(n_neighbors=bp_knn["model__n_neighbors"], weights=bp_knn["model__weights"], p=bp_knn["model__p"]))]).fit(Xtr_t, ytr_t),
            "SVR": Pipeline([("s", MinMaxScaler()), ("m", SVR(kernel="rbf", C=bp_svr["model__C"], gamma=bp_svr["model__gamma"], epsilon=bp_svr["model__epsilon"]))]).fit(Xtr_t, ytr_t),
            "ANN": Pipeline([("s", MinMaxScaler()), ("m", MLPRegressor(hidden_layer_sizes=tuple(bp_ann["model__hidden_layer_sizes"]), alpha=bp_ann["model__alpha"], solver=bp_ann["model__solver"], activation="relu", max_iter=600, random_state=42))]).fit(Xtr_t, ytr_t),
            "EB_HWB": hwb,
        }
        tail_results[tname] = {"n_test": int(np.sum(te_mask)), "n_train": int(np.sum(tr_mask)), "models": {}}
        for m, est in ests.items():
            p = est.predict(Xte_t)
            tail_results[tname]["models"][m] = {
                "RMSE": round(float(np.sqrt(mean_squared_error(yte_t, p))), 4),
                "MAE": round(float(mean_absolute_error(yte_t, p)), 4),
                "Bias": round(float(np.mean(p - yte_t)), 4),
            }
        p_hyb = hwb.predict(Xte_t) + hyb_xgb.predict(Xte_t)
        tail_results[tname]["models"]["Hybrid_HWB_XGB"] = {
            "RMSE": round(float(np.sqrt(mean_squared_error(yte_t, p_hyb))), 4),
            "MAE": round(float(mean_absolute_error(yte_t, p_hyb)), 4),
            "Bias": round(float(np.mean(p_hyb - yte_t)), 4),
        }

    # 3. Synthetic out-of-range sweeps on full F5 dataset
    X_all, y_all = df[f5].values, df["Tout_C"].values
    hwb_full = exp_mod.HottelWhillierBlissModel(feature_names=f5).fit(X_all, y_all)
    hyb_full_xgb = Pipeline([("s", MinMaxScaler()), ("m", build_xgb_from_bp(bp_xgb, 42))]).fit(X_all, y_all - hwb_full.predict(X_all))
    full_models = {
        "XGBoost": Pipeline([("s", MinMaxScaler()), ("m", build_xgb_from_bp(bp_xgb, 42))]).fit(X_all, y_all),
        "KNN": Pipeline([("s", MinMaxScaler()), ("m", KNeighborsRegressor(n_neighbors=bp_knn["model__n_neighbors"], weights=bp_knn["model__weights"], p=bp_knn["model__p"]))]).fit(X_all, y_all),
        "SVR": Pipeline([("s", MinMaxScaler()), ("m", SVR(kernel="rbf", C=bp_svr["model__C"], gamma=bp_svr["model__gamma"], epsilon=bp_svr["model__epsilon"]))]).fit(X_all, y_all),
        "ANN": Pipeline([("s", MinMaxScaler()), ("m", MLPRegressor(hidden_layer_sizes=tuple(bp_ann["model__hidden_layer_sizes"]), alpha=bp_ann["model__alpha"], solver=bp_ann["model__solver"], activation="relu", max_iter=600, random_state=42))]).fit(X_all, y_all),
        "EB_HWB": hwb_full,
    }

    def eval_sweep(X_grid):
        out = {}
        for m, est in full_models.items():
            out[m] = est.predict(X_grid)
        out["Hybrid_HWB_XGB"] = hwb_full.predict(X_grid) + hyb_full_xgb.predict(X_grid)
        return out

    # Sweep I in [0, 1400] at Tin=10, H=0.04, N=8, m=0.025 (train I in [151.6, 1087])
    I_grid = np.linspace(0.0, 1400.0, 141)
    X_I = np.column_stack([np.full(141, 10.0), np.full(141, 0.04), np.full(141, 8.0), np.full(141, 0.025), I_grid])
    sw_I = eval_sweep(X_I)

    # Sweep m_dot in [0.004, 0.065] at Tin=10, H=0.04, N=8, I=800 (train m in [0.0077, 0.0487])
    m_grid = np.linspace(0.004, 0.065, 123)
    X_m = np.column_stack([np.full(123, 10.0), np.full(123, 0.04), np.full(123, 8.0), m_grid, np.full(123, 800.0)])
    sw_m = eval_sweep(X_m)

    # Sweep Tin in [-15, 45] at H=0.04, N=8, m=0.025, I=800 (train Tin in [-5.24, 29.54])
    T_grid = np.linspace(-15.0, 45.0, 121)
    X_T = np.column_stack([T_grid, np.full(121, 0.04), np.full(121, 8.0), np.full(121, 0.025), np.full(121, 800.0)])
    sw_T = eval_sweep(X_T)

    feasibility = {}
    for m in sw_I.keys():
        pI = sw_I[m]
        pm = sw_m[m]
        pT = sw_T[m]
        # Plateau outside training bounds: change in Tout from I=1087 to 1400, and from m=0.0487 to 0.065
        dI_hi = abs(float(pI[-1] - pI[np.argmin(np.abs(I_grid - 1090.0))]))
        dm_hi = abs(float(pm[-1] - pm[np.argmin(np.abs(m_grid - 0.049))]))
        feasibility[m] = {
            "zero_irrad_offset_C (Tout(I=0)-10)": round(float(pI[0] - 10.0), 3),
            "I_monotonicity_violations (dTout/dI<-0.01)": int(np.sum(np.diff(pI) < -0.01)),
            "m_monotonicity_violations (dTout/dm>0.01)": int(np.sum(np.diff(pm) > 0.01)),
            "Tin_monotonicity_violations (dTout/dTin<0)": int(np.sum(np.diff(pT) < 0.0)),
            "lower_bound_violations (Tout<Tin for I>0)": int(np.sum(pI[1:] < 10.0)),
            "high_I_plateau_delta (1090->1400 W/m2)": round(dI_hi, 3),
            "high_m_plateau_delta (0.049->0.065 kg/s)": round(dm_hi, 3),
            "Tout_at_I1400": round(float(pI[-1]), 2),
            "Tout_at_m0004": round(float(pm[0]), 2),
            "Tout_at_m0065": round(float(pm[-1]), 2),
            "Tout_at_Tin_minus15": round(float(pT[0]), 2),
            "Tout_at_Tin_45": round(float(pT[-1]), 2),
        }

    # Save sweeps for Figure 9
    np.savez(
        os.path.join(OUT_DIR, "extrapolation_sweeps.npz"),
        I_grid=I_grid, m_grid=m_grid, T_grid=T_grid,
        **{f"I_{k}": v for k, v in sw_I.items()},
        **{f"m_{k}": v for k, v in sw_m.items()},
        **{f"T_{k}": v for k, v in sw_T.items()},
    )

    return {
        "local_sparsity_slicing_v1_ref": sparsity_slice,
        "extreme_tail_holdouts": tail_results,
        "synthetic_out_of_range_feasibility": feasibility,
    }


def run_pareto_optimization(df, exp_res):
    print("  [6/6] Running demonstrative multi-parameter Pareto optimization...")
    # Train strictly on Jia-scale SSAH records (A_c = 1.7949 m^2, n=752) using F5
    # to ensure physically consistent collector area and valid in-domain support
    jia_df = df[df["source_id"].isin(["Exp_Authors", "Jia2019", "Jia2021"])].copy().reset_index(drop=True)
    f5 = exp_mod.FEAT_SETS["F5"]
    X = jia_df[f5].values
    y = jia_df["Tout_C"].values

    bp_xgb = exp_res["V1_F5"]["ref_seed42_best_params"]["XGBoost"]

    # Design search grid inside validated Jia-scale support:
    # H in [0.04, 0.05] (step 0.002 -> 6 values), N in {1, 2, 4, 6, 8}, m_dot in [0.012, 0.045] (67 steps)
    H_vals = np.array([0.040, 0.042, 0.044, 0.046, 0.048, 0.050])
    N_vals = np.array([1, 2, 4, 6, 8])
    m_vals = np.linspace(0.012, 0.045, 67)

    grid_H, grid_N, grid_m = np.meshgrid(H_vals, N_vals, m_vals, indexing="ij")
    flat_H = grid_H.ravel()
    flat_N = grid_N.ravel()
    flat_m = grid_m.ravel()
    n_grid = len(flat_H)

    # Nominal condition: I = 800 W/m^2, Tin = 10 C
    Tin_des, I_des = 10.0, 800.0
    X_cand = np.column_stack([
        np.full(n_grid, Tin_des),
        flat_H,
        flat_N,
        flat_m,
        np.full(n_grid, I_des),
    ])

    # 100-bootstrap ensemble for BOTH pure XGBoost and Hybrid_HWB_XGB
    n_boot = 100
    rng = np.random.default_rng(42)
    boot_xgb = np.zeros((n_boot, n_grid))
    boot_hyb = np.zeros((n_boot, n_grid))
    for b in range(n_boot):
        b_idx = rng.integers(0, len(jia_df), size=len(jia_df))
        Xb, yb = X[b_idx], y[b_idx]
        est = Pipeline([("s", MinMaxScaler()), ("m", build_xgb_from_bp(bp_xgb, seed=100 + b))]).fit(Xb, yb)
        boot_xgb[b] = est.predict(X_cand)

        hwb_b = exp_mod.HottelWhillierBlissModel(feature_names=f5).fit(Xb, yb)
        res_b = yb - hwb_b.predict(Xb)
        xgb_rb = Pipeline([("s", MinMaxScaler()), ("m", build_xgb_from_bp(bp_xgb, seed=200 + b))]).fit(Xb, res_b)
        boot_hyb[b] = hwb_b.predict(X_cand) + xgb_rb.predict(X_cand)

    tout_xgb_mean = np.mean(boot_xgb, axis=0)
    tout_xgb_lo = np.percentile(boot_xgb, 2.5, axis=0)
    tout_xgb_hi = np.percentile(boot_xgb, 97.5, axis=0)
    eta_xgb_mean = flat_m * CP_AIR * (tout_xgb_mean - Tin_des) / (I_des * DEFAULT_AC) * 100.0
    eta_xgb_lo = flat_m * CP_AIR * (tout_xgb_lo - Tin_des) / (I_des * DEFAULT_AC) * 100.0
    eta_xgb_hi = flat_m * CP_AIR * (tout_xgb_hi - Tin_des) / (I_des * DEFAULT_AC) * 100.0

    tout_hyb_mean = np.mean(boot_hyb, axis=0)
    tout_hyb_lo = np.percentile(boot_hyb, 2.5, axis=0)
    tout_hyb_hi = np.percentile(boot_hyb, 97.5, axis=0)
    eta_hyb_mean = flat_m * CP_AIR * (tout_hyb_mean - Tin_des) / (I_des * DEFAULT_AC) * 100.0
    eta_hyb_lo = flat_m * CP_AIR * (tout_hyb_lo - Tin_des) / (I_des * DEFAULT_AC) * 100.0
    eta_hyb_hi = flat_m * CP_AIR * (tout_hyb_hi - Tin_des) / (I_des * DEFAULT_AC) * 100.0

    is_supported = (
        ((np.isclose(flat_H, 0.04)) & (flat_N == 8) & (flat_m <= 0.040)) |
        ((np.isclose(flat_H, 0.05)) & (flat_m >= 0.014))
    )

    cand_df = pd.DataFrame({
        "H_m": flat_H,
        "N_baffles": flat_N.astype(int),
        "m_dot_kg_s": np.round(flat_m, 5),
        "is_supported": is_supported.astype(int),
        "Tout_xgb_mean_C": tout_xgb_mean,
        "Tout_xgb_ci95_lo": tout_xgb_lo,
        "Tout_xgb_ci95_hi": tout_xgb_hi,
        "eta_xgb_mean_pct": eta_xgb_mean,
        "eta_xgb_ci95_lo_pct": eta_xgb_lo,
        "eta_xgb_ci95_hi_pct": eta_xgb_hi,
        "Tout_hybrid_mean_C": tout_hyb_mean,
        "Tout_hybrid_ci95_lo": tout_hyb_lo,
        "Tout_hybrid_ci95_hi": tout_hyb_hi,
        "eta_hybrid_mean_pct": eta_hyb_mean,
        "eta_hybrid_ci95_lo_pct": eta_hyb_lo,
        "eta_hybrid_ci95_hi_pct": eta_hyb_hi,
    })

    # Extract physically consistent Pareto front on empirically supported manifold using Hybrid_HWB_XGB
    sup_df = cand_df[cand_df["is_supported"] == 1].copy().reset_index(drop=True)
    pareto_mask = np.ones(len(sup_df), dtype=bool)
    vals = sup_df[["Tout_hybrid_mean_C", "eta_hybrid_mean_pct"]].values
    for i in range(len(vals)):
        if np.any((vals[:, 0] >= vals[i, 0]) & (vals[:, 1] >= vals[i, 1]) & ((vals[:, 0] > vals[i, 0]) | (vals[:, 1] > vals[i, 1]))):
            pareto_mask[i] = False
    pareto_df = sup_df[pareto_mask].sort_values("Tout_hybrid_mean_C").reset_index(drop=True)

    # Identify baseline prototype (H=0.04, N=8, m_dot=0.025) and 3 illustrative optimal points
    base_row = sup_df[(sup_df["H_m"] == 0.04) & (sup_df["N_baffles"] == 8) & (np.isclose(sup_df["m_dot_kg_s"], 0.025, atol=3e-4))].iloc[0].to_dict()

    # Point A: Max efficiency with Tout >= 35 C
    sub_a = pareto_df[pareto_df["Tout_hybrid_mean_C"] >= 35.0]
    pt_a = sub_a.loc[sub_a["eta_hybrid_mean_pct"].idxmax()].to_dict()

    # Point B: Balanced space-heating point with Tout >= 45 C
    sub_b = pareto_df[pareto_df["Tout_hybrid_mean_C"] >= 45.0]
    pt_b = sub_b.loc[sub_b["eta_hybrid_mean_pct"].idxmax()].to_dict()

    # Point C: High-temperature drying point with Tout >= 60 C
    sub_c = pareto_df[pareto_df["Tout_hybrid_mean_C"] >= 60.0]
    pt_c = sub_c.loc[sub_c["eta_hybrid_mean_pct"].idxmax()].to_dict() if len(sub_c) > 0 else pareto_df.iloc[-1].to_dict()

    cand_df.to_csv(os.path.join(OUT_DIR, "pareto_candidates_I800_Tin10.csv"), index=False)
    pareto_df.to_csv(os.path.join(OUT_DIR, "pareto_front_I800_Tin10.csv"), index=False)

    return {
        "design_condition": {"I_W_m2": I_des, "Tin_C": Tin_des, "A_c_m2": DEFAULT_AC, "n_train_jia_scale": len(jia_df)},
        "n_candidates_total": int(n_grid),
        "n_candidates_supported": int(len(sup_df)),
        "n_pareto_points_supported_hybrid": int(len(pareto_df)),
        "unconstrained_pure_xgb_max_eta_pct": round(float(cand_df["eta_xgb_mean_pct"].max()), 2),
        "unconstrained_pure_xgb_n_eta_gt100": int((cand_df["eta_xgb_mean_pct"] > 100.0).sum()),
        "unconstrained_hybrid_max_eta_pct": round(float(cand_df["eta_hybrid_mean_pct"].max()), 2),
        "unconstrained_hybrid_n_eta_gt100": int((cand_df["eta_hybrid_mean_pct"] > 100.0).sum()),
        "baseline_prototype_H0.04_N8_m0.025": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in base_row.items()},
        "PointA_MaxEfficiency_Tout_ge35C": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in pt_a.items()},
        "PointB_BalancedHeating_Tout_ge45C": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in pt_b.items()},
        "PointC_HighTempDrying_Tout_ge60C": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in pt_c.items()},
    }


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH)
    with open(EXP_JSON, "r", encoding="utf-8") as f:
        exp_res = json.load(f)

    diag = {
        "shap_and_importance": run_shap_and_importance(df, exp_res),
        "incidence_angle": run_incidence_angle_study(df, exp_res),
        "uncertainty_analysis": run_uncertainty_analysis(df, exp_res),
        "cnn_ablation": run_cnn_ablation(df),
        "sparsity_and_extrapolation": run_sparsity_and_extrapolation(df, exp_res),
        "pareto_optimization": run_pareto_optimization(df, exp_res),
    }
    diag["elapsed_s"] = round(time.time() - t0, 2)
    out_path = os.path.join(OUT_DIR, "diagnostics_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    print(f"All diagnostics finished in {diag['elapsed_s']} s. Saved {out_path}")


if __name__ == "__main__":
    main()
