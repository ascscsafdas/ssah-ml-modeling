#!/usr/bin/env python3
"""
Step 5 (Restyled to sci-figure-style):
Generate all 8 composite figures (600-DPI PNG + PDF + SVG) in results/figures/
and standalone single panels in results/figures/panels/.
Strictly uses verified CSV/JSON/NPZ outputs; zero hard-coded fake metrics.
"""

import json
import os
import sys
import warnings

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogFormatterSciNotation, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_style import (
    apply_rc,
    sci_style,
    style_colorbar,
    boxed_legend,
    caption,
    save_both,
    SSAH_MODEL_STYLES,
    SEED_COLS,
    SHADE_TRAIN,
    SHADE_VAL,
    SHADE_TEST,
    SHADE_MINT,
)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "ssah_dataset_clean.csv")
RES_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RES_DIR, "figures")
PANEL_DIR = os.path.join(FIG_DIR, "panels")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)

apply_rc()


# ---------------------------------------------------------------------------
# Figure 3: 2x3 Parity Plots (+ standalone panels)
# ---------------------------------------------------------------------------
def _draw_fig3_panel(ax, y_true, y_pred, met, mkey, subcap, show_xlabel=True, show_ylabel=True):
    lim_lo, lim_hi = 0.0, 100.0
    x_line = np.linspace(lim_lo, lim_hi, 200)
    rd = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-6) * 100.0
    in10 = rd <= 10.0
    mid = (rd > 10.0) & (rd <= 20.0)
    gt20 = rd > 20.0
    st = SSAH_MODEL_STYLES[mkey]

    ax.fill_between(x_line, 0.9 * x_line, 1.1 * x_line, color=SHADE_TRAIN, alpha=0.95, zorder=0, label=r"$\pm 10\%$ band")
    ax.plot(x_line, x_line, color="#000000", ls="-", lw=1.4, zorder=2, label="1:1 line")
    ax.plot(x_line, 1.1 * x_line, color="#000000", ls="--", lw=0.9, zorder=2)
    ax.plot(x_line, 0.9 * x_line, color="#000000", ls="--", lw=0.9, zorder=2)
    ax.plot(x_line, 1.2 * x_line, color="#D62728", ls=":", lw=1.1, zorder=2, label=r"$\pm 20\%$ lines")
    ax.plot(x_line, 0.8 * x_line, color="#D62728", ls=":", lw=1.1, zorder=2)

    ax.scatter(y_true[in10], y_pred[in10], s=26, alpha=0.82, color=st["color"],
               edgecolors="#FFFFFF", linewidths=0.35, zorder=3, label=r"$\leq 10\%$ RD")
    if np.any(mid):
        ax.scatter(y_true[mid], y_pred[mid], s=36, alpha=0.92, color="#FF7F2A", marker="^",
                   edgecolors="#000000", linewidths=0.5, zorder=4, label=r"$10\text{--}20\%$ RD")
    if np.any(gt20):
        ax.scatter(y_true[gt20], y_pred[gt20], s=40, alpha=0.95, color="#D62728", marker="s",
                   edgecolors="#000000", linewidths=0.5, zorder=5, label=r"$> 20\%$ RD")

    txt = (
        f"$R^2 = {met['R2']:.4f}$\n"
        f"RMSE $= {met['RMSE']:.2f}$ °C\n"
        f"MAE $= {met['MAE']:.2f}$ °C\n"
        f"$>10\\%$: {met['N_gt10_C']}/{met['n']} | $>20\\%$: {met['N_gt20_C']}/{met['n']}"
    )
    ax.text(0.04, 0.95, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.1, alpha=1.0),
            zorder=6)

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    sci_style(ax)
    if show_xlabel:
        ax.set_xlabel("Measured outlet temperature (°C)")
    if show_ylabel:
        ax.set_ylabel("Predicted outlet temperature (°C)")
    if subcap:
        caption(ax, subcap, y=-0.20 if show_xlabel else -0.10)


def make_fig3(exp_res):
    df_ref = pd.read_csv(os.path.join(RES_DIR, "v1_ref_predictions_F5.csv"))
    y_true = df_ref["Tout_C"].values
    models = ["ANN", "1D-CNN", "KNN", "SVR", "Ridge", "XGBoost"]
    subcaps = ["(a) ANN (MLP)", "(b) 1D-CNN", "(c) KNN", "(d) SVR (RBF)", "(e) Ridge", "(f) XGBoost"]

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 9.2), sharex=True, sharey=True)
    for idx, (ax, m, sc) in enumerate(zip(axes.ravel(), models, subcaps)):
        r, c = divmod(idx, 3)
        _draw_fig3_panel(
            ax, y_true, df_ref[f"pred_{m}"].values,
            exp_res["V1_F5"]["ref_seed42_metrics"][m], m, sc,
            show_xlabel=(r == 1), show_ylabel=(c == 0)
        )

    leg_handles = [
        patches.Patch(facecolor=SHADE_TRAIN, edgecolor="#000000", linewidth=0.8, label=r"$\pm 10\%$ RD band"),
        Line2D([0], [0], color="#000000", lw=1.4, label="1:1 ideal line"),
        Line2D([0], [0], color="#D62728", ls=":", lw=1.2, label=r"$\pm 20\%$ lines"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4A90E2", markeredgecolor="#000000", markersize=6.2, label=r"$\leq 10\%$ RD"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#FF7F2A", markeredgecolor="#000000", markersize=6.5, label=r"$10\text{--}20\%$ RD"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#D62728", markeredgecolor="#000000", markersize=6.0, label=r"$> 20\%$ RD"),
    ]
    leg_labels = [h.get_label() for h in leg_handles]
    boxed_legend(None, loc="lower center", ncol=6, anchor=(0.5, 0.005), fontsize=9.0,
                 fig=fig, handles=leg_handles, labels=leg_labels)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.17, wspace=0.12, hspace=0.34)
    save_both(fig, "Fig3_parity_plots", FIG_DIR)

    # Standalone panels
    for idx, (m, sc) in enumerate(zip(models, subcaps)):
        fp, axp = plt.subplots(figsize=(5.4, 5.0))
        _draw_fig3_panel(
            axp, y_true, df_ref[f"pred_{m}"].values,
            exp_res["V1_F5"]["ref_seed42_metrics"][m], m, None,
            show_xlabel=True, show_ylabel=True
        )
        boxed_legend(axp, loc="lower right", ncol=1, fontsize=8.0, handles=leg_handles, labels=leg_labels)
        save_both(fp, f"Fig3{chr(97 + idx)}_{m}", PANEL_DIR)


# ---------------------------------------------------------------------------
# Figure 4: 1x3 Overall Model Performance Comparison (+ standalone panels)
# ---------------------------------------------------------------------------
def make_fig4(exp_res):
    order = ["Persistence", "Linear_dT", "EB_Simple", "EB_HWB", "Ridge", "1D-CNN", "ANN", "SVR", "XGBoost", "KNN", "Hybrid_HWB_XGB"]
    disp = [SSAH_MODEL_STYLES[m]["label"] for m in order]
    cols = [SSAH_MODEL_STYLES[m]["color"] for m in order]
    summ = exp_res["V1_F5"]["summary_10repeats"]

    r2_m = [summ[m]["R2"]["mean"] for m in order]
    r2_s = [summ[m]["R2"]["std"] for m in order]
    rmse_m = [summ[m]["RMSE"]["mean"] for m in order]
    rmse_s = [summ[m]["RMSE"]["std"] for m in order]
    mae_m = [summ[m]["MAE"]["mean"] for m in order]
    mae_s = [summ[m]["MAE"]["std"] for m in order]

    y_pos = np.arange(len(order))

    def _draw_panel(ax, vals, sds, mode, show_ylabel=True, subcap=None):
        if mode == "r2":
            plot_v = [max(v, 0.0) for v in vals]
            err_v = [0.0 if v < 0 else s for v, s in zip(vals, sds)]
            ax.barh(y_pos, plot_v, xerr=err_v, color=cols, edgecolor="#000000", linewidth=0.9,
                    capsize=3.0, error_kw=dict(ecolor="#000000", lw=1.1), height=0.66)
            ax.set_xlim(0.0, 1.24)
            ax.set_xlabel(r"Coefficient of determination, $R^2$ (–)")
            for i, (v, s) in enumerate(zip(vals, sds)):
                if v < 0:
                    ax.text(0.03, i, f"{v:.2f} $\\pm$ {s:.2f} (< 0)", va="center", ha="left", fontsize=8.3, fontweight="bold")
                else:
                    ax.text(v + s + 0.018, i, f"{v:.4f}", va="center", ha="left", fontsize=8.3)
        elif mode == "rmse":
            ax.barh(y_pos, vals, xerr=sds, color=cols, edgecolor="#000000", linewidth=0.9,
                    capsize=3.0, error_kw=dict(ecolor="#000000", lw=1.1), height=0.66)
            ax.set_xlim(0.0, max(vals) * 1.32)
            ax.set_xlabel("RMSE (°C)")
            for i, (v, s) in enumerate(zip(vals, sds)):
                ax.text(v + s + 0.6, i, f"{v:.2f} $\\pm$ {s:.2f}", va="center", ha="left", fontsize=8.3)
        else:
            ax.barh(y_pos, vals, xerr=sds, color=cols, edgecolor="#000000", linewidth=0.9,
                    capsize=3.0, error_kw=dict(ecolor="#000000", lw=1.1), height=0.66)
            ax.set_xlim(0.0, max(vals) * 1.32)
            ax.set_xlabel("MAE (°C)")
            for i, (v, s) in enumerate(zip(vals, sds)):
                ax.text(v + s + 0.6, i, f"{v:.2f} $\\pm$ {s:.2f}", va="center", ha="left", fontsize=8.3)

        ax.set_yticks(y_pos)
        if show_ylabel:
            ax.set_yticklabels(disp)
        sci_style(ax, minor_x=True, minor_y=False)
        ax.invert_yaxis()
        if subcap:
            caption(ax, subcap, y=-0.18)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.6), sharey=True)
    _draw_panel(axes[0], r2_m, r2_s, "r2", show_ylabel=True, subcap=r"(a) Test $R^2$ (10 repeats, mean $\pm$ SD)")
    _draw_panel(axes[1], rmse_m, rmse_s, "rmse", show_ylabel=False, subcap="(b) Test RMSE in outlet temperature")
    _draw_panel(axes[2], mae_m, mae_s, "mae", show_ylabel=False, subcap="(c) Test MAE in outlet temperature")
    fig.subplots_adjust(left=0.14, right=0.985, top=0.96, bottom=0.18, wspace=0.10)
    save_both(fig, "Fig4_model_comparison", FIG_DIR)

    for tag, vals, sds, mode in [("Fig4a_R2", r2_m, r2_s, "r2"), ("Fig4b_RMSE", rmse_m, rmse_s, "rmse"), ("Fig4c_MAE", mae_m, mae_s, "mae")]:
        fp, axp = plt.subplots(figsize=(5.8, 5.0))
        _draw_panel(axp, vals, sds, mode, show_ylabel=True, subcap=None)
        save_both(fp, tag, PANEL_DIR)


# ---------------------------------------------------------------------------
# Figure 5: 1x3 Single-Thread Timing Breakdown (+ standalone panels)
# ---------------------------------------------------------------------------
def make_fig5(exp_res):
    order = ["EB_HWB", "Ridge", "KNN", "SVR", "XGBoost", "Hybrid_HWB_XGB", "ANN", "1D-CNN"]
    disp = [SSAH_MODEL_STYLES[m]["label"] for m in order]
    cols = [SSAH_MODEL_STYLES[m]["color"] for m in order]
    t_res = exp_res["V1_F5"]["timing_10repeats"]

    tune_m = [t_res[m]["tune_time_s"]["mean"] for m in order]
    tune_s = [t_res[m]["tune_time_s"]["std"] for m in order]
    fit_m = [t_res[m]["fit_time_s"]["mean"] for m in order]
    fit_s = [t_res[m]["fit_time_s"]["std"] for m in order]
    inf_m = [t_res[m]["infer_per_sample_us"]["mean"] for m in order]
    inf_s = [t_res[m]["infer_per_sample_us"]["std"] for m in order]

    y_pos = np.arange(len(order))

    def _draw_t(ax, vals, sds, xlab, fmt, mult, pad, show_y=True, subcap=None):
        ax.barh(y_pos, vals, xerr=sds, color=cols, edgecolor="#000000", linewidth=0.9,
                capsize=3.0, error_kw=dict(ecolor="#000000", lw=1.1), height=0.64)
        ax.set_xlim(0.0, max(vals) * mult)
        ax.set_xlabel(xlab)
        for i, (v, s) in enumerate(zip(vals, sds)):
            ax.text(v + s + pad, i, fmt(v), va="center", ha="left", fontsize=8.4)
        ax.set_yticks(y_pos)
        if show_y:
            ax.set_yticklabels(disp)
        sci_style(ax, minor_x=True, minor_y=False)
        ax.invert_yaxis()
        if subcap:
            caption(ax, subcap, y=-0.20)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.9), sharey=True)
    _draw_t(axes[0], tune_m, tune_s, "Hyperparameter search time (s)", lambda v: f"{v:.2f} s", 1.28, 0.45, True, "(a) 5-fold inner CV search")
    _draw_t(axes[1], fit_m, fit_s, "Single final fit time (s)", lambda v: f"{v:.4f} s" if v < 0.05 else f"{v:.3f} s", 1.32, 0.018, False, "(b) Final fit (fixed hyperparams)")
    _draw_t(axes[2], inf_m, inf_s, r"Prediction latency ($\mu$s / sample)", lambda v: f"{v:.1f} $\\mu$s", 1.28, 0.16, False, "(c) Single-sample inference")
    fig.subplots_adjust(left=0.14, right=0.985, top=0.96, bottom=0.20, wspace=0.10)
    save_both(fig, "Fig5_timing_comparison", FIG_DIR)

    for tag, vals, sds, xlab, fmt, mult, pad in [
        ("Fig5a_tuning", tune_m, tune_s, "Hyperparameter search time (s)", lambda v: f"{v:.2f} s", 1.28, 0.45),
        ("Fig5b_fit", fit_m, fit_s, "Single final fit time (s)", lambda v: f"{v:.4f} s" if v < 0.05 else f"{v:.3f} s", 1.32, 0.018),
        ("Fig5c_inference", inf_m, inf_s, r"Prediction latency ($\mu$s / sample)", lambda v: f"{v:.1f} $\\mu$s", 1.28, 0.16),
    ]:
        fp, axp = plt.subplots(figsize=(5.6, 4.4))
        _draw_t(axp, vals, sds, xlab, fmt, mult, pad, True, None)
        save_both(fp, tag, PANEL_DIR)


# ---------------------------------------------------------------------------
# Figure 6: SCI-styled Leakage-Free Nested CV Flowchart
# ---------------------------------------------------------------------------
def make_fig6():
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#000000", lw=1.6, fs=9.5, bold=False, tc="#000000"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.0",
                                      facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=tc)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.7", lw=1.5, color="#000000"))

    box(2.5, 43.5, 22, 11.5,
        "Audited Dataset\n780 Unique Records\n(694 Exp + 86 Lit/CFD)\n1 Duplicate Removed",
        fc=SHADE_TRAIN, lw=1.6, fs=9.5, bold=True)
    arrow(25.2, 49.2, 30.8, 49.2)

    box(31.5, 41.0, 32, 16.5,
        "Four Outer Split Protocols\n"
        "• V1: 10× Repeated 80:20 (624 / 156)\n"
        "• V2: Leave-One-Date-Out (11 Exp Days)\n"
        "• V3: Leave-One-Source-Out (5 Studies)\n"
        "• V4: Leave-One-Geometry-Out ($N$ & $H$)",
        fc=SHADE_VAL, lw=1.6, fs=9.3, bold=True)

    arrow(64.2, 51.8, 70.3, 51.8)
    arrow(64.2, 44.5, 70.3, 29.0)

    box(71.0, 45.5, 26.5, 11.5,
        "Isolated Outer Test Set\n(Zero Scaler / Tuning Leakage)\nEvaluated Once per Outer Fold",
        fc=SHADE_TEST, lw=1.6, fs=9.5, bold=True)

    # Inner green container
    box(2.5, 17.5, 61.0, 20.0, "", fc=SHADE_MINT, lw=1.6)
    ax.text(33.0, 35.4, "Strictly Inside Outer Training Fold (No Test Information)",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#000000")
    box(5.0, 20.0, 25.0, 12.2,
        "Training-Only Preprocessing\n• Fit MinMaxScaler [0, 1]\n• Fit EB-HWB physics baseline\n• Compute training residuals",
        fc="#FFFFFF", lw=1.3, fs=9.0)
    arrow(30.7, 26.1, 35.3, 26.1)
    box(36.0, 20.0, 25.0, 12.2,
        "Inner 5-Fold CV Tuning\n• Maximize inner-fold $R^2$\n• Select winning hyperparameters\n• Refit on outer training fold",
        fc="#FFFFFF", lw=1.3, fs=9.0)

    arrow(47.5, 40.3, 33.0, 38.2)
    arrow(64.2, 26.1, 70.3, 26.1)

    box(71.0, 19.0, 26.5, 14.2,
        "Evaluated Models & Metrics\n"
        "• 4 Physical/Linear Baselines\n"
        "• 6 ML Models + Hybrid HWB+XGB\n"
        "• Metrics in $T_o$, $\\Delta T$, and $\\eta$\n"
        "• Paired Wilcoxon & Bootstrap CI",
        fc=SHADE_TRAIN, lw=1.6, fs=9.0)
    arrow(84.2, 44.8, 84.2, 33.9)

    box(2.5, 2.5, 95.0, 10.5,
        "Downstream Diagnostics & Engineering Deployment (Strictly on Training / OOF Predictions)\n"
        "TreeSHAP Attribution  |  Incidence-Angle ($\\theta$) Ablation  |  Monte Carlo Uncertainty  |  OOR Stress Tests  |  Pareto Optimization",
        fc="#F4F6F7", lw=1.6, fs=9.1)
    arrow(84.2, 18.3, 66.0, 13.7)

    save_both(fig, "Fig6_validation_workflow", FIG_DIR)


# ---------------------------------------------------------------------------
# Figure 7: 2x2 Provenance, Geometry Imbalance & Cross-Regime Generalization
# ---------------------------------------------------------------------------
def make_fig7(data_audit, exp_res):
    prov = data_audit["provenance_by_source"]
    src_names = ["Authors' Exp\n(11 days)", "Jia et al.\n(2019, CFD)", "Jia et al.\n(2021, CFD)",
                 "Amara & Bouabidi\n(2023, CFD)", "Ben Amara et al.\n(2024, Exp/CFD)"]
    # Align exact order with provenance_by_source: Exp_Authors, Jia2019, Jia2021, Amara2023, BenAmara2024
    counts = [p["n_samples"] for p in prov]
    pcts = [p["pct_total"] for p in prov]
    areas = [p["A_c_m2"] for p in prov]
    p_cols = ["#4A90E2", "#17A589", "#3CC21C", "#FF7F2A", "#D62728"]

    gcov = data_audit["geometry_coverage"]
    g_labels = [f"$H={g['H_m']:.3f}$ m, $N={g['N_baffles']}$" for g in gcov]
    g_counts = [g["count"] for g in gcov]

    regimes = [
        ("V1 Random\n(F5)", exp_res["V1_F5"]["summary_10repeats"], True),
        ("V2 LODO\n(11 Days)", exp_res["V2_LODO_F5"]["oof_pooled_metrics"], False),
        ("V3 LOSO\n(F5, no $A_c$)", exp_res["V3_LOSO_F5"]["oof_pooled_metrics"], False),
        ("V3 LOSO\n(F6, +$A_c$)", exp_res["V3_LOSO_F6_area"]["oof_pooled_metrics"], False),
        ("V4 LOGO-$N$\n(F5)", exp_res["V4_LOGO_N_F5"]["oof_pooled_metrics"], False),
        ("V4 LOGO-$H$\n(F6, +$A_c$)", exp_res["V4_LOGO_H_F6_area"]["oof_pooled_metrics"], False),
    ]
    comp_models = ["EB_HWB", "KNN", "SVR", "XGBoost", "Hybrid_HWB_XGB"]

    srcs = ["Exp_Authors", "Jia2019", "Jia2021", "Amara2023", "BenAmara2024"]
    src_disp = ["Exp_Authors\n($n=694$)", "Jia2019\n($n=28$)", "Jia2021\n($n=30$)", "Amara2023\n($n=24$)", "BenAmara2024\n($n=4$)"]

    def _p7a(ax, subcap=None):
        xp = np.arange(len(src_names))
        ax.bar(xp, counts, color=p_cols, edgecolor="#000000", linewidth=0.9, width=0.62)
        ax.set_yscale("log")
        ax.set_ylim(1, 4500)
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(minor_thresholds=(3, 0.5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(xp)
        ax.set_xticklabels(src_names, fontsize=8.8)
        ax.set_ylabel("Sample count (log scale)")
        for i, (c, pct, a) in enumerate(zip(counts, pcts, areas)):
            ax.text(i, c * 1.18, f"$n={c}$\n({pct:.1f}%)\n$A_c={a:.2f}\\ \\mathrm{{m}}^2$",
                    ha="center", va="bottom", fontsize=7.8)
        sci_style(ax, minor_x=False, minor_y=False, no_offset=False)
        if subcap:
            caption(ax, subcap, y=-0.24)

    def _p7b(ax, subcap=None):
        yg = np.arange(len(g_labels))
        bcols = ["#4A90E2" if i == 0 else "#FF7F2A" for i in range(len(g_labels))]
        ax.barh(yg, g_counts, color=bcols, edgecolor="#000000", linewidth=0.8, height=0.68)
        ax.set_xscale("log")
        ax.set_xlim(1, 5000)
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(minor_thresholds=(3, 0.5)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(yg)
        ax.set_yticklabels(g_labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Sample count (log scale)")
        for i, g in enumerate(gcov):
            ax.text(g["count"] * 1.15, i, f"{g['count']} ({g['pct']:.2f}%)", va="center", fontsize=8.0)
        sci_style(ax, minor_x=False, minor_y=False, no_offset=False)
        if subcap:
            caption(ax, subcap, y=-0.24)

    def _p7c(ax, subcap=None):
        x = np.arange(len(regimes))
        w = 0.155
        for mi, m in enumerate(comp_models):
            st = SSAH_MODEL_STYLES[m]
            vals = [r[1][m]["RMSE"]["mean"] if r[2] else r[1][m]["RMSE"] for r in regimes]
            ax.bar(x + (mi - 2) * w, vals, width=w, color=st["color"], edgecolor="#000000",
                   linewidth=0.7, label=st["label"])
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in regimes], fontsize=8.4)
        ax.set_ylim(0, 29.5)
        ax.set_ylabel("Pooled RMSE in $T_o$ (°C)")
        sci_style(ax, minor_x=False, minor_y=True)
        boxed_legend(ax, loc="upper left", ncol=2, fontsize=8.0)
        if subcap:
            caption(ax, subcap, y=-0.24)

    def _p7d(ax, subcap=None):
        f5_vals = [exp_res["V3_LOSO_F5"]["per_fold"][s]["metrics"]["XGBoost"]["RMSE"] for s in srcs]
        f6_vals = [exp_res["V3_LOSO_F6_area"]["per_fold"][s]["metrics"]["XGBoost"]["RMSE"] for s in srcs]
        fl_vals = [exp_res["V3_LOSO_F5_flux"]["per_fold"][s]["metrics"]["XGBoost"]["RMSE"] for s in srcs]
        hy_vals = [exp_res["V3_LOSO_F6_area"]["per_fold"][s]["metrics"]["Hybrid_HWB_XGB"]["RMSE"] for s in srcs]
        xs = np.arange(len(srcs))
        ws = 0.19
        ax.bar(xs - 1.5 * ws, f5_vals, width=ws, color="#FF7F2A", edgecolor="#000000", linewidth=0.7, label=r"XGBoost (F5, no $A_c$)")
        ax.bar(xs - 0.5 * ws, f6_vals, width=ws, color="#4A90E2", edgecolor="#000000", linewidth=0.7, label=r"XGBoost (F6, +$A_c$)")
        ax.bar(xs + 0.5 * ws, fl_vals, width=ws, color="#3CC21C", edgecolor="#000000", linewidth=0.7, label=r"XGBoost (flux $\dot{m}/A_c$)")
        ax.bar(xs + 1.5 * ws, hy_vals, width=ws, color="#D62728", edgecolor="#000000", linewidth=0.7, label="Hybrid (HWB+XGB, F6)")
        ax.set_xticks(xs)
        ax.set_xticklabels(src_disp, fontsize=8.5)
        ax.set_ylim(0, 31.5)
        ax.set_ylabel("Held-out source RMSE (°C)")
        sci_style(ax, minor_x=False, minor_y=True)
        boxed_legend(ax, loc="upper right", ncol=2, fontsize=7.8)
        if subcap:
            caption(ax, subcap, y=-0.24)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.6))
    _p7a(axes[0, 0], r"(a) Dataset provenance and collector area ($n = 780$)")
    _p7b(axes[0, 1], f"(b) Joint geometry imbalance (Evenness $= {data_audit['geometry_evenness_ratio']:.3f}$)")
    _p7c(axes[1, 0], "(c) Cross-regime generalization across splits")
    _p7d(axes[1, 1], r"(d) Leave-one-source-out: effect of $A_c$ harmonization")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.97, bottom=0.11, wspace=0.22, hspace=0.38)
    save_both(fig, "Fig7_generalization_and_data", FIG_DIR)

    for tag, fn in [("Fig7a_provenance", _p7a), ("Fig7b_imbalance", _p7b), ("Fig7c_regimes", _p7c), ("Fig7d_loso_area", _p7d)]:
        fp, axp = plt.subplots(figsize=(6.0, 4.6))
        fn(axp, None)
        save_both(fp, tag, PANEL_DIR)


# ---------------------------------------------------------------------------
# Figure 8: 2x2 TreeSHAP Global Attribution & Physics Dependence
# ---------------------------------------------------------------------------
def make_fig8(diag_res):
    npz = np.load(os.path.join(RES_DIR, "shap_arrays.npz"))
    X_all = npz["X_all_f5"]
    shap_all = npz["oof_shap5"]
    fnames = ["Tin_C", "H_m", "N_baffles", "m_dot_kg_s", "I_W_m2"]
    fdisp = {
        "I_W_m2": r"Solar irradiance, $I$ (W/m$^2$)",
        "Tin_C": r"Inlet temperature, $T_{\mathrm{in}}$ (°C)",
        "m_dot_kg_s": r"Mass flow rate, $\dot{m}$ (kg/s)",
        "N_baffles": r"Number of baffles, $N$ (–)",
        "H_m": r"Channel height, $H$ (m)",
    }
    mean_abs = diag_res["shap_and_importance"]["F5_mean_abs_SHAP_oof"]
    pct_shap = diag_res["shap_and_importance"]["F5_pct_SHAP_oof"]
    sorted_f = sorted(fnames, key=lambda k: mean_abs[k], reverse=True)
    yp = np.arange(len(sorted_f))

    def _p8a(ax, subcap=None):
        vals = [mean_abs[f] for f in sorted_f]
        ax.barh(yp, vals, color="#4A90E2", edgecolor="#000000", linewidth=0.9, height=0.58)
        ax.set_yticks(yp)
        ax.set_yticklabels([fdisp[f] for f in sorted_f], fontsize=9.2)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.38)
        ax.set_xlabel(r"Mean $|\mathrm{SHAP}|$ (°C)")
        for i, f in enumerate(sorted_f):
            ax.text(mean_abs[f] + 0.16, i, f"{mean_abs[f]:.2f} °C ({pct_shap[f]:.1f}%)", va="center", fontsize=8.8)
        sci_style(ax, minor_x=True, minor_y=False)
        if subcap:
            caption(ax, subcap, y=-0.22)

    def _p8b(ax, fig_obj, subcap=None):
        rng = np.random.default_rng(42)
        cmap = plt.cm.coolwarm
        for yi, f in enumerate(sorted_f):
            fi = fnames.index(f)
            sv = shap_all[:, fi]
            fv = X_all[:, fi]
            fv_norm = (fv - fv.min()) / max(fv.max() - fv.min(), 1e-9)
            jitter = rng.uniform(-0.22, 0.22, size=len(sv))
            sc = ax.scatter(sv, np.full(len(sv), yi) + jitter, c=fv_norm, cmap=cmap, s=12, alpha=0.80, edgecolors="none")
        ax.axvline(0.0, color="#000000", lw=1.0, ls="--")
        ax.set_yticks(yp)
        ax.set_yticklabels([fdisp[f] for f in sorted_f], fontsize=9.2)
        ax.invert_yaxis()
        ax.set_xlabel(r"SHAP value on $T_o$ (°C)")
        sci_style(ax, minor_x=True, minor_y=False)
        cbar = fig_obj.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        style_colorbar(cbar, "Normalized feature value", fontsize=9.5)
        if subcap:
            caption(ax, subcap, y=-0.22)

    def _p8c(ax, fig_obj, subcap=None):
        sc1 = ax.scatter(X_all[:, 4], shap_all[:, 4], c=X_all[:, 3], cmap="viridis", s=18, alpha=0.82, edgecolors="none")
        ax.axhline(0.0, color="#000000", lw=1.0, ls="--")
        ax.set_xlabel(r"Solar irradiance, $I$ (W/m$^2$)")
        ax.set_ylabel(r"SHAP value for $I$ (°C)")
        sci_style(ax)
        cb1 = fig_obj.colorbar(sc1, ax=ax, fraction=0.046, pad=0.03)
        style_colorbar(cb1, r"Mass flow rate $\dot{m}$ (kg/s)", fontsize=9.5)
        if subcap:
            caption(ax, subcap, y=-0.22)

    def _p8d(ax, fig_obj, subcap=None):
        sc2 = ax.scatter(X_all[:, 3], shap_all[:, 3], c=X_all[:, 4], cmap="plasma", s=18, alpha=0.82, edgecolors="none")
        ax.axhline(0.0, color="#000000", lw=1.0, ls="--")
        ax.set_xlabel(r"Mass flow rate, $\dot{m}$ (kg/s)")
        ax.set_ylabel(r"SHAP value for $\dot{m}$ (°C)")
        sci_style(ax)
        cb2 = fig_obj.colorbar(sc2, ax=ax, fraction=0.046, pad=0.03)
        style_colorbar(cb2, r"Solar irradiance $I$ (W/m$^2$)", fontsize=9.5)
        if subcap:
            caption(ax, subcap, y=-0.22)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.4))
    _p8a(axes[0, 0], "(a) Global feature importance (5-fold OOF TreeSHAP)")
    _p8b(axes[0, 1], fig, r"(b) SHAP summary beeswarm ($n = 780$)")
    _p8c(axes[1, 0], fig, r"(c) SHAP dependence: $I$ vs $\dot{m}$")
    _p8d(axes[1, 1], fig, r"(d) SHAP dependence: $\dot{m}$ vs $I$")
    fig.subplots_adjust(left=0.18, right=0.96, top=0.97, bottom=0.10, wspace=0.44, hspace=0.36)
    save_both(fig, "Fig8_shap_analysis", FIG_DIR)

    for tag, kind in [("Fig8a_importance", "a"), ("Fig8b_beeswarm", "b"), ("Fig8c_dep_I", "c"), ("Fig8d_dep_m", "d")]:
        fp, axp = plt.subplots(figsize=(5.8, 4.6))
        if kind == "a":
            _p8a(axp, None)
        elif kind == "b":
            _p8b(axp, fp, None)
        elif kind == "c":
            _p8c(axp, fp, None)
        else:
            _p8d(axp, fp, None)
        save_both(fp, tag, PANEL_DIR)


# ---------------------------------------------------------------------------
# Figure 9: 2x2 Out-of-Range Stress Tests, Incidence Angle & Uncertainty
# ---------------------------------------------------------------------------
def make_fig9(diag_res):
    sw = np.load(os.path.join(RES_DIR, "extrapolation_sweeps.npz"))
    mc = diag_res["uncertainty_analysis"]["monte_carlo_propagation"]
    sweep_models = ["EB_HWB", "XGBoost", "KNN", "ANN", "Hybrid_HWB_XGB"]

    def _p9a(ax, subcap=None):
        I_g = sw["I_grid"]
        ax.axvspan(151.6, 1087.0, color=SHADE_TRAIN, alpha=1.0, zorder=0, label="Training range [152, 1087]")
        for m in sweep_models:
            st = SSAH_MODEL_STYLES[m]
            ax.plot(I_g, sw[f"I_{m}"], color=st["color"], ls=st["ls"], lw=st["lw"],
                    marker=st["marker"], markevery=14, markersize=5.2, zorder=st["zorder"], label=st["label"])
        ax.axhline(10.0, color="#000000", lw=0.9, ls=":", alpha=0.7)
        ax.set_xlim(0, 1400)
        ax.set_ylim(5, 76)
        ax.set_xlabel(r"Solar irradiance, $I$ (W/m$^2$)")
        ax.set_ylabel("Predicted outlet temperature (°C)")
        sci_style(ax)
        boxed_legend(ax, loc="upper left", ncol=1, fontsize=8.0)
        if subcap:
            caption(ax, subcap, y=-0.22)

    def _p9b(ax, subcap=None):
        m_g = sw["m_grid"]
        ax.axvspan(0.0077, 0.0487, color=SHADE_TRAIN, alpha=1.0, zorder=0, label="Training range [0.0077, 0.0487]")
        for m in sweep_models:
            st = SSAH_MODEL_STYLES[m]
            ax.plot(m_g, sw[f"m_{m}"], color=st["color"], ls=st["ls"], lw=st["lw"],
                    marker=st["marker"], markevery=12, markersize=5.2, zorder=st["zorder"], label=st["label"])
        ax.set_xlim(0.004, 0.065)
        ax.set_ylim(18, 128)
        ax.set_xlabel(r"Mass flow rate, $\dot{m}$ (kg/s)")
        ax.set_ylabel("Predicted outlet temperature (°C)")
        sci_style(ax)
        boxed_legend(ax, loc="upper right", ncol=1, fontsize=8.0)
        if subcap:
            caption(ax, subcap, y=-0.22)

    def _p9c(ax, subcap=None):
        binned = diag_res["incidence_angle"]["binned_errors_lodo_xgb"]
        bins_lbl = [b["bin_deg"] for b in binned]
        xb = np.arange(len(bins_lbl))
        wb = 0.34
        ax.bar(xb - wb / 2, [b["RMSE_no_theta"] for b in binned], width=wb, color="#FF7F2A", edgecolor="#000000", linewidth=0.9, label=r"Without $\theta$")
        ax.bar(xb + wb / 2, [b["RMSE_with_theta"] for b in binned], width=wb, color="#4A90E2", edgecolor="#000000", linewidth=0.9, label=r"With $\theta$")
        for i, b in enumerate(binned):
            ax.text(i - wb / 2, b["RMSE_no_theta"] + 0.14, f"{b['RMSE_no_theta']:.2f}", ha="center", va="bottom", fontsize=8.3)
            ax.text(i + wb / 2, b["RMSE_with_theta"] + 0.14, f"{b['RMSE_with_theta']:.2f}", ha="center", va="bottom", fontsize=8.3)
            ax.text(i, 0.35, f"$n={b['n']}$", ha="center", va="bottom", fontsize=8.2, color="#FFFFFF", fontweight="bold")
        ax.set_xticks(xb)
        ax.set_xticklabels([f"$\\theta \\in {lbl}^\\circ$" for lbl in bins_lbl], fontsize=9.5)
        ax.set_ylim(0, max(b["RMSE_with_theta"] for b in binned) * 1.24)
        ax.set_xlabel("Solar incidence angle bin (deg)")
        ax.set_ylabel("LODO RMSE (°C)")
        sci_style(ax, minor_x=False, minor_y=True)
        boxed_legend(ax, loc="upper left", ncol=1, fontsize=8.5)
        if subcap:
            caption(ax, subcap, y=-0.22)

    def _p9d(ax, subcap=None):
        mc_models = ["XGBoost", "KNN", "SVR", "ANN", "Ridge", "EB_HWB"]
        xm = np.arange(len(mc_models))
        wm = 0.25
        b_rmse = [mc[m]["baseline_RMSE_C"] for m in mc_models]
        i_rmse = [mc[m]["inf_noisy_RMSE_mean_C"] for m in mc_models]
        e_rmse = [mc[m]["e2e_noisy_RMSE_mean_C"] for m in mc_models]
        sigmas = [mc[m]["inf_propagated_sigma_mean_C"] for m in mc_models]

        ax.bar(xm - wm, b_rmse, width=wm, color="#4A90E2", edgecolor="#000000", linewidth=0.8, label="Clean baseline RMSE")
        ax.bar(xm, i_rmse, width=wm, color="#3CC21C", edgecolor="#000000", linewidth=0.8, label=r"Noisy test inputs ($M=200$)")
        ax.bar(xm + wm, e_rmse, width=wm, color="#FF7F2A", edgecolor="#000000", linewidth=0.8, label=r"Noisy train+test ($M=40$)")
        for i, (ir, er, sig) in enumerate(zip(i_rmse, e_rmse, sigmas)):
            ax.text(i + 0.04, max(ir, er) + 0.14, f"$\\sigma={sig:.2f}$°C", ha="center", va="bottom", fontsize=7.9)
        ax.set_xticks(xm)
        ax.set_xticklabels([SSAH_MODEL_STYLES[m]["label"] for m in mc_models], fontsize=8.6)
        ax.set_ylim(0, max(e_rmse) * 1.26)
        ax.set_xlabel("Model")
        ax.set_ylabel("Test RMSE (°C)")
        sci_style(ax, minor_x=False, minor_y=True)
        boxed_legend(ax, loc="upper left", ncol=1, fontsize=8.0)
        if subcap:
            caption(ax, subcap, y=-0.22)

    r_p, p_p = diag_res["incidence_angle"]["residual_vs_theta_pearson"]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.6))
    _p9a(axes[0, 0], r"(a) Out-of-range irradiance sweep ($T_{\mathrm{in}}=10$°C, $\dot{m}=0.025$ kg/s)")
    _p9b(axes[0, 1], r"(b) Out-of-range flow rate sweep ($T_{\mathrm{in}}=10$°C, $I=800$ W/m$^2$)")
    _p9c(axes[1, 0], f"(c) Incidence angle effect (LODO, $r={r_p:.3f}$, $p={p_p:.3g}$)")
    _p9d(axes[1, 1], r"(d) Monte Carlo uncertainty ($u_T=0.5$°C, $u_{\dot{m}}=3\%$, $u_I=2\%$)")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.97, bottom=0.10, wspace=0.20, hspace=0.36)
    save_both(fig, "Fig9_stress_inc_uncertainty", FIG_DIR)

    for tag, fn in [("Fig9a_oor_irrad", _p9a), ("Fig9b_oor_flow", _p9b), ("Fig9c_incidence", _p9c), ("Fig9d_mc_uncertainty", _p9d)]:
        fp, axp = plt.subplots(figsize=(5.8, 4.6))
        fn(axp, None)
        save_both(fp, tag, PANEL_DIR)


# ---------------------------------------------------------------------------
# Figure 10: 1x2 Multi-Parameter Pareto Optimization (+ standalone panels)
# ---------------------------------------------------------------------------
def make_fig10(diag_res):
    cand = pd.read_csv(os.path.join(RES_DIR, "pareto_candidates_I800_Tin10.csv"))
    sup = cand[cand["is_supported"] == 1].copy()
    pareto = pd.read_csv(os.path.join(RES_DIR, "pareto_front_I800_Tin10.csv"))
    opt = diag_res["pareto_optimization"]
    base = opt["baseline_prototype_H0.04_N8_m0.025"]
    pta = opt["PointA_MaxEfficiency_Tout_ge35C"]
    ptb = opt["PointB_BalancedHeating_Tout_ge45C"]
    ptc = opt["PointC_HighTempDrying_Tout_ge60C"]

    def _p10a(ax, fig_obj, subcap=None):
        sc = ax.scatter(sup["Tout_hybrid_mean_C"], sup["eta_hybrid_mean_pct"], c=sup["m_dot_kg_s"],
                        cmap="viridis", s=20, alpha=0.68, edgecolors="none", label=f"Supported candidates ($n={len(sup)}$)")
        ax.fill_between(pareto["Tout_hybrid_mean_C"], pareto["eta_hybrid_ci95_lo_pct"], pareto["eta_hybrid_ci95_hi_pct"],
                        color="#D62728", alpha=0.14, zorder=3, label="Pareto 95% bootstrap CI")
        ax.plot(pareto["Tout_hybrid_mean_C"], pareto["eta_hybrid_mean_pct"], color="#D62728", ls="-", lw=2.1, zorder=4,
                label="Pareto front (Hybrid HWB+XGB)")

        ax.scatter([base["Tout_hybrid_mean_C"]], [base["eta_hybrid_mean_pct"]], s=110, marker="X", color="#000000", zorder=6,
                   label=r"Baseline ($H=0.04, N=8, \dot{m}=0.025$)")
        ax.scatter([pta["Tout_hybrid_mean_C"]], [pta["eta_hybrid_mean_pct"]], s=95, marker="o", color="#3CC21C", edgecolors="#000000", lw=1.0, zorder=6,
                   label=r"Point A: Max $\eta$ ($T_o \geq 35$°C)")
        ax.scatter([ptb["Tout_hybrid_mean_C"]], [ptb["eta_hybrid_mean_pct"]], s=100, marker="D", color="#4A90E2", edgecolors="#000000", lw=1.0, zorder=6,
                   label=r"Point B: Balanced ($T_o \geq 45$°C)")
        ax.scatter([ptc["Tout_hybrid_mean_C"]], [ptc["eta_hybrid_mean_pct"]], s=100, marker="^", color="#FF7F2A", edgecolors="#000000", lw=1.0, zorder=6,
                   label=r"Point C: High-$T$ ($T_o \geq 60$°C)")

        ax.annotate(f"Base\n({base['Tout_hybrid_mean_C']:.1f}°C, {base['eta_hybrid_mean_pct']:.1f}%)",
                    xy=(base["Tout_hybrid_mean_C"], base["eta_hybrid_mean_pct"]),
                    xytext=(base["Tout_hybrid_mean_C"] + 5.0, base["eta_hybrid_mean_pct"] - 5.2),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#000000"), fontsize=8.2)
        ax.annotate(f"A: $H={pta['H_m']:.2f}, N={int(pta['N_baffles'])}, \\dot{{m}}={pta['m_dot_kg_s']:.3f}$\n({pta['Tout_hybrid_mean_C']:.1f}°C, {pta['eta_hybrid_mean_pct']:.1f}%)",
                    xy=(pta["Tout_hybrid_mean_C"], pta["eta_hybrid_mean_pct"]),
                    xytext=(pta["Tout_hybrid_mean_C"] + 2.0, pta["eta_hybrid_mean_pct"] + 3.0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#000000"), fontsize=8.0)
        ax.annotate(f"B: $H={ptb['H_m']:.2f}, N={int(ptb['N_baffles'])}, \\dot{{m}}={ptb['m_dot_kg_s']:.4f}$\n({ptb['Tout_hybrid_mean_C']:.1f}°C, {ptb['eta_hybrid_mean_pct']:.1f}%)",
                    xy=(ptb["Tout_hybrid_mean_C"], ptb["eta_hybrid_mean_pct"]),
                    xytext=(ptb["Tout_hybrid_mean_C"] - 8.8, ptb["eta_hybrid_mean_pct"] - 10.5),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#000000"),
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor="#000000", lw=0.7, alpha=0.94),
                    fontsize=8.0)
        ax.annotate(f"C: $H={ptc['H_m']:.2f}, N={int(ptc['N_baffles'])}, \\dot{{m}}={ptc['m_dot_kg_s']:.3f}$\n({ptc['Tout_hybrid_mean_C']:.1f}°C, {ptc['eta_hybrid_mean_pct']:.1f}%)",
                    xy=(ptc["Tout_hybrid_mean_C"], ptc["eta_hybrid_mean_pct"]),
                    xytext=(ptc["Tout_hybrid_mean_C"] + 1.5, ptc["eta_hybrid_mean_pct"] + 5.5),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#000000"), fontsize=8.0)

        cb = fig_obj.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        style_colorbar(cb, r"Mass flow rate $\dot{m}$ (kg/s)", fontsize=9.5)
        ax.set_xlim(32, 78)
        ax.set_ylim(37, 87.5)
        ax.set_xlabel(r"Predicted outlet temperature, $T_o$ (°C)")
        ax.set_ylabel(r"Predicted thermal efficiency, $\eta$ (%)")
        sci_style(ax)
        boxed_legend(ax, loc="lower left", ncol=1, fontsize=7.6)
        if subcap:
            caption(ax, subcap, y=-0.22)

    def _p10b(ax, subcap=None):
        sub_h = sup[np.isclose(sup["H_m"], 0.050)]
        ax_t = ax.twinx()
        n_cols = {1: "#7F8C8D", 2: "#FF7F2A", 4: "#3CC21C", 6: "#17A589", 8: "#4A90E2"}
        for n_b in [1, 2, 4, 6, 8]:
            sn = sub_h[sub_h["N_baffles"] == n_b].sort_values("m_dot_kg_s")
            ax.plot(sn["m_dot_kg_s"], sn["eta_hybrid_mean_pct"], "-", lw=1.7, color=n_cols[n_b], label=rf"$\eta$ Hybrid ($N={n_b}$)")
            ax_t.plot(sn["m_dot_kg_s"], sn["Tout_hybrid_mean_C"], "--", lw=1.2, color=n_cols[n_b], alpha=0.78)

        sn2 = sub_h[sub_h["N_baffles"] == 2].sort_values("m_dot_kg_s")
        ax.plot(sn2["m_dot_kg_s"], sn2["eta_xgb_mean_pct"], color="#D62728", ls=":", lw=2.0, zorder=5,
                label=r"$\eta$ Pure XGBoost ($N=2$, $>100\%$)")
        ax.axhline(100.0, color="#D62728", lw=1.0, ls="-.", alpha=0.75)

        ax.set_xlim(0.013, 0.046)
        ax.set_ylim(44, 108)
        ax_t.set_ylim(15, 76)
        ax.set_xlabel(r"Mass flow rate, $\dot{m}$ (kg/s)")
        ax.set_ylabel(r"Thermal efficiency, $\eta$ (%) — solid/dotted")
        ax_t.set_ylabel(r"Outlet temperature, $T_o$ (°C) — dashed")
        sci_style(ax)
        # Ensure twin axis also obeys inward ticks & thick right spine
        ax_t.spines["right"].set_linewidth(1.6)
        ax_t.spines["right"].set_color("#000000")
        ax_t.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax_t.tick_params(axis="y", which="major", direction="in", length=5.0, width=1.3, colors="#000000", pad=5)
        ax_t.tick_params(axis="y", which="minor", direction="in", length=2.8, width=0.9, colors="#000000")
        boxed_legend(ax, loc="lower right", ncol=2, fontsize=7.6)
        if subcap:
            caption(ax, subcap, y=-0.22)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    _p10a(axes[0], fig, r"(a) Supported-domain Pareto front ($I = 800\ \mathrm{W/m}^2$, $T_{\mathrm{in}} = 10$°C)")
    _p10b(axes[1], r"(b) Coupled $\dot{m}$ and $N$ effects at $H = 0.050$ m (Hybrid vs Pure XGBoost)")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.96, bottom=0.18, wspace=0.30)
    save_both(fig, "Fig10_pareto_optimization", FIG_DIR)

    fp1, axp1 = plt.subplots(figsize=(6.4, 5.0))
    _p10a(axp1, fp1, None)
    save_both(fp1, "Fig10a_pareto", PANEL_DIR)

    fp2, axp2 = plt.subplots(figsize=(6.2, 5.0))
    _p10b(axp2, None)
    save_both(fp2, "Fig10b_tradeoff", PANEL_DIR)


def main():
    with open(os.path.join(RES_DIR, "data_audit.json"), "r", encoding="utf-8") as f:
        data_audit = json.load(f)
    with open(os.path.join(RES_DIR, "experiment_results.json"), "r", encoding="utf-8") as f:
        exp_res = json.load(f)
    with open(os.path.join(RES_DIR, "diagnostics_results.json"), "r", encoding="utf-8") as f:
        diag_res = json.load(f)

    make_fig3(exp_res)
    make_fig4(exp_res)
    make_fig5(exp_res)
    make_fig6()
    make_fig7(data_audit, exp_res)
    make_fig8(diag_res)
    make_fig9(diag_res)
    make_fig10(diag_res)
    print("Generated all 8 composite figures (600-DPI PNG + PDF + SVG) and standalone panels in:", FIG_DIR, flush=True)


if __name__ == "__main__":
    main()
