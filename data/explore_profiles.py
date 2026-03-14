"""
data/explore_profiles.py
=========================
Statistical exploration of data/profiles.csv.
Produces a single figure with 10 panels covering all columns.

Run
---
  cd flowscore/
  python data/explore_profiles.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter

# ── Load and expand JSON columns ──────────────────────────────────────────────
df = pd.read_csv("data/profiles.csv")

df["income_mean"]  = df["monthly_income_hist"].apply(lambda s: np.mean(json.loads(s)))
df["income_cv"]    = df["monthly_income_hist"].apply(
    lambda s: np.std(json.loads(s)) / max(np.mean(json.loads(s)), 1)
)
df["fixed_mean"]    = df["monthly_fixed_exp_hist"].apply(lambda s: np.mean(json.loads(s)))
df["variable_mean"] = df["monthly_variable_exp_hist"].apply(lambda s: np.mean(json.loads(s)))
df["saving_final"]  = df["monthly_saving_hist"].apply(lambda s: json.loads(s)[-1])
df["saving_min"]    = df["monthly_saving_hist"].apply(lambda s: min(json.loads(s)))
df["ever_negative"] = df["saving_min"] < 0

CAT_ORDER  = ["gig", "part_time", "freelance", "fixed_term"]
CAT_COLORS = {"gig": "#e05c5c", "part_time": "#5c8fe0",
              "freelance": "#e0b25c", "fixed_term": "#5cb87a"}
palette    = [CAT_COLORS[c] for c in CAT_ORDER]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 23))
fig.suptitle("FlowScore – Synthetic Dataset Exploration  (N=500)",
             fontsize=15, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.50, wspace=0.38)

ax1  = fig.add_subplot(gs[0, 0])
ax2  = fig.add_subplot(gs[0, 1])
ax3  = fig.add_subplot(gs[0, 2])
ax4  = fig.add_subplot(gs[1, 0])
ax5  = fig.add_subplot(gs[1, 1])
ax6  = fig.add_subplot(gs[1, 2])
ax7  = fig.add_subplot(gs[2, 0])
ax8  = fig.add_subplot(gs[2, 1])
ax9  = fig.add_subplot(gs[2, 2])
ax10 = fig.add_subplot(gs[3, :])   # full-width expense-evolution panel
ax11 = fig.add_subplot(gs[4, :])   # full-width debt trajectory panel


# ── 1. Working category distribution ─────────────────────────────────────────
counts = df["working_category"].value_counts().reindex(CAT_ORDER)
bars = ax1.bar(CAT_ORDER, counts.values, color=palette, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
             f"{val}\n({100*val/len(df):.0f}%)", ha="center", va="bottom", fontsize=8)
ax1.set_title("Working category distribution")
ax1.set_ylabel("Count")
ax1.set_ylim(0, counts.max() * 1.25)
ax1.tick_params(axis="x", labelsize=8)


# ── 2. Mean monthly income by category (box plot) ────────────────────────────
data_by_cat = [df.loc[df["working_category"] == c, "income_mean"].values for c in CAT_ORDER]
bp = ax2.boxplot(data_by_cat, patch_artist=True, medianprops=dict(color="black", linewidth=1.5),
                 widths=0.5)
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax2.set_xticklabels(CAT_ORDER, fontsize=8)
ax2.set_title("Mean monthly income by category")
ax2.set_ylabel("EUR / month")


# ── 3. Income CV (volatility) by category ────────────────────────────────────
cv_by_cat = [df.loc[df["working_category"] == c, "income_cv"].values for c in CAT_ORDER]
bp2 = ax3.boxplot(cv_by_cat, patch_artist=True, medianprops=dict(color="black", linewidth=1.5),
                  widths=0.5)
for patch, color in zip(bp2["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax3.set_xticklabels(CAT_ORDER, fontsize=8)
ax3.set_title("Income volatility (CV) by category")
ax3.set_ylabel("Coefficient of Variation")


# ── 4. int_defaults distribution ─────────────────────────────────────────────
def_counts = df["int_defaults"].value_counts().sort_index()
def_colors = ["#5cb87a", "#e0b25c", "#e05c5c"]
bars2 = ax4.bar(def_counts.index, def_counts.values,
                color=def_colors[:len(def_counts)], edgecolor="white")
for bar, val in zip(bars2, def_counts.values):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             f"{val} ({100*val/len(df):.1f}%)", ha="center", va="bottom", fontsize=8)
ax4.set_title("int_defaults distribution")
ax4.set_xlabel("Number of historical defaults")
ax4.set_ylabel("Count")
ax4.set_xticks([0, 1, 2])
ax4.set_ylim(0, def_counts.max() * 1.20)


# ── 5. Default rate by category (stacked bar) ────────────────────────────────
def_by_cat = (
    df.groupby(["working_category", "int_defaults"])
    .size()
    .unstack(fill_value=0)
    .reindex(CAT_ORDER)
)
def_pct = def_by_cat.div(def_by_cat.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(CAT_ORDER))
colors_stack = ["#5cb87a", "#e0b25c", "#e05c5c"]
labels_stack = ["0 defaults", "1 default", "2 defaults"]
for col, col_color, col_label in zip(def_pct.columns, colors_stack, labels_stack):
    vals = def_pct[col].values
    ax5.bar(CAT_ORDER, vals, bottom=bottom, color=col_color,
            label=col_label, edgecolor="white", linewidth=0.6)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 4:
            ax5.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold")
    bottom += vals
ax5.set_title("Default rate by category")
ax5.set_ylabel("% of category")
ax5.legend(fontsize=7, loc="upper right")
ax5.set_ylim(0, 108)
ax5.tick_params(axis="x", labelsize=8)


# ── 6. Fixed vs variable expenses scatter (using per-profile means) ───────────
for cat in CAT_ORDER:
    mask = df["working_category"] == cat
    ax6.scatter(df.loc[mask, "fixed_mean"],
                df.loc[mask, "variable_mean"],
                c=CAT_COLORS[cat], alpha=0.45, s=18, label=cat)
ax6.set_title("Fixed vs variable expenses (6-month mean)")
ax6.set_xlabel("Mean fixed exp (EUR/month)")
ax6.set_ylabel("Mean variable exp (EUR/month)")
ax6.legend(fontsize=7, markerscale=1.5)


# ── 7. Saving trajectory: median ± IQR per month ─────────────────────────────
months = np.arange(6)
for cat, color in zip(CAT_ORDER, palette):
    mask = df["working_category"] == cat
    histories = np.array(
        df.loc[mask, "monthly_saving_hist"].apply(json.loads).tolist()
    )
    med = np.median(histories, axis=0)
    q25 = np.percentile(histories, 25, axis=0)
    q75 = np.percentile(histories, 75, axis=0)
    ax7.plot(months, med, color=color, linewidth=2, label=cat)
    ax7.fill_between(months, q25, q75, color=color, alpha=0.15)
ax7.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax7.set_title("Saving trajectory (median ± IQR)")
ax7.set_xlabel("Month")
ax7.set_ylabel("Running balance (EUR)")
ax7.set_xticks(months)
ax7.legend(fontsize=7)


# ── 8. BNPL exposure distribution ────────────────────────────────────────────
bnpl_users = df[df["bnpl_exposure"] > 0]["bnpl_exposure"]
ax8.hist(bnpl_users, bins=20, color="#9b77d1", edgecolor="white", linewidth=0.7)
ax8.set_title(f"BNPL exposure (users only, n={len(bnpl_users)})")
ax8.set_xlabel("Monthly BNPL instalment (EUR)")
ax8.set_ylabel("Count")
non_users = int((df["bnpl_exposure"] == 0).sum())
ax8.text(0.97, 0.95, f"No BNPL: {non_users} ({100*non_users/len(df):.0f}%)\n"
         f"BNPL: {len(bnpl_users)} ({100*len(bnpl_users)/len(df):.0f}%)",
         transform=ax8.transAxes, ha="right", va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))


# ── 9. pay_on_time_bills distribution ────────────────────────────────────────
ax9.hist(df["pay_on_time_bills"], bins=np.linspace(0, 1, 8),
         color="#5c8fe0", edgecolor="white", linewidth=0.7)
ax9.xaxis.set_major_formatter(PercentFormatter(xmax=1))
ax9.set_title("Pay-on-time bills distribution")
ax9.set_xlabel("Fraction of months balance > -€50")
ax9.set_ylabel("Count")
mu = df["pay_on_time_bills"].mean()
ax9.axvline(mu, color="crimson", linewidth=1.4, linestyle="--",
            label=f"Mean = {mu:.2f}")
ax9.legend(fontsize=8)


# ── 10. Expense evolution over time (median fixed & variable per category) ────
months_labels = [f"M{m}" for m in range(6)]
for cat, color in zip(CAT_ORDER, palette):
    mask = df["working_category"] == cat

    fixed_histories = np.array(
        df.loc[mask, "monthly_fixed_exp_hist"].apply(json.loads).tolist()
    )
    var_histories = np.array(
        df.loc[mask, "monthly_variable_exp_hist"].apply(json.loads).tolist()
    )

    med_fixed = np.median(fixed_histories, axis=0)
    med_var   = np.median(var_histories,   axis=0)

    ax10.plot(months, med_fixed, color=color, linewidth=2,
              linestyle="-",  label=f"{cat} fixed")
    ax10.plot(months, med_var,   color=color, linewidth=1.5,
              linestyle="--", label=f"{cat} variable")

ax10.set_title("Expense evolution over time (median per category)  — solid=fixed, dashed=variable")
ax10.set_xlabel("Month")
ax10.set_ylabel("EUR / month")
ax10.set_xticks(months)
ax10.set_xticklabels(months_labels)
ax10.legend(fontsize=7, ncol=4, loc="upper right")
ax10.axhline(0, color="black", linewidth=0.6, linestyle=":")


# ── 11. Debt trajectory (BNPL users only, median ± IQR per category) ─────────
bnpl_df = df[df["bnpl_exposure"] > 0]   # exclude non-BNPL users
for cat, color in zip(CAT_ORDER, palette):
    mask = bnpl_df["working_category"] == cat
    if mask.sum() == 0:
        continue
    histories = np.array(
        bnpl_df.loc[mask, "debt_hist"].apply(json.loads).tolist()
    )
    med = np.median(histories, axis=0)
    q25 = np.percentile(histories, 25, axis=0)
    q75 = np.percentile(histories, 75, axis=0)
    n   = mask.sum()
    ax11.plot(months, med, color=color, linewidth=2, label=f"{cat} (n={n})")
    ax11.fill_between(months, q25, q75, color=color, alpha=0.15)

    # Vertical dashed line at mean peak month for this category
    peak_months = np.argmax(histories, axis=1)   # peak month for each profile
    mean_peak   = float(np.mean(peak_months))
    ax11.axvline(mean_peak, color=color, linewidth=1.0, linestyle=":",
                 alpha=0.7, label=f"{cat} avg peak @ m{mean_peak:.1f}")

ax11.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax11.set_title("Debt trajectory (BNPL users only, median ± IQR)  — dotted = mean peak month per category")
ax11.set_xlabel("Month")
ax11.set_ylabel("Debt balance (EUR)")
ax11.set_xticks(months)
ax11.legend(fontsize=7, ncol=4, loc="upper left")


# ── Save & show ───────────────────────────────────────────────────────────────
plt.savefig("data/profiles_stats.png", dpi=150, bbox_inches="tight")
print("Saved: data/profiles_stats.png")
plt.show()
