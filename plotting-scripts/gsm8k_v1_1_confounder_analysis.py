"""
Confounder analysis for gsm8k-v1 test set (gsm8k-v1.1 version of
humaneval_v1_confounder_analysis.py).

Checks whether pos/neg (correct) is confounded with:
  1. Answer length (characters and tokens)
  2. Generation strategy
  3. Model used
  4. Temperature
  5. Interactions (strategy × model, strategy × length, etc.)

Produces figures in output-metrics/confounder_gsm8k_v1.1.1/ and prints summary tables.
"""

import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = "output-metrics/confounder_gsm8k_v1.1.1"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob("data/gsm8k/v1.1/gsm8k_test_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df["is_correct"] = (df["correct"] == "Yes").astype(int)
df["answer_len"] = df["answer"].astype(str).str.len()
df["answer_lines"] = df["answer"].astype(str).str.count("\n") + 1

df["temp_bin"] = pd.cut(
    df["temperature"],
    bins=[0, 0.3, 0.5, 0.8, 1.1, 1.3, 2.0],
    labels=["≤0.3", "0.3-0.5", "0.5-0.8", "0.8-1.1", "1.1-1.3", ">1.3"],
)

MODEL_SHORT = {
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3-mini",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "deepseek-ai/deepseek-coder-1.3b-instruct": "DS-Coder-1.3B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "allenai/OLMo-2-0425-1B-Instruct": "OLMo-2-1B",
    "CodeLlama-34b-Instruct": "CodeLlama-34B",
    "Llama-3.3-70B": "Llama-3.3-70B",
    "Mistral-3.2-24B": "Mistral-3.2-24B",
}
df["model_short"] = df["model"].map(lambda x: MODEL_SHORT.get(x, x))

PALETTE = {"Yes": "#4CAF50", "No": "#E53935"}

print("=" * 70)
print("GSM8K-V1 TEST SET — CONFOUNDER ANALYSIS")
print(f"Total samples: {len(df)}  |  Tasks: {df['task_id'].nunique()}")
print(f"Correct: {df['is_correct'].sum()} ({df['is_correct'].mean():.1%})  "
      f"Incorrect: {(1 - df['is_correct']).sum():.0f} ({1 - df['is_correct'].mean():.1%})")
print("=" * 70)

print("\n── 1. ANSWER LENGTH vs CORRECT ──")

for col, label in [("answer_len", "characters"), ("answer_lines", "lines")]:
    yes = df.loc[df["correct"] == "Yes", col]
    no = df.loc[df["correct"] == "No", col]
    u_stat, u_p = stats.mannwhitneyu(yes, no, alternative="two-sided")
    d = (no.mean() - yes.mean()) / np.sqrt((yes.std() ** 2 + no.std() ** 2) / 2)
    print(f"\n  {label}:")
    print(f"    Correct mean={yes.mean():.1f}  median={yes.median():.1f}  std={yes.std():.1f}")
    print(f"    Incorrect mean={no.mean():.1f}  median={no.median():.1f}  std={no.std():.1f}")
    print(f"    Mann-Whitney U={u_stat:.0f}, p={u_p:.2e}, Cohen's d={d:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, label in [(axes[0], "answer_len", "Answer Length (chars)"),
                         (axes[1], "answer_lines", "Answer Length (lines)")]:
    for val, color in [("Yes", PALETTE["Yes"]), ("No", PALETTE["No"])]:
        subset = df.loc[df["correct"] == val, col]
        clip = subset.clip(upper=subset.quantile(0.98))
        ax.hist(clip, bins=50, alpha=0.55, label=f"correct={val} (n={len(subset)})",
                color=color, density=True)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"{label} by Correctness")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/1a_length_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(14, 6))
plot_df = df.copy()
plot_df["answer_len_clipped"] = plot_df["answer_len"].clip(upper=plot_df["answer_len"].quantile(0.98))
sns.boxplot(data=plot_df, x="strategy", y="answer_len_clipped", hue="correct",
            palette=PALETTE, ax=ax, fliersize=2)
ax.set_xlabel("Strategy")
ax.set_ylabel("Answer Length (chars, clipped at 98th pct)")
ax.set_title("Answer Length by Strategy and Correctness")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/1b_length_by_strategy_correct.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n── 2. STRATEGY vs CORRECT ──")

ct_strat = pd.crosstab(df["strategy"], df["correct"])
ct_strat["total"] = ct_strat.sum(axis=1)
ct_strat["pct_correct"] = (ct_strat["Yes"] / ct_strat["total"] * 100).round(1)
ct_strat = ct_strat.sort_values("pct_correct", ascending=False)
print(ct_strat.to_string())

chi2, p_chi, dof, _ = stats.chi2_contingency(pd.crosstab(df["strategy"], df["correct"]))
print(f"\n  Chi-squared={chi2:.2f}, df={dof}, p={p_chi:.2e}")

fig, ax = plt.subplots(figsize=(10, 5))
strat_pct = df.groupby("strategy")["is_correct"].agg(["mean", "count"]).reset_index()
strat_pct = strat_pct.sort_values("mean", ascending=True)
bars = ax.barh(strat_pct["strategy"], strat_pct["mean"] * 100, color="#5C6BC0")
for bar, n in zip(bars, strat_pct["count"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"n={n}", va="center", fontsize=9)
ax.set_xlabel("% Correct")
ax.set_title("Correctness Rate by Generation Strategy")
ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
ax.set_xlim(0, 100)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/2_strategy_correctness.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n── 3. MODEL vs CORRECT ──")

ct_model = pd.crosstab(df["model_short"], df["correct"])
ct_model["total"] = ct_model.sum(axis=1)
ct_model["pct_correct"] = (ct_model["Yes"] / ct_model["total"] * 100).round(1)
ct_model = ct_model.sort_values("pct_correct", ascending=False)
print(ct_model.to_string())

chi2m, pm, dofm, _ = stats.chi2_contingency(pd.crosstab(df["model_short"], df["correct"]))
print(f"\n  Chi-squared={chi2m:.2f}, df={dofm}, p={pm:.2e}")

fig, ax = plt.subplots(figsize=(10, 7))
model_pct = df.groupby("model_short")["is_correct"].agg(["mean", "count"]).reset_index()
model_pct = model_pct.sort_values("mean", ascending=True)
colors = ["#E53935" if m < 0.40 else "#FF9800" if m < 0.55 else "#4CAF50"
          for m in model_pct["mean"]]
bars = ax.barh(model_pct["model_short"], model_pct["mean"] * 100, color=colors)
for bar, n in zip(bars, model_pct["count"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"n={n}", va="center", fontsize=8)
ax.set_xlabel("% Correct")
ax.set_title("Correctness Rate by Model")
ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
ax.set_xlim(0, 100)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/3_model_correctness.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n── 4. TEMPERATURE vs CORRECT ──")

temp_df = df.dropna(subset=["temperature"])
ct_temp = pd.crosstab(temp_df["temp_bin"], temp_df["correct"])
ct_temp["total"] = ct_temp.sum(axis=1)
ct_temp["pct_correct"] = (ct_temp["Yes"] / ct_temp["total"] * 100).round(1)
print(ct_temp.to_string())

r_pb, p_pb = stats.pointbiserialr(temp_df["is_correct"], temp_df["temperature"])
print(f"\n  Point-biserial r={r_pb:.4f}, p={p_pb:.4f}")
print(f"  (NaN temps excluded: {df['temperature'].isna().sum()} rows)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.violinplot(data=temp_df, x="correct", y="temperature", palette=PALETTE, ax=axes[0],
               inner="quartile")
axes[0].set_title("Temperature Distribution by Correctness")

temp_rate = temp_df.groupby("temp_bin", observed=True)["is_correct"].agg(["mean", "count"]).reset_index()
bars = axes[1].bar(temp_rate["temp_bin"].astype(str), temp_rate["mean"] * 100,
                   color="#5C6BC0", alpha=0.8)
for bar, n in zip(bars, temp_rate["count"]):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"n={n}", ha="center", fontsize=9)
axes[1].set_xlabel("Temperature Bin")
axes[1].set_ylabel("% Correct")
axes[1].set_title("Correctness Rate by Temperature")
axes[1].axhline(50, color="gray", linestyle="--", alpha=0.5)
axes[1].set_ylim(0, 100)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/4_temperature.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n── 5. STRATEGY × MODEL INTERACTION ──")

model_counts = df["model_short"].value_counts()
big_models = model_counts[model_counts >= 30].index.tolist()
sub = df[df["model_short"].isin(big_models)]

pivot = sub.pivot_table(values="is_correct", index="model_short",
                        columns="strategy", aggfunc="mean") * 100
pivot_n = sub.pivot_table(values="is_correct", index="model_short",
                          columns="strategy", aggfunc="count")

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", center=50,
            linewidths=0.5, ax=ax, vmin=0, vmax=100,
            cbar_kws={"label": "% Correct"})
ax.set_title("Correctness Rate (%) — Strategy × Model\n(models with n ≥ 30 samples)")
ax.set_ylabel("Model")
ax.set_xlabel("Strategy")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/5_strategy_model_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n── 6. MODEL × STRATEGY COMPOSITION (% of each strategy's samples) ──")

comp = pd.crosstab(df["model_short"], df["strategy"], normalize="columns") * 100
print(comp.round(1).to_string())

fig, ax = plt.subplots(figsize=(12, 7))
comp_t = comp.T
comp_t.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.8)
ax.set_ylabel("% of Samples")
ax.set_xlabel("Strategy")
ax.set_title("Model Composition Within Each Strategy")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/6_model_strategy_composition.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n── 7. LENGTH DIFFERENCE WITHIN EACH STRATEGY ──")

len_by_strat = df.groupby(["strategy", "correct"])["answer_len"].agg(["mean", "median", "count"])
len_by_strat = len_by_strat.unstack("correct")
print(len_by_strat.round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
strategies = df["strategy"].unique()
med_data = df.groupby(["strategy", "correct"])["answer_len"].median().unstack("correct")
med_data = med_data.reindex(ct_strat.index[::-1])
x = np.arange(len(med_data))
w = 0.35
ax.barh(x - w / 2, med_data["Yes"], w, label="Correct", color=PALETTE["Yes"], alpha=0.8)
ax.barh(x + w / 2, med_data["No"], w, label="Incorrect", color=PALETTE["No"], alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(med_data.index)
ax.set_xlabel("Median Answer Length (chars)")
ax.set_title("Median Answer Length by Strategy & Correctness")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/7_length_within_strategy.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n── 8. LOGISTIC REGRESSION (all factors) ──")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    reg_df = df.dropna(subset=["temperature"]).copy()
    reg_df["log_len"] = np.log1p(reg_df["answer_len"])

    strat_dummies = pd.get_dummies(reg_df["strategy"], prefix="strat", drop_first=True)
    model_dummies = pd.get_dummies(reg_df["model_short"], prefix="model", drop_first=True)

    X = pd.concat([reg_df[["log_len", "temperature"]], strat_dummies, model_dummies], axis=1)
    y = reg_df["is_correct"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, penalty=None)
    lr.fit(X_scaled, y)

    coef_df = pd.DataFrame({"feature": X.columns, "coef": lr.coef_[0]})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    print(f"  Accuracy: {lr.score(X_scaled, y):.3f}")
    print(f"\n  Top 15 features by |coefficient| (standardized):")
    print(coef_df.head(15).to_string(index=False))

    top = coef_df.head(20).sort_values("coef")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#4CAF50" if c > 0 else "#E53935" for c in top["coef"]]
    ax.barh(top["feature"], top["coef"], color=colors)
    ax.set_xlabel("Logistic Regression Coefficient (standardized)")
    ax.set_title("Top 20 Predictors of Correctness\n(positive = predicts correct)")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/8_logistic_regression_coefs.png", dpi=150, bbox_inches="tight")
    plt.close()

except ImportError:
    print("  sklearn not available, skipping logistic regression")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
for val, color in [("Yes", PALETTE["Yes"]), ("No", PALETTE["No"])]:
    subset = df.loc[df["correct"] == val, "answer_len"].clip(upper=df["answer_len"].quantile(0.98))
    ax.hist(subset, bins=50, alpha=0.55, label=f"correct={val}", color=color, density=True)
ax.set_xlabel("Answer Length (chars)")
ax.set_ylabel("Density")
ax.set_title("A) Length Distributions")
ax.legend()

ax = axes[0, 1]
sp = strat_pct.sort_values("mean", ascending=True)
bars = ax.barh(sp["strategy"], sp["mean"] * 100, color="#5C6BC0")
for bar, n in zip(bars, sp["count"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"n={n}", va="center", fontsize=8)
ax.set_xlabel("% Correct")
ax.set_title("B) Correctness by Strategy")
ax.axvline(50, color="gray", linestyle="--", alpha=0.5)

ax = axes[1, 0]
mp = model_pct.sort_values("mean", ascending=True)
colors_mp = ["#E53935" if m < 0.40 else "#FF9800" if m < 0.55 else "#4CAF50"
             for m in mp["mean"]]
bars = ax.barh(mp["model_short"], mp["mean"] * 100, color=colors_mp)
for bar, n in zip(bars, mp["count"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"n={n}", va="center", fontsize=7)
ax.set_xlabel("% Correct")
ax.set_title("C) Correctness by Model")
ax.axvline(50, color="gray", linestyle="--", alpha=0.5)

ax = axes[1, 1]
bars = ax.bar(temp_rate["temp_bin"].astype(str), temp_rate["mean"] * 100,
              color="#5C6BC0", alpha=0.8)
for bar, n in zip(bars, temp_rate["count"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"n={n}", ha="center", fontsize=9)
ax.set_xlabel("Temperature Bin")
ax.set_ylabel("% Correct")
ax.set_title("D) Correctness by Temperature")
ax.axhline(50, color="gray", linestyle="--", alpha=0.5)

fig.suptitle("GSM8K-V1 Test Set — Confounder Dashboard", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT_DIR}/0_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n{'=' * 70}")
print(f"All figures saved to {OUT_DIR}/")
print(f"{'=' * 70}")
