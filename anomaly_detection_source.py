"""
Real-Time Anomaly Detection in Financial Transactions (SecurePay case study)
Isolation Forest and Local Outlier Factor on the credit card fraud dataset.
Needs creditcard.csv in the same folder. Run: python anomaly_detection_source.py
Plots open one by one, close each window to continue.
"""

# ## Problem Statement
# Financial fraud is a very costly problem for a FinTech company like SecurePay. A fraudulent transaction has to be caught in milliseconds, before the money is lost. But fraud is extremely rare, so a normal classifier can simply say "legitimate" every time and still look more than 99% accurate while catching nothing.
#
# In this project I build an unsupervised anomaly detection system. Instead of teaching the model what fraud looks like, I teach it what normal transactions look like, and flag anything that is very different. I use two algorithms, Isolation Forest and Local Outlier Factor (LOF), compare them with precision and recall, and recommend a threshold (the contamination parameter) that balances catching fraud against blocking genuine customers.

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def display(x):
    print(x)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
pd.set_option("display.max_columns", 40)
RANDOM_STATE = 42

def show(text):
    # prints the findings with the real numbers
    print(text.replace('**', ''))

# ## Data Loading and First Look
# I loaded the credit card transactions dataset. The columns are Time, V1 to V28, Amount and Class. The V columns are anonymous features made with PCA for privacy, so their real meaning is hidden. Class is the label (0 is normal, 1 is fraud). I use the label only to check the results, never to train the models.

df = pd.read_csv("creditcard.csv")
display(df.head())

n_rows, n_cols = df.shape
missing = int(df.isnull().sum().sum())
dups = int(df.duplicated().sum())
show(f"- Shape: **{n_rows:,} rows** and **{n_cols} columns**\n- Missing values: **{missing}**\n- Duplicate rows: **{dups:,}**")
df.info()

# ### Data quality
# All columns are numeric and there are no missing values, so no encoding or imputation is needed. I kept the duplicate rows, because two real customers can make the same purchase and they are a very small part of the data.

# ## Class Imbalance
# Fraud is extremely rare, so accuracy is a useless metric here. I judge the models with precision (how many flagged transactions were really fraud) and recall (how much of the fraud was caught).

counts = df["Class"].value_counts().sort_index()
n_normal, n_fraud = int(counts[0]), int(counts[1])
fraud_pct = n_fraud / len(df) * 100

show(
    f"- Normal transactions: **{n_normal:,}**\n"
    f"- Fraud transactions: **{n_fraud:,}**\n"
    f"- Fraud share: **{fraud_pct:.3f}%**\n"
    f"- About **{n_normal / n_fraud:,.0f}** normal transactions for every fraud\n"
    f"- A model that always says normal would be **{100 - fraud_pct:.2f}%** accurate and catch no fraud at all"
)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, use_log in zip(ax, [False, True]):
    sns.countplot(x="Class", data=df, ax=a)
    a.set_xticks([0, 1])
    a.set_xticklabels(["Normal (0)", "Fraud (1)"])
    a.set_ylabel("Number of transactions")
    a.set_title("Class distribution (log scale)" if use_log else "Class distribution")
    if use_log:
        a.set_yscale("log")
plt.tight_layout()
plt.show()

# In the left plot the fraud bar is too small to see next to the normal bar. In the right plot I used a log scale so both bars are visible, and the gap is still huge. This is why normal supervised classification is hard here, and why I treat fraud as an anomaly and train on normal transactions only.

# ## Amount and Time
# ### Amount
# I compare the transaction Amount of normal and fraud transactions, and look at its shape.

amount_stats = df.groupby("Class")["Amount"].describe().round(2)
amount_stats.index = ["Normal (0)", "Fraud (1)"]
display(amount_stats)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["Amount"], bins=50, ax=ax[0])
ax[0].set_title("Amount (all transactions)")
sns.histplot(np.log1p(df["Amount"]), bins=50, ax=ax[1])
ax[1].set_title("log(1 + Amount)")
sns.boxplot(x="Class", y="Amount", data=df, showfliers=False, ax=ax[2])
ax[2].set_xticklabels(["Normal (0)", "Fraud (1)"])
ax[2].set_title("Amount by class (outliers hidden)")
plt.tight_layout()
plt.show()

med = df.groupby("Class")["Amount"].median()
show(
    f"- Skewness of Amount: **{df['Amount'].skew():.1f}** (above 1 means strongly right skewed)\n"
    f"- Median amount: normal **{med[0]:.2f}**, fraud **{med[1]:.2f}**\n"
    f"- Mean amount: normal **{amount_stats.loc['Normal (0)', 'mean']:.2f}**, fraud **{amount_stats.loc['Fraud (1)', 'mean']:.2f}**\n"
    f"- Largest amount: **{df['Amount'].max():,.2f}**"
)

# Amount is heavily right skewed. Most transactions are small and a few are very large, so the mean is pulled far above the median. After taking log(1 + Amount) the shape is much more balanced. The typical amounts of normal and fraud transactions overlap a lot, so Amount alone cannot separate fraud. The large outliers can hurt distance based methods like LOF, which is why I use RobustScaler later.

# ### Time
# Time is the number of seconds since the first transaction. I convert it to hours to read it easily.

hours = df["Time"] / 3600
hour_of_day = (hours % 24).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(hours, bins=48, ax=ax[0])
ax[0].set_title("Transactions over time")
ax[0].set_xlabel("Hours since first transaction")
sns.histplot(x=hours, hue=df["Class"], bins=48, stat="density", common_norm=False, ax=ax[1])
ax[1].set_title("Normal (0) vs Fraud (1) over time")
ax[1].set_xlabel("Hours since first transaction")
plt.tight_layout()
plt.show()

by_hour = df.groupby(hour_of_day).agg(transactions=("Class", "size"), fraud_rate=("Class", "mean"))
by_hour["fraud_rate"] = by_hour["fraud_rate"] * 100
show(
    f"- The data covers about **{hours.max():.1f} hours**, so roughly two days\n"
    f"- Quietest hour of the day: **{by_hour['transactions'].idxmin()}:00** ({int(by_hour['transactions'].min()):,} transactions)\n"
    f"- Highest fraud rate by hour: **{by_hour['fraud_rate'].idxmax()}:00** ({by_hour['fraud_rate'].max():.2f}% of its transactions)"
)

# The left plot shows a day and night pattern, with quiet periods when fewer transactions happen. In the right plot each class is scaled to 100%, so the shapes can be compared. The fraud rate by hour is noisy because there are so few fraud cases, so I do not read too much into it. Time is only the seconds since the first transaction, not a real clock time.

# ## PCA Features
# V1 to V28 are anonymous features. I first inspect their distributions, then check which ones show the biggest difference between normal and fraud transactions.

v_cols = [f"V{i}" for i in range(1, 29)]

fig, axes = plt.subplots(4, 7, figsize=(20, 10))
for a, col in zip(axes.ravel(), v_cols):
    a.hist(df[col], bins=50)
    a.set_yscale("log")
    a.set_title(col, fontsize=10)
plt.suptitle("Distribution of the PCA features (log scale on y)", y=1.01)
plt.tight_layout()
plt.show()

heavy = df[v_cols].kurtosis().sort_values(ascending=False)
show(
    f"- Largest absolute mean of any V column: **{df[v_cols].mean().abs().max():.3f}** (all are centred near 0, as PCA outputs are)\n"
    f"- Heaviest tails (kurtosis): **{heavy.index[0]}** ({heavy.iloc[0]:.0f}), **{heavy.index[1]}** ({heavy.iloc[1]:.0f}), **{heavy.index[2]}** ({heavy.iloc[2]:.0f})"
)

# The PCA features are centred near 0 and most are bell shaped, but many have long tails (the log scale on the y-axis makes the tails visible). Those tail points are extreme values, which is what an anomaly detector looks for.

normal_df = df[df["Class"] == 0]
fraud_df = df[df["Class"] == 1]

# how far the fraud average is from the normal average, in normal standard deviations
shift = (fraud_df[v_cols].mean() - normal_df[v_cols].mean()) / normal_df[v_cols].std()
shift_sorted = shift.reindex(shift.abs().sort_values(ascending=False).index)

plt.figure(figsize=(7, 5))
shift_sorted.head(10)[::-1].plot(kind="barh")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Fraud mean minus normal mean (in normal standard deviations)")
plt.title("Top 10 PCA features that separate fraud")
plt.tight_layout()
plt.show()

top6 = shift_sorted.index[:6].tolist()
kde_df = pd.concat([normal_df.sample(min(20000, len(normal_df)), random_state=RANDOM_STATE), fraud_df])
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for a, col in zip(axes.ravel(), top6):
    lo, hi = kde_df[col].quantile([0.005, 0.995])
    sub = kde_df[(kde_df[col] >= lo) & (kde_df[col] <= hi)]
    sns.kdeplot(data=sub, x=col, hue="Class", common_norm=False, ax=a)
    a.set_title(f"{col}: normal (0) vs fraud (1)")
plt.tight_layout()
plt.show()

top_text = ", ".join([f"{c} ({shift[c]:+.1f})" for c in shift_sorted.index[:5]])
show(f"- Features with the biggest shift (in normal standard deviations): {top_text}")

# Fraud transactions sit in a different place from normal ones for several features, so these features carry most of the signal for anomaly detection. In the density plots the fraud curve is shifted and often wider than the normal one, which means fraud is not just different but also more spread out. Because the shift is clear but not perfect, I expect the models to catch a good part of fraud but not all of it.

# ## Scaling with RobustScaler
# Distance based methods like LOF are sensitive to the scale and to outliers. StandardScaler uses the mean and standard deviation, and both are pulled by a few huge amounts. RobustScaler uses the median and the interquartile range (IQR), so extreme values do not distort it. The comparison below is only a demonstration, the real scaler is fitted on the training data later.

amt = df[["Amount"]]
cmp = pd.DataFrame({
    "Original": df["Amount"],
    "StandardScaler": StandardScaler().fit_transform(amt).ravel(),
    "RobustScaler": RobustScaler().fit_transform(amt).ravel(),
})
display(cmp.describe().round(2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=cmp[["StandardScaler", "RobustScaler"]], showfliers=False, ax=ax[0])
ax[0].set_title("Amount after scaling (outliers hidden)")
sns.boxplot(data=cmp[["StandardScaler", "RobustScaler"]], ax=ax[1])
ax[1].set_title("Amount after scaling (outliers shown)")
plt.tight_layout()
plt.show()

iqr = cmp.quantile(0.75) - cmp.quantile(0.25)
show(
    f"- Middle 50% of Amount spans **{iqr['StandardScaler']:.2f}** units after StandardScaler and **{iqr['RobustScaler']:.2f}** units after RobustScaler\n"
    f"- Largest Amount becomes **{cmp['StandardScaler'].max():.1f}** after StandardScaler and **{cmp['RobustScaler'].max():.1f}** after RobustScaler"
)

# After StandardScaler the bulk of the transactions is squeezed into a very narrow band, because the few huge amounts inflate the standard deviation. After RobustScaler the middle 50% of the data spans exactly one unit, so the normal behaviour keeps its spread and the extreme values stay far away, which is what I want for anomaly detection. I apply RobustScaler to Amount and Time. The V columns are already on a PCA scale, so I leave them as they are.

# ## Train, Validation and Test Split
# I split the data into three parts, stratified so each part keeps the same tiny fraud share:
# - **Train (60%):** only the normal rows are used to teach the models what normal looks like.
# - **Validation (20%):** used to try hyperparameters and to choose the threshold.
# - **Test (20%):** used once at the end, so the final numbers are fair.

features = [c for c in df.columns if c != "Class"]
X = df[features]
y = df["Class"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)

split_table = pd.DataFrame({
    "Rows": [len(y_train), len(y_val), len(y_test)],
    "Fraud": [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())],
}, index=["Train", "Validation", "Test"])
split_table["Fraud %"] = (split_table["Fraud"] / split_table["Rows"] * 100).round(3)
display(split_table)

scale_cols = ["Time", "Amount"]
scaler = RobustScaler()
scaler.fit(X_train.loc[y_train == 0, scale_cols])   # fitted on normal training rows only

def apply_scaler(part):
    part = part.copy()
    part[scale_cols] = scaler.transform(part[scale_cols])
    return part

X_train_s = apply_scaler(X_train)
X_val_s = apply_scaler(X_val)
X_test_s = apply_scaler(X_test)
display(X_train_s[scale_cols].describe().round(2))

# The scaler was fitted on the normal training rows only and then applied to all three parts. If I fitted it on all the data, information from the test set would leak into training.

# normal-only data sets
rng = np.random.RandomState(RANDOM_STATE)
train_normal = X_train_s[y_train == 0]
order = rng.permutation(len(train_normal))

CALIB_SIZE = min(20000, len(train_normal) // 4)
LOF_SAMPLE = 20000

calib_normal = train_normal.iloc[order[:CALIB_SIZE]]     # normal rows used only to set thresholds
fit_normal = train_normal.iloc[order[CALIB_SIZE:]]       # normal rows used to train Isolation Forest
lof_fit = fit_normal.sample(n=min(LOF_SAMPLE, len(fit_normal)), random_state=RANDOM_STATE)   # smaller sample for LOF

show(
    f"- Isolation Forest is trained on **{len(fit_normal):,}** normal transactions\n"
    f"- LOF is trained on a sample of **{len(lof_fit):,}** normal transactions, because LOF is slow on large data\n"
    f"- **{len(calib_normal):,}** normal transactions are kept aside to set the score threshold from the contamination value"
)

# Both models learn only from normal transactions. No fraud row is used for training. The labels are used only to measure the results on the validation and test sets.

def threshold_for(c, calib_scores):
    # flag the top c share of normal scores: c is the share of normal transactions we allow to be flagged
    return np.quantile(calib_scores, 1 - c)

def evaluate_flags(y_true, flagged):
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged).astype(bool)
    tp = int(((y_true == 1) & flagged).sum())
    fp = int(((y_true == 0) & flagged).sum())
    fn = int(((y_true == 1) & ~flagged).sum())
    tn = int(((y_true == 0) & ~flagged).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"Flagged": tp + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision, "Recall": recall, "F1": f1, "FPR": fpr}

c_grid = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

def contamination_table(s_val, s_cal, y_true):
    rows = []
    for c in c_grid:
        r = evaluate_flags(y_true, s_val >= threshold_for(c, s_cal))
        rows.append({"contamination": c, **r})
    return pd.DataFrame(rows)

def plot_contamination(tab, title):
    plt.figure(figsize=(7, 4))
    for col in ["Precision", "Recall", "F1"]:
        plt.plot(tab["contamination"], tab[col], marker="o", label=col)
    plt.xscale("log")
    plt.xlabel("Contamination (share of normal transactions flagged)")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.show()

# Anomaly detectors give every transaction an anomaly score (higher means more unusual). The contamination value decides the cut-off: it is the share of normal transactions I am willing to flag. Because the models are trained on normal data only, contamination is the same thing as the false alarm rate on normal transactions.

# ## Isolation Forest
# Isolation Forest is built on the idea that anomalies are few and different. It builds many random trees. Each tree picks a random feature and a random split value again and again, until a single transaction is alone. A fraud transaction is unusual, so it is isolated after only a few splits (a short path). A normal transaction sits inside a crowd, so it needs many more splits (a long path). The anomaly score comes from the average path length over all trees.

t0 = time.time()
iso = IsolationForest(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(fit_normal)
fit_t = time.time() - t0

t0 = time.time()
s_val = -iso.score_samples(X_val_s)     # higher means more anomalous
score_t = time.time() - t0
s_cal = -iso.score_samples(calib_normal)

r = evaluate_flags(y_val, s_val >= threshold_for(0.01, s_cal))
show(
    f"**First run (100 trees, contamination 0.01) on the validation set**\n\n"
    f"- Training time: **{fit_t:.1f} s**\n"
    f"- Precision: **{r['Precision']:.3f}**, recall: **{r['Recall']:.3f}**, F1: **{r['F1']:.3f}**\n"
    f"- Caught **{r['TP']}** of **{int(y_val.sum())}** frauds with **{r['FP']:,}** false alarms\n"
    f"- PR-AUC (threshold free): **{average_precision_score(y_val, s_val):.3f}**, ROC-AUC: **{roc_auc_score(y_val, s_val):.3f}**"
)

# ### Number of trees (n_estimators)
# I train Isolation Forest with a different number of trees and compare the threshold free scores (PR-AUC and ROC-AUC) on the validation set. These do not depend on the contamination value, so they are a fair way to compare. The precision and recall columns use a fixed contamination of 0.01.

rows = []
if_models = {}
if_times = {}
for n in [50, 100, 200, 300]:
    t0 = time.time()
    m = IsolationForest(n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1).fit(fit_normal)
    ft = time.time() - t0
    t0 = time.time()
    sv = -m.score_samples(X_val_s)
    st = time.time() - t0
    sc = -m.score_samples(calib_normal)
    r = evaluate_flags(y_val, sv >= threshold_for(0.01, sc))
    rows.append({"n_estimators": n, "Fit time (s)": ft, "PR-AUC": average_precision_score(y_val, sv),
                 "ROC-AUC": roc_auc_score(y_val, sv), "Precision": r["Precision"],
                 "Recall": r["Recall"], "F1": r["F1"]})
    if_models[n] = (m, sv, sc)
    if_times[n] = (ft, st)

if_n_table = pd.DataFrame(rows).set_index("n_estimators").round(3)
display(if_n_table)

best_n = int(if_n_table["PR-AUC"].idxmax())
show(f"Best number of trees by PR-AUC on the validation set: **{best_n}**")

# More trees make the score more stable, but after a point the gain is tiny while the training time keeps growing. I choose the number of trees with the best PR-AUC, which measures how well the score ranks fraud above normal over all possible thresholds.

# ### Contamination
# Now I keep the best number of trees and change the contamination value, which is the cut-off. A low value flags very few transactions (high precision, low recall). A high value flags many (high recall, low precision).

_, if_s_val, if_s_cal = if_models[best_n]
if_c_table = contamination_table(if_s_val, if_s_cal, y_val)
display(if_c_table.round(3))
plot_contamination(if_c_table, "Isolation Forest: effect of contamination (validation set)")

# As contamination goes up, recall rises because more transactions are flagged, but precision falls because most of the extra flags are normal customers. The best choice is somewhere in the middle, and I pick it properly with the Precision-Recall curve later.

# ## Local Outlier Factor (LOF)
# LOF is a density based method. For each transaction it looks at its nearest neighbours and compares its own local density with the density of those neighbours. A transaction in a sparse area surrounded by denser neighbours gets a high LOF score and is flagged as an anomaly. LOF has to compare every new transaction with the training points, so it gets slow when the training data is large. For that reason I train it on a sample of normal transactions. I use novelty=True, which lets a trained LOF score new, unseen transactions.

t0 = time.time()
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(lof_fit)
fit_t = time.time() - t0

t0 = time.time()
l_val = -lof.score_samples(X_val_s)
score_t = time.time() - t0
l_cal = -lof.score_samples(calib_normal)

r = evaluate_flags(y_val, l_val >= threshold_for(0.01, l_cal))
show(
    f"**First run (20 neighbours, contamination 0.01) on the validation set**\n\n"
    f"- Training time: **{fit_t:.1f} s**, time to score the validation set: **{score_t:.1f} s**\n"
    f"- Precision: **{r['Precision']:.3f}**, recall: **{r['Recall']:.3f}**, F1: **{r['F1']:.3f}**\n"
    f"- Caught **{r['TP']}** of **{int(y_val.sum())}** frauds with **{r['FP']:,}** false alarms\n"
    f"- PR-AUC (threshold free): **{average_precision_score(y_val, l_val):.3f}**, ROC-AUC: **{roc_auc_score(y_val, l_val):.3f}**"
)

# ### Number of neighbours
# The main setting of LOF is n_neighbors, the number of neighbours used to measure the local density. A small value reacts to tiny local changes, a large value looks at a wider area.

rows = []
lof_models = {}
lof_times = {}
for k in [10, 20, 50]:
    t0 = time.time()
    m = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1).fit(lof_fit)
    ft = time.time() - t0
    t0 = time.time()
    lv = -m.score_samples(X_val_s)
    st = time.time() - t0
    lc = -m.score_samples(calib_normal)
    r = evaluate_flags(y_val, lv >= threshold_for(0.01, lc))
    rows.append({"n_neighbors": k, "Fit time (s)": ft, "Score time (s)": st,
                 "PR-AUC": average_precision_score(y_val, lv), "ROC-AUC": roc_auc_score(y_val, lv),
                 "Precision": r["Precision"], "Recall": r["Recall"], "F1": r["F1"]})
    lof_models[k] = (m, lv, lc)
    lof_times[k] = (ft, st)

lof_k_table = pd.DataFrame(rows).set_index("n_neighbors").round(3)
display(lof_k_table)

best_k = int(lof_k_table["PR-AUC"].idxmax())
show(f"Best number of neighbours by PR-AUC on the validation set: **{best_k}**")

_, lof_s_val, lof_s_cal = lof_models[best_k]
lof_c_table = contamination_table(lof_s_val, lof_s_cal, y_val)
display(lof_c_table.round(3))
plot_contamination(lof_c_table, "LOF: effect of contamination (validation set)")

# LOF shows the same trade off as Isolation Forest: a higher contamination raises recall and lowers precision. The next step compares how long the two models take.

# ### Execution time
# The project notes that LOF can be expensive on large data. I time both models with the same growing training size, and score the same 10,000 validation transactions each time.

probe = X_val_s.sample(n=min(10000, len(X_val_s)), random_state=RANDOM_STATE)
sizes = [s for s in [2500, 5000, 10000, 20000] if s <= len(fit_normal)]

rows = []
for sz in sizes:
    sub = fit_normal.sample(n=sz, random_state=RANDOM_STATE)

    t0 = time.time()
    l = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(sub)
    l.score_samples(probe)
    lt = time.time() - t0

    t0 = time.time()
    i = IsolationForest(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(sub)
    i.score_samples(probe)
    it = time.time() - t0

    rows.append({"Training rows": sz, "Isolation Forest (s)": it, "LOF (s)": lt})

time_table = pd.DataFrame(rows).set_index("Training rows").round(2)
display(time_table)

plt.figure(figsize=(6, 4))
plt.plot(time_table.index, time_table["Isolation Forest (s)"], marker="o", label="Isolation Forest")
plt.plot(time_table.index, time_table["LOF (s)"], marker="o", label="LOF")
plt.xlabel("Number of training rows")
plt.ylabel("Time to train and score 10,000 transactions (s)")
plt.title("Execution time")
plt.legend()
plt.show()

big = time_table.iloc[-1]
show(
    f"- With **{time_table.index[-1]:,}** training rows, Isolation Forest took **{big['Isolation Forest (s)']:.2f} s** and LOF took **{big['LOF (s)']:.2f} s**\n"
    f"- LOF's time grows with the training size, while Isolation Forest stays almost flat because each tree only uses a small sub-sample"
)

# Isolation Forest takes about the same time however large the training set is, because each tree uses only a small random sub-sample. LOF has to compare each new transaction with the stored training points, so its time grows as the training set grows. This is why I trained LOF on a sample, and it is a real disadvantage for a system that must score transactions in real time.

# ## Threshold Tuning with Precision-Recall Curves
# The anomaly score alone does not say what to flag, I have to choose a cut-off.
# - **Scenario A, high threshold:** very few transactions are flagged. Precision is high but recall is low, so few false alarms but a lot of missed fraud.
# - **Scenario B, low threshold:** many transactions are flagged. Recall is high but precision is low, so most fraud is caught but many genuine customers are blocked.
#
# The Precision-Recall curve shows every possible cut-off at once. I choose the point with the best F1 score on the validation set, which balances the two.

def best_f1_point(y_true, scores):
    p, r, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    i = int(np.argmax(f1))
    return p, r, thr, i, f1

pr = {}
for name, sc in [("Isolation Forest", if_s_val), ("LOF", lof_s_val)]:
    p, r, thr, i, f1 = best_f1_point(y_val, sc)
    pr[name] = {"p": p, "r": r, "thr": thr[i], "i": i, "f1": f1[i], "scores": sc}

plt.figure(figsize=(7, 5))
for name, d in pr.items():
    plt.plot(d["r"], d["p"], label=f"{name} (PR-AUC {average_precision_score(y_val, d['scores']):.3f})")
    plt.scatter(d["r"][d["i"]], d["p"][d["i"]], s=80, zorder=5)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curves on the validation set (dots = best F1)")
plt.legend()
plt.show()

c_if = round(float(np.clip((if_s_val[y_val.values == 0] >= pr["Isolation Forest"]["thr"]).mean(), 0.0001, 0.5)), 4)
c_lof = round(float(np.clip((lof_s_val[y_val.values == 0] >= pr["LOF"]["thr"]).mean(), 0.0001, 0.5)), 4)

show(
    f"- Isolation Forest: best F1 **{pr['Isolation Forest']['f1']:.3f}**, which flags about **{c_if:.2%}** of normal transactions (contamination **{c_if}**)\n"
    f"- LOF: best F1 **{pr['LOF']['f1']:.3f}**, which flags about **{c_lof:.2%}** of normal transactions (contamination **{c_lof}**)"
)

# A curve that stays closer to the top right corner is better. The dots mark the cut-off with the best F1 for each model. Each dot is turned into a contamination value, which is the share of normal transactions that gets flagged at that cut-off. I use these two values for the final models.

# ## Final Models on the Test Set
# Now I train the final models with the chosen settings and the contamination found above, and evaluate them once on the test set, which was never used to choose anything.

final_if = IsolationForest(n_estimators=best_n, contamination=c_if, random_state=RANDOM_STATE, n_jobs=-1).fit(fit_normal)
final_lof = LocalOutlierFactor(n_neighbors=best_k, novelty=True, contamination=c_lof, n_jobs=-1).fit(lof_fit)

if_flags = final_if.predict(X_test_s) == -1
lof_flags = final_lof.predict(X_test_s) == -1
if_test_scores = -final_if.score_samples(X_test_s)
lof_test_scores = -final_lof.score_samples(X_test_s)

rows = {}
for name, flags, sc in [("Isolation Forest", if_flags, if_test_scores), ("LOF", lof_flags, lof_test_scores)]:
    r = evaluate_flags(y_test, flags)
    r["PR-AUC"] = average_precision_score(y_test, sc)
    r["ROC-AUC"] = roc_auc_score(y_test, sc)
    rows[name] = r

test_table = pd.DataFrame(rows).T
shown = test_table[["Flagged", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "FPR", "PR-AUC", "ROC-AUC"]].copy()
for col in ["Flagged", "TP", "FP", "FN", "TN"]:
    shown[col] = shown[col].astype(int)
display(shown.round(3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, (name, flags) in zip(ax, [("Isolation Forest", if_flags), ("LOF", lof_flags)]):
    cm = pd.crosstab(pd.Series(y_test.values, name="Actual"), pd.Series(flags.astype(int), name="Predicted"))
    cm = cm.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Fraud"])
    a.set_title(f"{name} (test set)")
plt.tight_layout()
plt.show()

# Each box shows a number of transactions. The bottom right box is fraud that was caught. The bottom left box is fraud that was missed, which is the costly mistake because the money is lost. The top right box is normal customers that were wrongly flagged, which are the false alarms that annoy customers.

# ## Where the Anomalies Sit (PCA Scatter Plot)
# To see the detected anomalies, I squeeze the 30 features into two PCA components and plot the test transactions. All fraud cases are shown, together with a random sample of normal ones so the plot stays readable.

pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(fit_normal)
test_2d = pca.transform(X_test_s)

yt = y_test.values
idx_fraud = np.where(yt == 1)[0]
idx_norm = np.where(yt == 0)[0]
idx_norm_s = np.random.RandomState(RANDOM_STATE).choice(idx_norm, size=min(5000, len(idx_norm)), replace=False)

fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
for a, (name, flags) in zip(ax, [("Isolation Forest", if_flags), ("LOF", lof_flags)]):
    n_ok = idx_norm_s[~flags[idx_norm_s]]
    n_fa = idx_norm_s[flags[idx_norm_s]]
    f_hit = idx_fraud[flags[idx_fraud]]
    f_miss = idx_fraud[~flags[idx_fraud]]
    a.scatter(test_2d[n_ok, 0], test_2d[n_ok, 1], s=6, c="lightgrey", label="Normal, not flagged")
    a.scatter(test_2d[n_fa, 0], test_2d[n_fa, 1], s=22, c="orange", label="Normal, flagged (false alarm)")
    a.scatter(test_2d[f_miss, 0], test_2d[f_miss, 1], s=30, c="black", marker="x", label="Fraud, missed")
    a.scatter(test_2d[f_hit, 0], test_2d[f_hit, 1], s=30, c="red", label="Fraud, caught")
    a.set_title(name)
    a.set_xlabel("PCA component 1")
    a.set_ylabel("PCA component 2")
    a.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()

show(f"The two components keep **{pca.explained_variance_ratio_.sum():.0%}** of the variation in the data.")

# In the plot, normal transactions form a dense cloud in the middle. Anomalies tend to sit away from this cloud, and the models flag those points. A 2D picture hides most of the 30 dimensions, so some fraud cases look like they sit inside the normal cloud even though they are different in other features. This is why some fraud is missed.

# ## Performance Analysis Report
# This section compares the two algorithms on the test set and gives a threshold recommendation. All numbers below are calculated by the code from the results above.

winner = "Isolation Forest" if pr["Isolation Forest"]["f1"] >= pr["LOF"]["f1"] else "LOF"
loser = "LOF" if winner == "Isolation Forest" else "Isolation Forest"
w, l = test_table.loc[winner], test_table.loc[loser]

n_if = if_times[best_n]
n_lof = lof_times[best_k]
compare = pd.DataFrame({
    "Precision": test_table["Precision"],
    "Recall": test_table["Recall"],
    "F1": test_table["F1"],
    "False positive rate": test_table["FPR"],
    "PR-AUC": test_table["PR-AUC"],
    "ROC-AUC": test_table["ROC-AUC"],
    "Training rows": [len(fit_normal), len(lof_fit)],
    "Fit time (s)": [n_if[0], n_lof[0]],
    "Time to score validation set (s)": [n_if[1], n_lof[1]],
}).round(3)
display(compare)

show(
    f"**Which algorithm is better?** On the test set, **{winner}** has the better balance with an F1 of **{w['F1']:.3f}** "
    f"(precision **{w['Precision']:.3f}**, recall **{w['Recall']:.3f}**), compared with **{l['F1']:.3f}** for {loser} "
    f"(precision **{l['Precision']:.3f}**, recall **{l['Recall']:.3f}**). "
    f"The threshold free PR-AUC is **{w['PR-AUC']:.3f}** for {winner} and **{l['PR-AUC']:.3f}** for {loser}.\n\n"
    f"**Speed:** Isolation Forest was trained on **{len(fit_normal):,}** rows in **{n_if[0]:.1f} s**. "
    f"LOF could only use **{len(lof_fit):,}** rows and took **{n_lof[0]:.1f} s** to train and **{n_lof[1]:.1f} s** to score the validation set, "
    f"against **{n_if[1]:.1f} s** for Isolation Forest."
)

ctab = if_c_table if winner == "Isolation Forest" else lof_c_table
low, high = ctab.iloc[0], ctab.iloc[-1]
c_rec = c_if if winner == "Isolation Forest" else c_lof
n_test_normal = int((y_test == 0).sum())
n_test_fraud = int((y_test == 1).sum())

show(
    f"**Scenario A, high threshold (contamination {low['contamination']}):** precision **{low['Precision']:.3f}** and recall **{low['Recall']:.3f}**. "
    f"Only **{int(low['Flagged']):,}** transactions are flagged, so false alarms are rare, but any fraud below the cut-off is missed.\n\n"
    f"**Scenario B, low threshold (contamination {high['contamination']}):** precision **{high['Precision']:.3f}** and recall **{high['Recall']:.3f}**. "
    f"**{int(high['Flagged']):,}** transactions are flagged, so more fraud is caught, but many genuine customers are blocked.\n\n"
    f"**Threshold recommendation:** I recommend using **{winner}** with a contamination of **{c_rec}**. "
    f"On the unseen test set it caught **{int(w['TP'])}** of **{n_test_fraud}** frauds (recall **{w['Recall']:.1%}**) with a precision of **{w['Precision']:.1%}**. "
    f"It raised **{int(w['FP'])}** false alarms out of **{n_test_normal:,}** normal transactions (false positive rate **{w['FPR']:.2%}**), "
    f"which is about **{w['FPR'] * 10000:.0f}** wrongly flagged customers for every 10,000 normal transactions."
)

# ### How the business should use this
# The best contamination depends on what a missed fraud costs compared with a blocked customer. If a fraud costs far more than a false alarm, the company can accept a higher contamination and flag more transactions. A good way to use the flagged transactions is to ask for extra verification (for example an OTP) instead of blocking them straight away, so a false alarm costs little.
#
# ### Limitations
# - The V features are anonymous PCA features, so I cannot explain in business words why a transaction was flagged.
# - The data covers only two days, and there are very few fraud cases, so the precision and recall numbers can change a lot with a different split.
# - LOF was trained on a sample of normal transactions, because of its running time, so it did not see all the normal data like Isolation Forest did.
# - Fraud patterns change over time, so the models should be retrained regularly and the threshold checked again.

# ## Conclusion
# In this project I explored the credit card transactions data, showed how extreme the class imbalance is, and used RobustScaler so that outliers do not distort the models. I trained Isolation Forest and Local Outlier Factor on normal transactions only, tuned their settings on a validation set, and used the Precision-Recall curve to choose a threshold. I evaluated the final models once on a separate test set with confusion matrices and a PCA scatter plot. The comparison and the final contamination recommendation, with the exact numbers, are in the Performance Analysis Report above.
