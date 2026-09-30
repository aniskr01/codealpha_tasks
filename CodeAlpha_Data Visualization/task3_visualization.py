import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("titanic_eda_dataset.csv")

# Create an age-group feature for analysis
df["Age Group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, np.inf],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
    include_lowest=True
)

# 1. Survival distribution
counts = df["survived"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(["Did Not Survive", "Survived"], counts.values)
ax.set_title("Passenger Survival Distribution")
ax.set_ylabel("Number of Passengers")
for i, v in enumerate(counts.values):
    ax.text(i, v + max(counts.values)*0.02, str(v), ha="center")
fig.tight_layout()
fig.savefig("visualizations/01_survival_distribution.png", dpi=200)
plt.close()

# 2. Survival by gender
rates = df.groupby("sex")["survived"].mean().mul(100)
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(rates.index, rates.values)
ax.set_title("Survival Rate by Gender")
ax.set_ylabel("Survival Rate (%)")
ax.set_ylim(0,110)
for i, v in enumerate(rates.values):
    ax.text(i, v+3, f"{v:.1f}%", ha="center")
fig.tight_layout()
fig.savefig("visualizations/02_survival_by_gender.png", dpi=200)
plt.close()

# 3. Survival by passenger class
rates = df.groupby("pclass")["survived"].mean().mul(100)
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(rates.index.astype(str), rates.values)
ax.set_title("Survival Rate by Passenger Class")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Survival Rate (%)")
ax.set_ylim(0,110)
for i, v in enumerate(rates.values):
    ax.text(i, v+3, f"{v:.1f}%", ha="center")
fig.tight_layout()
fig.savefig("visualizations/03_survival_by_class.png", dpi=200)
plt.close()

# 4. Age distribution
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(df["age"], bins=8, edgecolor="black")
ax.set_title("Passenger Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Number of Passengers")
fig.tight_layout()
fig.savefig("visualizations/04_age_distribution.png", dpi=200)
plt.close()

# 5. Fare by class
groups = [df.loc[df["pclass"] == c, "fare"].values
          for c in sorted(df["pclass"].unique())]
fig, ax = plt.subplots(figsize=(8,5))
ax.boxplot(groups, labels=[str(c) for c in sorted(df["pclass"].unique())])
ax.set_title("Fare Distribution by Passenger Class")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Fare")
fig.tight_layout()
fig.savefig("visualizations/05_fare_by_class.png", dpi=200)
plt.close()

# 6. Correlation matrix
numeric = df[["survived", "pclass", "age", "fare"]].corr()
fig, ax = plt.subplots(figsize=(7,6))
im = ax.imshow(numeric.values, interpolation="nearest")
ax.set_xticks(range(len(numeric.columns)))
ax.set_yticks(range(len(numeric.columns)))
ax.set_xticklabels(numeric.columns)
ax.set_yticklabels(numeric.columns)
ax.set_title("Correlation Matrix")
for i in range(len(numeric.columns)):
    for j in range(len(numeric.columns)):
        ax.text(j, i, f"{numeric.iloc[i,j]:.2f}", ha="center", va="center")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig("visualizations/06_correlation_matrix.png", dpi=200)
plt.close()

print("Task 3 visualizations generated successfully.")
