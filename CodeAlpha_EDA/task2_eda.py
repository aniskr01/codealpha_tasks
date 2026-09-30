import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

sns.set_theme(style="whitegrid")
df = pd.read_csv("titanic_eda_dataset.csv")

print("CODEALPHA TASK 2 - EXPLORATORY DATA ANALYSIS")
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nDescriptive statistics:\n", df.describe(include="all").transpose())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

eda = df.drop_duplicates().copy()
for c in ["age", "fare"]:
    if c in eda.columns:
        eda[c] = eda[c].fillna(eda[c].median())
if "embarked" in eda.columns and eda["embarked"].isnull().any():
    eda["embarked"] = eda["embarked"].fillna(eda["embarked"].mode().iloc[0])

eda["age_group"] = pd.cut(
    eda["age"], [0,12,18,35,60,np.inf],
    labels=["Child","Teen","Young Adult","Adult","Senior"],
    include_lowest=True
)

print("\nOverall survival rate:", round(eda["survived"].mean()*100,2), "%")
print("\nSurvival rate by sex (%):")
print((eda.groupby("sex")["survived"].mean()*100).round(2))
print("\nSurvival rate by class (%):")
print((eda.groupby("pclass")["survived"].mean()*100).round(2))

table = pd.crosstab(eda["sex"], eda["survived"])
chi2, p, dof, expected = chi2_contingency(table)
print("\nChi-square test: sex vs survival")
print("Chi-square:", round(chi2,4))
print("p-value:", round(p,6))
print("At alpha=0.05:", "statistically significant" if p < 0.05 else "not statistically significant")

fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(data=eda, x="survived", ax=ax)
ax.set_title("Passenger Survival Distribution")
ax.set_xlabel("Survived (0=No, 1=Yes)")
plt.tight_layout(); plt.savefig("01_survival_distribution.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=eda, x="sex", y="survived", errorbar=None, ax=ax)
ax.set_title("Survival Rate by Sex"); ax.set_ylabel("Survival Rate")
ax.set_ylim(0,1)
plt.tight_layout(); plt.savefig("02_survival_by_sex.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=eda, x="pclass", y="survived", errorbar=None, ax=ax)
ax.set_title("Survival Rate by Passenger Class"); ax.set_ylabel("Survival Rate")
ax.set_ylim(0,1)
plt.tight_layout(); plt.savefig("03_survival_by_class.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(data=eda, x="age", hue="survived", kde=True, bins=30, ax=ax)
ax.set_title("Age Distribution by Survival")
plt.tight_layout(); plt.savefig("04_age_distribution.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(data=eda, x="pclass", y="fare", ax=ax)
ax.set_title("Fare Distribution by Passenger Class")
plt.tight_layout(); plt.savefig("05_fare_outliers.png", dpi=200); plt.close()

num = eda.select_dtypes(include=np.number)
plt.figure(figsize=(8,6))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout(); plt.savefig("06_correlation_matrix.png", dpi=200); plt.close()

print("\nEDA complete. Six visualizations were saved.")
