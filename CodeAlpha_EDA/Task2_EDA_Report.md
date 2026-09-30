# CodeAlpha Data Analytics Internship - Task 2
## Exploratory Data Analysis (EDA)

### Objective
This project follows the CodeAlpha Task 2 requirements: ask meaningful questions, inspect variables and data types, identify trends/patterns/anomalies, validate assumptions with statistics and visualization, and detect data-quality problems.

### Dataset
**Titanic passenger dataset** (Embedded fallback sample)

Rows: **20**  
Columns: **6**  
Duplicate rows: **0**

### Questions
- What factors are associated with passenger survival?
- Does passenger class relate to survival rate?
- Is survival different between male and female passengers?
- How does age relate to survival?
- Are there missing values, duplicates, or suspicious values?
- Are fare and age distributed evenly, or are there outliers?

### EDA Process
1. Load and inspect the dataset.
2. Check shape, columns, data types and descriptive statistics.
3. Check missing values and duplicates.
4. Handle selected missing values using median/mode imputation.
5. Create age groups.
6. Compare survival rates across sex and passenger class.
7. Perform a chi-square test for sex vs survival.
8. Create visualizations for survival, sex, class, age, fare and correlations.
9. Summarize findings and limitations.

### Key Findings
- Overall survival rate: **55.00%**.
- Survival rate by sex: **{'female': 100.0, 'male': 18.18}**.
- Survival rate by passenger class: **{1: 100.0, 2: 100.0, 3: 10.0}**.
- Chi-square statistic for sex vs survival: **10.2867**.
- Chi-square p-value: **0.00133992**.
- At α = 0.05, the sex-survival association is **statistically significant**.

### Visualizations
- Passenger survival distribution
- Survival rate by sex
- Survival rate by passenger class
- Age distribution by survival
- Fare distribution by passenger class
- Correlation matrix

### Conclusion
The EDA identifies measurable differences in survival across passenger characteristics and highlights data-quality checks needed before downstream analysis or modeling. These findings describe associations in this dataset and should not be interpreted as proof of causation.

### Limitations
- Titanic is a historical dataset and is not a current business population.
- EDA alone cannot establish causality.
- Results depend on available variables and preprocessing decisions.
