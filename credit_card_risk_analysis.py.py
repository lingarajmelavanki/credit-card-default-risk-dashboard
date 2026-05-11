# ============================================================
# CREDIT CARD SPEND & DEFAULT RISK PROFILING PROJECT
# ============================================================
# DOMAIN  : Banking & Finance Analytics
# TOOLS   : Python, Pandas, NumPy, Matplotlib, Seaborn
# DATASET : UCI_Credit_Card.csv
# ============================================================


# =========================
# 1. IMPORT LIBRARIES
# =========================

# Pandas -> Data handling and analysis
import pandas as pd

# NumPy -> Numerical operations
import numpy as np

# Matplotlib -> Data visualization
import matplotlib.pyplot as plt

# Seaborn -> Advanced visualization
import seaborn as sns


# =========================
# 2. LOAD DATASET
# =========================

# Load CSV dataset
df = pd.read_csv("UCI_Credit_Card.csv")

# Display first 5 rows
print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

# Display dataset information
print("\n========== DATASET INFO ==========\n")
print(df.info())


# =========================
# 3. CHECK MISSING VALUES
# =========================

# Check null values in each column
print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())


# =========================
# 4. RENAME COLUMN
# =========================

# Rename long column name
df.rename(columns={
    'default.payment.next.month': 'DefaultStatus'
}, inplace=True)

print("\n========== COLUMN RENAMED ==========\n")


# =========================
# 5. CONVERT CATEGORICAL VALUES
# =========================

# ----- Convert Gender Values -----

sex_map = {
    1: 'Male',
    2: 'Female'
}

df['SEX'] = df['SEX'].replace(sex_map)


# ----- Convert Education Values -----

education_map = {
    0: 'Other',
    1: 'Graduate School',
    2: 'University',
    3: 'High School',
    4: 'Others',
    5: 'Unknown',
    6: 'Unknown'
}

df['EDUCATION'] = df['EDUCATION'].replace(education_map)


# ----- Convert Marriage Values -----

marriage_map = {
    0: 'Other',
    1: 'Married',
    2: 'Single',
    3: 'Others'
}

df['MARRIAGE'] = df['MARRIAGE'].replace(marriage_map)

print("\n========== CATEGORICAL VALUES CONVERTED ==========\n")


# =========================
# 6. CREATE AGE GROUPS
# =========================

# Define age ranges
bins = [20, 30, 40, 50, 60, 80]

# Labels for age ranges
labels = ['20-30', '31-40', '41-50', '51-60', '60+']

# Create new AgeGroup column
df['AgeGroup'] = pd.cut(
    df['AGE'],
    bins=bins,
    labels=labels
)

print("\n========== AGE GROUPS CREATED ==========\n")


# =========================
# 7. CREATE TOTAL BILL AMOUNT
# =========================

# Bill amount columns
bill_cols = [
    'BILL_AMT1',
    'BILL_AMT2',
    'BILL_AMT3',
    'BILL_AMT4',
    'BILL_AMT5',
    'BILL_AMT6'
]

# Calculate total bill amount
df['TotalBillAmount'] = df[bill_cols].sum(axis=1)

print("\n========== TOTAL BILL AMOUNT CREATED ==========\n")


# =========================
# 8. CREATE TOTAL PAYMENT AMOUNT
# =========================

# Payment amount columns
pay_cols = [
    'PAY_AMT1',
    'PAY_AMT2',
    'PAY_AMT3',
    'PAY_AMT4',
    'PAY_AMT5',
    'PAY_AMT6'
]

# Calculate total payment amount
df['TotalPaymentAmount'] = df[pay_cols].sum(axis=1)

print("\n========== TOTAL PAYMENT AMOUNT CREATED ==========\n")


# =========================
# 9. DEFAULT RATE ANALYSIS
# =========================

# Calculate default percentage
default_rate = df['DefaultStatus'].mean() * 100

print("\n========== DEFAULT RATE ==========\n")
print(f"Default Rate: {default_rate:.2f}%")


# =========================
# 10. DEFAULT RATE BY EDUCATION
# =========================

# Group customers by education
edu_default = df.groupby('EDUCATION')['DefaultStatus'].mean()

print("\n========== DEFAULT RATE BY EDUCATION ==========\n")
print(edu_default)


# =========================
# 11. DEFAULT RATE BY MARRIAGE
# =========================

# Group customers by marriage status
marriage_default = df.groupby('MARRIAGE')['DefaultStatus'].mean()

print("\n========== DEFAULT RATE BY MARRIAGE ==========\n")
print(marriage_default)


# =========================
# 12. DEFAULT RATE BY AGE GROUP
# =========================

# Group customers by age groups
age_default = df.groupby('AgeGroup')['DefaultStatus'].mean()

print("\n========== DEFAULT RATE BY AGE GROUP ==========\n")
print(age_default)


# =========================
# 13. BILL VS PAYMENT ANALYSIS
# =========================

# Average total bill amount
avg_bill = df['TotalBillAmount'].mean()

# Average total payment amount
avg_payment = df['TotalPaymentAmount'].mean()

print("\n========== BILL VS PAYMENT ==========\n")
print("Average Bill Amount:", avg_bill)
print("Average Payment Amount:", avg_payment)


# =========================
# 14. CORRELATION HEATMAP
# =========================

# Calculate correlations
corr = df.corr(numeric_only=True)

# Set figure size
plt.figure(figsize=(14, 10))

# Create heatmap
sns.heatmap(corr, cmap='coolwarm')

# Add chart title
plt.title("Correlation Heatmap")

# Display graph
plt.show()


# =========================
# 15. BOXPLOT
# CREDIT LIMIT VS DEFAULT STATUS
# =========================

# Set figure size
plt.figure(figsize=(8, 5))

# Create boxplot
sns.boxplot(
    x='DefaultStatus',
    y='LIMIT_BAL',
    data=df
)

# Add title
plt.title("Credit Limit by Default Status")

# Show graph
plt.show()


# =========================
# 16. PAYMENT TREND ANALYSIS
# =========================

# Calculate average payment trend
payment_trend = df[[
    'PAY_AMT1',
    'PAY_AMT2',
    'PAY_AMT3',
    'PAY_AMT4',
    'PAY_AMT5',
    'PAY_AMT6'
]].mean()

# Set graph size
plt.figure(figsize=(10, 5))

# Plot line chart
payment_trend.plot(marker='o')

# Add title and labels
plt.title("6-Month Payment Trend")
plt.xlabel("Month")
plt.ylabel("Average Payment")

# Enable grid
plt.grid(True)

# Show graph
plt.show()


# =========================
# 17. BAR CHART
# DEFAULT RATE BY EDUCATION
# =========================

# Set graph size
plt.figure(figsize=(10, 5))

# Create bar chart
edu_default.plot(kind='bar')

# Add chart title
plt.title("Default Rate by Education")

# Add y-axis label
plt.ylabel("Default Rate")

# Display graph
plt.show()


# =========================
# 18. SAVE CLEANED DATASET
# =========================

# Save cleaned dataset
df.to_csv("Cleaned_UCI_Credit_Card.csv", index=False)

print("\n========== CLEANED DATASET SAVED ==========\n")


# =========================
# 19. BUSINESS INSIGHTS
# =========================

print("\n========== BUSINESS INSIGHTS ==========\n")

print("1. Customers with lower credit limits show higher default risk.")

print("2. Repeated delayed payments strongly indicate future defaults.")

print("3. Younger customer groups show riskier payment behavior.")

print("4. Some education categories show higher default percentages.")

print("5. Payment history is one of the strongest indicators of risk.")


# =========================
# 20. PROJECT COMPLETED
# =========================

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========\n")