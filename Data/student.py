import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================
# 1. READ DATA
# =========================

math = pd.read_csv("week7/student-mat.csv", sep=";")
por = pd.read_csv("week7/student-por.csv", sep=";")

# =========================
# 2. MERGE DATASET
# =========================

cols = [
    "school","sex","age","address","famsize","Pstatus",
    "Medu","Fedu","Mjob","Fjob","reason","nursery","internet"
]

data = pd.merge(math, por, on=cols)

print("Math dataset:", len(math))
print("Portuguese dataset:", len(por))
print("Merged dataset:", len(data))

# lưu file merge
data.to_csv("student-merge.csv", index=False)

# =========================
# 3. DATA CLEANING
# =========================

print("\nMissing values:")
print(data.isnull().sum())

print("\nDataset summary:")
print(data.describe())

# =========================
# 4. DATA ANALYSIS
# =========================

# trung bình điểm
print("\nAverage Math Final Grade:", data["G3_x"].mean())
print("Average Portuguese Final Grade:", data["G3_y"].mean())

# =========================
# 5. VISUALIZATION
# =========================

# Histogram điểm toán
plt.figure()
plt.hist(data["G3_x"], bins=20)
plt.title("Distribution of Math Final Grade (G3)")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.show()

# Histogram điểm tiếng Bồ
plt.figure()
plt.hist(data["G3_y"], bins=20)
plt.title("Distribution of Portuguese Final Grade (G3)")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.show()

# Scatter plot G1 vs G3
plt.figure()
plt.scatter(data["G1_x"], data["G3_x"])
plt.title("G1 vs G3 (Math)")
plt.xlabel("First Period Grade")
plt.ylabel("Final Grade")
plt.show()

# =========================
# 6. LINEAR REGRESSION
# =========================

# sử dụng G1 và G2 để dự đoán G3
X = data[["G1_x", "G2_x"]]
y = data["G3_x"]

model = LinearRegression()
model.fit(X, y)

print("\nRegression Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# dự đoán
pred = model.predict(X)

# =========================
# 7. PLOT REGRESSION
# =========================

plt.figure()
plt.scatter(y, pred)
plt.title("Actual vs Predicted Final Grade")
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.show()

# =========================
# 8. CORRELATION MATRIX
# =========================

corr = data[["G1_x","G2_x","G3_x"]].corr()

print("\nCorrelation Matrix:")
print(corr)

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Matrix")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()