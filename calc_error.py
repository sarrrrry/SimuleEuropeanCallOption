import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set(context="paper")

# p = 16.27544888310897
p = 0.1646184399378
# p = 40.18806845336739
df = pd.read_csv("data/european_call_option.csv")
arr = df.values

print("*** S_n array ***")
print(arr)

y = arr[:, 1]
y = np.log10(np.abs(p-arr[:, 1]))
x = np.log10(arr[:, 0])
print("*** y ***")
print(y)
print("*** x ***")
print(x)
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)
print("*** coef ***")
print("inter: ", lr.intercept_)
print("coef : ", lr.coef_)
plt.scatter(x, y)
x_reg = np.linspace(2, 3, 10)
plt.plot(x_reg, x_reg*lr.coef_[0]+lr.intercept_, "r--")
plt.title(f"intercept: {lr.intercept_:.4f} | coefficient: {lr.coef_[0]:.4f}")
plt.ylabel(r"$\log_{10} | p - S_n |$")
plt.xlabel(r"$\log_{10}n$")
plt.savefig("data/q3.eps", bbox_inches="tight", dpi=300)
plt.show()
