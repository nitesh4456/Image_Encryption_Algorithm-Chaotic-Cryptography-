import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math


# 🔹 Entropy
def entropy_test(series):
    keys = [int(x * 256) for x in series]
    count = Counter(keys)
    total = len(keys)
    entropy = -sum((c/total) * math.log2(c/total) for c in count.values())
    return entropy


# 🔹 Autocorrelation (lag 1)
def autocorrelation_value(series):
    z = np.array(series)
    z_mean = np.mean(z)
    num = np.sum((z[:-1] - z_mean) * (z[1:] - z_mean))
    den = np.sum((z - z_mean) ** 2)
    return num / den


# 🔹 Correlation coefficient
def correlation_coeff(series):
    x = np.array(series[:-1])
    y = np.array(series[1:])
    return np.corrcoef(x, y)[0, 1]


# 🔹 MASTER TEST FUNCTION
def test_randomness(series):
    print("========== RANDOMNESS TEST REPORT ==========\n")

    arr = np.array(series)

    # 🔸 Mean
    mean = np.mean(arr)
    print("Mean:")
    print(f"  Value: {mean:.5f}")
    print("  Ideal: 0.5")
    print("  Accepted: 0.48 – 0.52")
    if 0.48 <= mean <= 0.52:
        print("  Result: PASS\n")
    else:
        print("  Result: FAIL\n")

    # 🔸 Variance
    variance = np.var(arr)
    print("Variance:")
    print(f"  Value: {variance:.5f}")
    print("  Ideal: 0.0833")
    print("  Accepted: 0.07 – 0.09")
    if 0.07 <= variance <= 0.09:
        print("  Result: PASS\n")
    else:
        print("  Result: FAIL\n")

    # 🔸 Entropy
    entropy = entropy_test(series)
    print("Entropy:")
    print(f"  Value: {entropy:.5f}")
    print("  Ideal: 8")
    print("  Accepted: ≥ 7.5 (≥7.9 excellent)")
    if entropy >= 7.9:
        print("  Result: EXCELLENT\n")
    elif entropy >= 7.5:
        print("  Result: PASS (GOOD)\n")
    else:
        print("  Result: FAIL\n")

    # 🔸 Autocorrelation
    auto = autocorrelation_value(series)
    print("Autocorrelation (lag 1):")
    print(f"  Value: {auto:.5f}")
    print("  Ideal: 0")
    print("  Accepted: |value| < 0.05")
    if abs(auto) < 0.05:
        print("  Result: PASS\n")
    else:
        print("  Result: FAIL\n")

    # 🔸 Correlation
    corr = correlation_coeff(series)
    print("Correlation (x_n vs x_n+1):")
    print(f"  Value: {corr:.5f}")
    print("  Ideal: 0")
    print("  Accepted: -0.05 to 0.05")
    if -0.05 <= corr <= 0.05:
        print("  Result: PASS\n")
    else:
        print("  Result: FAIL\n")

    # 🔸 Histogram
    print("Histogram:")
    print("  Ideal: Uniform (flat distribution)")
    print("  Check visually in the plot")

    plt.figure()
    plt.hist(series, bins=50)
    plt.title("Histogram")
    plt.show()

    # 🔸 Correlation Plot
    plt.figure()
    plt.scatter(series[:-1], series[1:], s=1)
    plt.title("Correlation Plot")
    plt.show()

    print("\n========== TESTING COMPLETE ==========")