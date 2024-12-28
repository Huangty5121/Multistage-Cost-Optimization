import math
from scipy.stats import norm, binom

# Calculate sample size: n = (z^2 * p * (1 - p)) / (e^2)
def calculate_sample_size(p: float, z: float, e: float) -> int:
    n = (z**2 * p * (1 - p)) / (e**2)
    print(f"Sample size (before rounding): {n}")
    return math.ceil(n)

# Calculate Z value based on confidence level
def calculate_z_value(confidence_level: float) -> float:
    return norm.ppf(1 - (1 - confidence_level) / 2) 

# Calculate rejection and acceptance critical values
def calculate_critical_values(n: int, p: float, confidence_level: float) -> tuple:
    alpha = 1 - confidence_level # Significance level
    return binom.ppf(1 - alpha, n, p), binom.ppf(alpha, n, p) 

# Sampling plan implementation
def sampling_plan(p_list: list, e_list: list, confidence_levels: list):
    for p in p_list:
        for e in e_list:
            for confidence_level in confidence_levels:
                z_value = calculate_z_value(confidence_level)
                sample_size = calculate_sample_size(p, z_value, e)
                print(f"For {int(confidence_level * 100)}% confidence, defect rate {p*100}% and error margin {e*100}%, minimum sample size: {sample_size} items")
                
                k_reject, k_accept = calculate_critical_values(sample_size, p, confidence_level)
                print(f"For {int(confidence_level * 100)}% confidence:")
                print(f"Rejection critical value: {k_reject}")
                print(f"Acceptance critical value: {k_accept}")
                print("-" * 50)

# Main function to run the sampling plan calculations
def main():
    p_list = [0.1, 0.15] 
    e_list = [0.05, 0.1]
    confidence_levels = [0.95, 0.90]
    
    sampling_plan(p_list, e_list, confidence_levels)

if __name__ == "__main__":
    main()
