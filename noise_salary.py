import numpy as np
import pandas as pd

# Sample salary data
df = pd.DataFrame({
    'position': ['Java Dev', 'Data Scientist', 'DevOps'],
    'base_salary': [30000, 40000, 35000]
})

def add_symmetric_exponential_noise(salary, scale_ratio=0.1, clip_ratio=0.3):
    """
    Adds symmetric exponential noise (positive or negative) to salary.

    Parameters:
        salary: base salary (int or float)
        scale_ratio: mean of exponential noise relative to salary
        clip_ratio: max change allowed relative to salary

    Returns:
        salary with noise
    """
    scale = salary * scale_ratio
    print("scale:" + str(scale))
    noise = np.random.exponential(scale)
    print("noise:" + str(noise))
    # Randomly flip sign
    if np.random.rand() < 0.5:
        noise = -noise

    # Clip noise to avoid extreme jumps
    noise = np.clip(noise, -salary * clip_ratio, salary * clip_ratio)

    return round(salary + noise)

# Apply to dataset
df['salary_with_noise'] = df['base_salary'].apply(lambda s: add_symmetric_exponential_noise(s))

print(df)