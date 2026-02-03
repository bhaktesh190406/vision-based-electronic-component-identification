import pandas as pd
import matplotlib.pyplot as plt

# Load results CSV
results_path = "/mnt/data/results.csv"
results = pd.read_csv(results_path)

# Display the first few rows to confirm structure
results.head()
