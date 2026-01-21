import pandas as pd
df = pd.read_csv('multi_condition_samples.csv')
df['status'] = 'pending'
df.to_csv('multi_condition_samples.csv', index=False)
print("Reset 792 samples to pending")