import pandas as pd

word = "NVDA,"

# Create a DataFrame with 20000 rows
df = pd.DataFrame({'Word_Column': [word] * 24464})

# Save to CSV
df.to_csv('ticker.csv', index=False)

print("CSV saved successfully!")
