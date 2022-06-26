import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import requests

# Load data
url = "https://api.covid19api.com/country/singapore/status/confirmed"

payload={}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)
load_data = json.loads(response.text)

# Extract number of daily cases and date
total_cases = [x['Cases'] for x in load_data]
daily_cases = np.array(total_cases[1:]) - np.array(total_cases[:-1])
dates = [x['Date'][:7] for x in load_data[1:]]

df = pd.DataFrame({'Daily cases': daily_cases, 'Date': dates})
df.plot(x='Date', y='Daily cases', xticks=range(0, len(df), 200), color='b')
plt.plot(df['Daily cases'].rolling(7).mean(), 'r', linewidth=2)
plt.legend(['Daily cases', '7 days average'])
plt.show()

