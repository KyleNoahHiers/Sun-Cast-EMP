import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the CSV file
df = pd.read_csv('../85%/graphable.csv')

#read br_egauge_nov
df2 = pd.read_csv('../src/theNetwork/data/br_test/br_egauge_feb.csv')

# turn its "Solar+ [kWh]" to be the difference between the current and next value
df2['Solar+ [kWh]'] = -1*df2['Solar+ [kWh]'].diff()
#exclude all rows in df2 with Date & Time not between 6am and 6pm\
df2["Date & Time"] = pd.to_datetime(df2["Date & Time"])
df2 = df2[(df2['Date & Time'].dt.hour >= 6) *(df2['Date & Time'].dt.hour < 18)]
#now group the rows by day and sum the "Solar+ [kWh]" for each day but do not sum the "Date & Time" column
#make the "Date & Time" column the index
df2_grouped = df2.groupby(df2["Date & Time"].dt.date)['Solar+ [kWh]'].sum()
#print num rows in df2_grouped
print(len(df2_grouped))




df["Solar+ [kWh]"] = df2["Solar+ [kWh]"]
#make integer day column just literally counting up from 0
df2['integer day'] = np.arange(len(df))
#do a bar graph of the solar vs integer day, if the prediction is 1, then make the bar green, if the prediction is 0, then make the bar red
fig, ax = plt.subplots()
colors = {0:'red', 1:'green'}
ax.bar(df['integer day'], df['Solar+ [kWh]'], color=[colors[i] for i in df['predictions']])
plt.xlabel('Integer Day')
plt.ylabel('Solar+ [kWh]')
plt.title('Solar+ [kWh] vs Integer Day')
plt.show()