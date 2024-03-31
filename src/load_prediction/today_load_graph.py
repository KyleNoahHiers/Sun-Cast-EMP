import pandas as pd
import matplotlib.pyplot as plt

#read graph_comparison.csv

df = pd.read_csv('../load_prediction/week_of_17.csv')
print(df.columns)
df['Date & Time'] = pd.to_datetime(df['Date & Time'])
#exclude all dates that after 23rd
df = df[df['Date & Time'] < '2024-03-24']
print(df.head())

df['load'] = -1 * df["Usage [kWh]"].diff()

df["Date & Time"] = pd.to_datetime(df["Date & Time"])
df["Hour"] = df["Date & Time"].dt.hour
df["Day"] = df["Date & Time"].dt.day
#exclude all rows whose hor is not between 6 and 18
df = df[(df['Hour'] >= 6) & (df['Hour'] <= 18)]

predicted_df= pd.read_csv('../load_prediction/predictions.csv')
#merge the rows of predicted and df with the same day and hour
predicted_df['day'] = predicted_df['day'].astype(int)
predicted_df['hour'] = predicted_df['hour'].astype(int)
df['Day'] = df['Day'].astype(int)
df['Hour'] = df['Hour'].astype(int)
predicted_df = predicted_df.rename(columns={'day': 'Day', 'hour': 'Hour'})
predicted_df = predicted_df.astype(int)
df = pd.merge(df, predicted_df, on=['Day', 'Hour'], how='left')
#exclude all rows



df['predicted_load'] = predicted_df['prediction'];
# add predicted load to df such that the first value represents the load at 6am and the last value represents the
# load at 6pm

df['Date & Time'] = pd.to_datetime(df['Date & Time'])
df['Hour'] = df['Date & Time'].dt.hour

#make an integer index for the dataframe
df = df.reset_index()

plt.bar(df.index, df['load'], color='b', label='Actual Load')
plt.plot(df.index, df['predicted_load'], color='r', marker='o', label='Predicted Load')
plt.xlabel('Date & Time')
plt.ylabel('Load')
plt.title('Actual vs Predicted Load')
plt.legend()
plt.show()





