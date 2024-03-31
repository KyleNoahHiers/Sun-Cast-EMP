import pandas as pd
import numpy as np

def prep_load_file(load_file):
    # Make the usage column called load and make it be a subtraction of the column after it and itself
    load_file['load'] = -1 * load_file["Usage [kWh]"].diff()

    # Keep only 'Date & Time' and 'load' columns
    load_file = load_file[['Date & Time', 'load']]

    # Convert 'Date & Time' into datetime format and split into separate components
    load_file['Date & Time'] = pd.to_datetime(load_file['Date & Time'])
    load_file['Year'] = load_file['Date & Time'].dt.year
    load_file['Month'] = load_file['Date & Time'].dt.month
    load_file['Day'] = load_file['Date & Time'].dt.day
    load_file['Hour'] = load_file['Date & Time'].dt.hour
    load_file['Day of the Week'] = load_file['Date & Time'].dt.dayofweek

    #drop all rows that arent between 6am and 6pm
    load_file = load_file[load_file['Hour'] >= 6]
    load_file = load_file[load_file['Hour'] <= 18]

    # Implement cyclic encoding for Month, Day, Hour, and Day of the Week
    load_file['Month_sin'] = np.sin(2 * np.pi * load_file['Month'] / 12)
    load_file['Month_cos'] = np.cos(2 * np.pi * load_file['Month'] / 12)

    # Note: For Day, we calculate sin and cos using max day of each month, taking leap years into account
    days_in_month = load_file['Date & Time'].dt.days_in_month
    load_file['Day_sin'] = np.sin(2 * np.pi * load_file['Day'] / days_in_month)
    load_file['Day_cos'] = np.cos(2 * np.pi * load_file['Day'] / days_in_month)

    load_file['Hour_sin'] = np.sin(2 * np.pi * load_file['Hour'] / 24)
    load_file['Hour_cos'] = np.cos(2 * np.pi * load_file['Hour'] / 24)

    load_file['Day of the Week_sin'] = np.sin(2 * np.pi * load_file['Day of the Week'] / 7)
    load_file['Day of the Week_cos'] = np.cos(2 * np.pi * load_file['Day of the Week'] / 7)

    # Delete the original 'Date & Time', 'Year', 'Month', 'Day', 'Hour', 'Day of the Week' columns
    load_file.drop(['Date & Time', 'Year', 'Month', 'Day', 'Hour', 'Day of the Week'], axis=1, inplace=True)



    # Drop the rows with null 'load'
    load_file = load_file.dropna()

    # Make 'load' an integer
    load_file['load'] = load_file['load'].astype(int)

    # Save the modified DataFrame
    load_file.to_csv('load_file_encoded.csv', index=False)
