# This file should take an input of an egauge file and a weather file and output a csv file that can be used as input to the neural network
# sepcifically the input file should match each each hours prediction from the 6am forecast of a weather file to the actual energy production of the egauge file

import pandas as pd
import numpy as np
import webbrowser









    #drop all columns that are not needed\


def prep_weather_file(weather_file):
    weather = pd.read_csv(weather_file)
    #delete Query Time (computer local) column


    for col in weather.columns:
        try:
            if "Pressure" in col or "Wind" in col or "Humidity" in "col" or "Description" in col or "Current UTC" in col or "Daily" in col or "Current" in col:
                weather.drop(col, axis=1, inplace=True)
            elif "Hourly Forecast UTC" in col:
                if int(weather[col][0]) != 1:
                    weather.drop(col, axis=1, inplace=True)
            elif int(weather[col][0]) >= 27:
                weather.drop(col, axis=1, inplace=True)
        except Exception as e:
            weather.drop(col, axis=1, inplace=True)

    weather = weather.iloc[1:]
    print(weather.columns)
    #remove all null values in Hourly Forecast UT
    # Iterate through each row and try to convert 'Hourly Forecast UTC' to datetime
    for index, row in weather.iterrows():
            try:
                # Attempt to convert 'Hourly Forecast UTC' to datetime
                weather.at[index, 'Hourly Forecast UTC'] = pd.to_datetime(row['Hourly Forecast UTC'])
            except ValueError:
                # If conversion fails, drop the row
                weather.drop(index, inplace=True)


    weather.dropna(subset=['Hourly Forecast UTC'], inplace=True)
    # Convert 'Hourly Forecast UTC' to datetime
    weather['Hourly Forecast UTC'] = pd.to_datetime(weather['Hourly Forecast UTC'])


    weather['Hourly Forecast UTC'] = weather['Hourly Forecast UTC'] + pd.to_timedelta(weather.groupby('Hourly Forecast UTC').cumcount(), unit='h')
    #check if tz aware
    if weather['Hourly Forecast UTC'].dt.tz is None:
        #localize to utc
        weather['Hourly Forecast UTC'] = weather['Hourly Forecast UTC'].dt.tz_localize('UTC')
    weather['Hourly Forecast UTC'] = weather['Hourly Forecast UTC'].dt.tz_convert('America/Chicago')
    try:
        weather['Hourly Forecast UTC'] = pd.to_datetime(weather['Hourly Forecast UTC'], errors='coerce')
        weather = weather.dropna(subset=['Hourly Forecast UTC'])
        #exclude all rows with hourly forecast utc that is not 6
        weather = weather[weather['Hourly Forecast UTC'].dt.hour == 6]
        #print timezone
        print(weather['Hourly Forecast UTC'].dt.tz)
        #print the first 5 rows
        print(weather.head())

    except ValueError:
        pass
    new_weather = pd.DataFrame()

    for index, row in weather.iterrows():
        for i in range(1, 25):  # Looping through hours 1 to 24
            hour_data = {}

            # Only include data from columns for the current hour
            for col in weather.columns:
                if '.' in col:  # Check if the column is for hourly data
                    param, hour_str = col.rsplit('.', 1)  # Split on the last period
                    if hour_str.isdigit():  # Ensure the part after the period is a number
                        hour = int(hour_str)
                        if hour == i:  # Include this column's data if the hour matches
                            hour_data[param] = row[col]

            #add a column for date and time of the forecast by adding the hour to the forecast time
            hour_data['Hourly Forecast UTC'] = row['Hourly Forecast UTC'] + pd.to_timedelta(i, unit='h')

            # Append this hour's data to the new DataFrame
            # To avoid AttributeError for 'append', ensure pandas is correctly installed and imported
            new_hour_data_df = pd.DataFrame([hour_data])
            new_weather = pd.concat([new_weather, new_hour_data_df], ignore_index=True)
            #drop all rows with null values
            new_weather = new_weather.dropna()

    return new_weather

    return weather





# take in a weather file and output a dataframe with the desired formatting in each row.
def prepare_weather_file(weather_file, no_date_or_description = False, columns_to_keep = None, irridance_file = None):
    # Read in the weather file
    weather = pd.read_csv(weather_file)
    #drop columns with null values
    weather = weather.dropna(axis=1)

    for col in weather.columns:

        if col == "Current UTC" in col or "Daily" in col:
            weather.drop(col, axis=1, inplace=True)
        elif "Hourly Forecast UTC" in col:
            if int(weather[col][0]) != 1:
                weather.drop(col, axis=1, inplace=True)
        else:
            try:

                if (int(weather[col][0]) > 24):
                    weather.drop(col, axis=1, inplace=True)
            except TypeError:
                weather.drop(col, axis=1, inplace=True)

    weather = weather.iloc[1:]
    try:
        weather['Hourly Forecast UTC'] = pd.to_datetime(weather['Hourly Forecast UTC'], errors='coerce')
        weather = weather.dropna(subset=['Hourly Forecast UTC'])
        weather['Hourly Forecast UTC'] = weather['Hourly Forecast UTC'].dt.tz_localize('UTC').dt.tz_convert('America/Chicago')
        weather = pd.concat([weather.iloc[:1], weather.iloc[1:][weather.iloc[1:]['Hourly Forecast UTC'].dt.hour == 6]])
    except ValueError:
        pass


    # delete all pressure wind speed and wind direction columns
    for col in weather.columns:
        if "Pressure" in col or "Wind" in col or "Humidity" in col:
            weather.drop(col, axis=1, inplace=True)

    #normalize all columns that contain "uvi" to have a max of 100
    for col in weather.columns:
        if weather[col].dtype in ['int64', 'float64']:
            max_value = round(weather[col].max())
            min_value = round(weather[col].min())
            # Check if already normalized
            if not (min_value == 0 and max_value == 100):
                if max_value != 0:  # Prevent division by zero
                    weather[col] = ((weather[col] / weather[col].max()) * 100).round()

    if columns_to_keep is not None:
        columns_to_retain = [col for col in weather.columns if any(keep in col for keep in columns_to_keep) or "Hourly Forecast UTC" in col]

    return weather
# take in an egauge file and output a dataframe associate each day with total 6am-6pm production
def prepare_egauge_file(egauge_file, solar_column = "Solar Production+ [kWh]"):
    egauge = pd.read_csv(egauge_file, parse_dates=['Date & Time'])

    # Ensure the 'Date & Time' column is parsed as datetime
    egauge['Date'] = egauge['Date & Time'].dt.date
    egauge['Hour'] = egauge['Date & Time'].dt.hour

    # Filter rows for 6 AM and 6 PM
    df_am = egauge[egauge['Hour'] == 6][['Date', solar_column]].set_index('Date')
    df_pm = egauge[egauge['Hour'] == 18][['Date',solar_column]].set_index('Date')

    # Calculate the difference in 'Generation [kWh]' between 6 PM and 6 AM for each day
    production_diff = df_pm[solar_column] - df_am[solar_column]
    production_diff = production_diff.reset_index()
    production_diff.columns = ['Date', 'Production Difference']  # Renaming the difference column
    production_diff = production_diff[production_diff['Production Difference'] >1]
    return production_diff


def label_weather_for_training(weather_file, egauge_file, solar_column = "Solar Production+ [kWh]",label = True):
    weather = prep_weather_file(weather_file)
    egauge = pd.read_csv(egauge_file)

    #convert the solar_column to be the difference of the column after it and the column before it
    egauge['Production Difference'] = -1*egauge[solar_column].diff()
    egauge = egauge.dropna()
    # Ensure the 'Date' column is parsed as datetime
    egauge['Date & Time'] = pd.to_datetime(egauge['Date & Time'])
    #exclude all columns that are not production difference or date
    egauge = egauge[['Date & Time', 'Production Difference']]
    # Merge the DataFrames on the Date column
    #ensure that the date column is in the correct format for both dataframes
    weather['Hourly Forecast UTC'] = pd.to_datetime(weather['Hourly Forecast UTC'])
    # ensure egauge['Date & Time'] is in the correct format
    egauge['Date & Time'] = pd.to_datetime(egauge['Date & Time'])
    #unlocalize the weather data
    weather['Hourly Forecast UTC'] = weather['Hourly Forecast UTC'].dt.tz_localize(None)
    merged_df = pd.merge(weather[['Hourly Forecast UTC']], egauge[['Date & Time', 'Production Difference']], left_on='Hourly Forecast UTC', right_on='Date & Time', how='inner')
    # add an hour month and day column to the dataframe
    merged_df['Hour'] = merged_df['Hourly Forecast UTC'].dt.hour
    merged_df['Month'] = merged_df['Hourly Forecast UTC'].dt.month
    merged_df['Day'] = merged_df['Hourly Forecast UTC'].dt.day
    #cycle encode the month, day, and hour columns
    merged_df['Month_sin'] = np.sin(2 * np.pi * merged_df['Month'] / 12)
    merged_df['Month_cos'] = np.cos(2 * np.pi * merged_df['Month'] / 12)
    merged_df['Day_sin'] = np.sin(2 * np.pi * merged_df['Day'] / 31)
    merged_df['Day_cos'] = np.cos(2 * np.pi * merged_df['Day'] / 31)
    merged_df['Hour_sin'] = np.sin(2 * np.pi * merged_df['Hour'] / 24)
    merged_df['Hour_cos'] = np.cos(2 * np.pi * merged_df['Hour'] / 24)
    #normalize all columns other than production difference and date and time to have a max of 100
    for col in merged_df.columns:
        if merged_df[col].dtype in ['int64', 'float64']:
            max_value = round(merged_df[col].max())
            min_value = round(merged_df[col].min())
            merged_df[col] = ((merged_df[col] / merged_df[col].max()) * 100).round()
    #drop all columns that are not needed
    merged_df = merged_df.drop(columns=['Hourly Forecast UTC', 'Hour', 'Month', 'Day'])
    #drop all rows with production difference less than 0.1
    if label:
        merged_df = merged_df[merged_df['Production Difference'] > 0.01]
    else:
        #delete the production difference column
        merged_df = merged_df.drop(columns=['Production Difference'])



    return merged_df




def combine_and_label(solar_value, weather_df, egauge_df, classification = True):
    # Check if weather_df and egauge_df are DataFrames
    if weather_df is None or egauge_df is None:
        raise ValueError("Input data is None and cannot be processed.")

    # Ensure the 'Date' column is in the correct format for both DataFrames
    weather_df['Date'] = pd.to_datetime(weather_df['Hourly Forecast UTC']).dt.date
    egauge_df['Date'] = pd.to_datetime(egauge_df['Date']).dt.date

    # Merge the DataFrames on the Date column
    merged_df = pd.merge(weather_df[['Date']], egauge_df[['Date', 'Production Difference']], on='Date', how='inner')

    # Add a new column to indicate whether production difference is higher than solar_value
    if classification:
        merged_df['Label'] = merged_df['Production Difference'] > solar_value
    else:
        merged_df['Label'] = merged_df['Production Difference']

    # Create a new DataFrame with the full weather data and the new Label column
    # Note: Since 'Label' is now part of merged_df, we can directly merge merged_df with weather_df
    result_df = pd.merge(weather_df, merged_df[['Date', 'Label']], on='Date', how='inner')

     # Remove 'Date' column and any columns containing the word 'Description'
    result_df.drop(columns=['Hourly Forecast UTC'] + [col for col in result_df if 'Description' in col] + ['Date'], inplace=True)
    # remove all rows with null values
    result_df = result_df.dropna()
    return result_df


#output a irriance file that is formatted correctly


