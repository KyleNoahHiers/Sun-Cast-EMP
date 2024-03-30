# This file should take an input of an egauge file and a weather file and output a csv file that can be used as input to the neural network
# sepcifically the input file should match each each hours prediction from the 6am forecast of a weather file to the actual energy production of the egauge file

import pandas as pd
import numpy as np
import webbrowser

phrase_encode = {
    'scattered clouds' : 1
    ,'very heavy rain':2,
    'snow':3,
    'mist':4,
    'broken clouds':5,
    'light snow':6,
    'clear sky':7,
    'moderate rain':8,
    'overcast clouds':9,
    'few clouds':10,
    'heavy intensity rain':11,
    'light rain':12}

def merge_egauge_files(file1, file2):
    # Load the eGauge data files
    egauge1 = pd.read_csv(file1)
    egauge2 = pd.read_csv(file2)

    # Concatenate the DataFrames. Assuming 'ignore_index=True' to reset index.
    merged_df = pd.concat([egauge2, egauge1], ignore_index=True)

    # Remove duplicates based on 'Date & Time' column. Keeping the first occurrence.
    merged_df = merged_df.drop_duplicates(subset=['Date & Time'], keep='first')

    # Optionally, sort by 'Date & Time' if the data isn't in chronological order
    merged_df = merged_df.sort_values(by='Date & Time').reset_index(drop=True)

    return merged_df.to_csv('merged_egauge.csv', index=False)

# take in a weather file and output a dataframe with the desired formatting in each row.
def prepare_weather_file(weather_file, no_date_or_description = False, columns_to_keep = None):
    # Read in the weather file
    weather = pd.read_csv(weather_file)

    for col in weather.columns: 

        if col == "Current UTC" in col or "Daily" in col: 
            weather.drop(col, axis=1, inplace=True)
        elif "Hourly Forecast UTC" in col:
            if int(weather[col][0]) != 1:
                weather.drop(col, axis=1, inplace=True)
        else: 
            try:
                if int(weather[col][0]) > 12:
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

    # one hot encode columns with description in the name using get dummies
    #implement this later


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
    for col in weather.columns:
        if(no_date_or_description):
            if "Hourly Forecast UTC" in col or "Description Hourly" in col:
                weather.drop(col, axis=1, inplace=True)


    return weather

# take in an egauge file and output a dataframe associate each day with total 6am-6pm production    
def prepare_egauge_file(egauge_file):
    egauge = pd.read_csv(egauge_file, parse_dates=['Date & Time'])
    
    # Ensure the 'Date & Time' column is parsed as datetime
    egauge['Date'] = egauge['Date & Time'].dt.date
    egauge['Hour'] = egauge['Date & Time'].dt.hour
    
    # Filter rows for 6 AM and 6 PM
    df_am = egauge[egauge['Hour'] == 6][['Date', "Solar Production+ [kWh]"]].set_index('Date')
    df_pm = egauge[egauge['Hour'] == 18][['Date', "Solar Production+ [kWh]"]].set_index('Date')
    
    # Calculate the difference in 'Generation [kWh]' between 6 PM and 6 AM for each day
    production_diff = df_pm["Solar Production+ [kWh]"] - df_am["Solar Production+ [kWh]"]
    production_diff = production_diff.reset_index()
    production_diff.columns = ['Date', 'Production Difference']  # Renaming the difference column
    production_diff = production_diff[production_diff['Production Difference'] >1]
    return production_diff




def combine_and_label(solar_value, weather_df, egauge_df): 
    # Check if weather_df and egauge_df are DataFrames
    if weather_df is None or egauge_df is None:
        raise ValueError("Input data is None and cannot be processed.")

    # Ensure the 'Date' column is in the correct format for both DataFrames
    weather_df['Date'] = pd.to_datetime(weather_df['Hourly Forecast UTC']).dt.date
    egauge_df['Date'] = pd.to_datetime(egauge_df['Date']).dt.date
    
    # Merge the DataFrames on the Date column
    merged_df = pd.merge(weather_df[['Date']], egauge_df[['Date', 'Production Difference']], on='Date', how='inner')

    # Add a new column to indicate whether production difference is higher than solar_value
    merged_df['Label'] = merged_df['Production Difference'] > solar_value

    # Create a new DataFrame with the full weather data and the new Label column
    # Note: Since 'Label' is now part of merged_df, we can directly merge merged_df with weather_df
    result_df = pd.merge(weather_df, merged_df[['Date', 'Label']], on='Date', how='inner')

     # Remove 'Date' column and any columns containing the word 'Description'
    result_df.drop(columns=['Hourly Forecast UTC'] + [col for col in result_df if 'Description' in col] + ['Date'], inplace=True)
    # remove all rows with null values
    result_df = result_df.dropna()
    return result_df

df = combine_and_label(140, prepare_weather_file('../theNetwork/data/washburn_wi_weather25 - washburn_wi_weather25.csv'), prepare_egauge_file('../theNetwork/data/data.csv')).to_csv('../the_sun_giggler/output.csv', index=False)
#count the number of true and false labels in the last column of the dataframe
#