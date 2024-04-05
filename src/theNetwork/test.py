
from network_input_creator import prep_weather_file
from network_input_creator import label_weather_for_training

label_df = label_weather_for_training('data/washburn_wi_weather1.csv', 'data/data.csv', label=False)
label_df.to_csv('data/labelled_weather.csv', index=False)
