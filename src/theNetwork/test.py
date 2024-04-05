
from network_input_creator import prep_weather_file
from network_input_creator import label_weather_for_training

label_df = label_weather_for_training('data/washburn_wi_weather1.csv', 'data/data.csv', label=True)
print(len(label_df.columns))
