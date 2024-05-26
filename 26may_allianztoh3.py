import pandas as pd
import h3
from tqdm import tqdm
tqdm.pandas()

# Constants
resolution = 4  # Define the resolution for H3
default_lat = 0  # Default latitude if missing
default_lon = 0  # Default longitude if missing

# Load data and reset index to create an 'index' column
df = pd.read_csv('/Users/milindsoni/Documents/projects/Allianz/may/Allianz_Data_Filtered_2023.csv')
df.reset_index(inplace=True)

# Swap and replace NaN values in lat and lon
df.rename(columns={'lat': 'temp', 'lon': 'lat'}, inplace=True)
df.rename(columns={'temp': 'lon'}, inplace=True)
df['lat'] = df['lat'].fillna(default_lat)
df['lon'] = df['lon'].fillna(default_lon)

# Function to generate H3 index and get the center coordinates
def get_h3_info(lat, lon, resolution):
    if lat == default_lat and lon == default_lon:
        return None, None, None
    try:
        index = h3.geo_to_h3(lat, lon, resolution)
        center = h3.h3_to_geo(index)
        return index, center[0], center[1]
    except Exception as e:
        print(f"Failed for lat: {lat}, lon: {lon} with error: {e}")
        return None, None, None

# Apply the function and create new columns
df[['h3_index', 'center_lat', 'center_lon']] = df.progress_apply(
    lambda row: pd.Series(get_h3_info(row['lat'], row['lon'], resolution)), axis=1
)

# Drop rows without a valid H3 index
df = df.dropna(subset=['h3_index'])

# Ensure the 'Day' column is included in the output and convert 'Day' to datetime
df['Day'] = pd.to_datetime(df['Day'], format='%d-%m-%Y', errors='coerce')

# Prepare columns for the merge, retaining only the necessary ones
columns_to_keep = ['index', 'h3_index', 'center_lat', 'center_lon', 'Day', 'lat', 'lon']
df = df[columns_to_keep]

# Load the original data again for the join and reset index
df_original = pd.read_csv('/Users/milindsoni/Documents/projects/Allianz/may/Allianz_Data_Filtered_2023.csv')
df_original.reset_index(inplace=True)  # Ensure the 'index' is reset here as well

# Join the processed data back with the original data using the index column
df_final = df_original.merge(df, on='index', how='left')

# Rename 'lat_y' and 'lon_y' to 'lat' and 'lon', and drop 'lat_x' and 'lon_x'
df_final.rename(columns={'lat_y': 'lat', 'lon_y': 'lon'}, inplace=True)
df_final.drop(['lat_x', 'lon_x'], axis=1, inplace=True)

# Save the final DataFrame to CSV
df_final.to_csv('final_output_with_h3_centers.csv', index=False)
