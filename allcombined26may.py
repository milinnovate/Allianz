import pandas as pd
import h3
import ee
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Initialize Earth Engine
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
def fetch_class_6_percentage(h3_index, scale=30):  # Adjust scale as needed for your application
    try:
        center = h3.h3_to_geo(h3_index)
        hex_area = ee.Geometry.Polygon(h3.h3_to_geo_boundary(h3_index, True))

        # Define the time range
        start_date = pd.to_datetime('now').strftime('%Y-%m-01')  # Start of current month
        end_date = pd.to_datetime('now').strftime('%Y-%m-%d')  # Today's date

        # Fetch Dynamic World LULC data
        dw_dataset = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate(start_date, end_date).select(['label'])
        if dw_dataset.size().getInfo() > 0:
            dw_image = dw_dataset.median()  # Using median to better handle overlapping images

            # Mask for class 6 areas
            class_6_mask = dw_image.eq(6)

            # Calculate the total class 6 area
            class_6_area = class_6_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=hex_area,
                scale=scale,
                maxPixels=1e13
            ).getInfo()

            # Calculate the total area of the hex
            total_area = ee.Image.pixelArea().reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=hex_area,
                scale=scale,
                maxPixels=1e13
            ).getInfo()

            # Calculate the percentage of class 6 area
            class_6_percentage = (class_6_area.get('label', 0) / total_area.get('area', 1)) * 100  # To avoid division by zero
            
            return {
                'h3_index': h3_index,
                'class_6_percentage': class_6_percentage
            }
        else:
            print(f"No images available for index {h3_index} between {start_date} and {end_date}")
            return {'h3_index': h3_index, 'class_6_percentage': 0}  # Assume no class 6 area if no images
    except Exception as e:
        print(f"Error fetching class 6 percentage for index {h3_index}: {str(e)}")
        return None
    
variables = [
    'total_precipitation_sum', 'temperature_2m_max', 'temperature_2m_min',
    'u_component_of_wind_10m_max', 'v_component_of_wind_10m_max'
]

# Function to fetch weather data
def fetch_weather_data(date, h3_index, scale=10000):
    try:
        center = h3.h3_to_geo(h3_index)
        point = ee.Geometry.Point(center[::-1])  # Correct order of lon, lat
        start_date = date.strftime('%Y-%m-%d')
        end_date = (date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        dataset = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_date, end_date)
        data = dataset.first().select(variables).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=point, scale=scale)
        results = data.getInfo()

        # Compute the Pythagorean sum of wind components
        wind_speed = (results['u_component_of_wind_10m_max']**2 + results['v_component_of_wind_10m_max']**2)**0.5
        results['wind_speed'] = wind_speed
        results.update({'date': date.strftime('%Y-%m-%d'), 'h3_index': h3_index})
        return results
    except Exception as e:
        print(f"Error fetching data for index {h3_index} on {date}: {str(e)}")
        return None
# Define directories
input_directory = "/Users/milindsoni/Documents/projects/Allianz/presentationdata/output"  # Adjust this path to your directory containing the output CSV files
result_directory = "/Users/milindsoni/Documents/projects/Allianz/presentationdata/output2"  # Adjust this path to where you want to save the results
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

# Function to safely load CSV files
def safe_load_csv(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame if file doesn't exist
    if os.stat(file_path).st_size == 0:
        print(f"File is empty: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame if file is empty
    return pd.read_csv(file_path)

# Process each file
for file_name in os.listdir(input_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_directory, file_name)
        df = safe_load_csv(file_path)
        district_name = file_name.replace('.csv', '').replace('output_', '')  # Example to extract district name from file name
        
        print(f"Processing data for district: {district_name}")  # Print the district being processed

        df.rename(columns={'Day_x': 'Day'}, inplace=True)
        df['Day'] = pd.to_datetime(df['Day'], format='%d-%m-%Y', errors='coerce')
        columns_to_drop = ['index_right', 'ID_0', 'ISO', 'NAME_0', 'ID_1', 'NAME_1', 'NL_NAME_2', 'VARNAME_2', 'TYPE_2', 'ENGTYPE_2']
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        df['h3_index'] = df['h3_index'].astype(str)
        unique_h3_indices = df['h3_index'].unique()
        date_range = pd.date_range(start=df['Day'].min(), end=df['Day'].max())

        lulc_data = []
        weather_data = []

        with ThreadPoolExecutor(max_workers=32) as executor:
            lulc_futures = [executor.submit(fetch_class_6_percentage, h3_index, 30) for h3_index in unique_h3_indices]
            for future in tqdm(as_completed(lulc_futures), total=len(unique_h3_indices), desc=f"Fetching LULC Data for {district_name}"):
                result = future.result()
                if result:
                    lulc_data.append(result)

            weather_futures = [executor.submit(fetch_weather_data, date, h3_index, 10000) for date in date_range for h3_index in unique_h3_indices]
            for future in tqdm(as_completed(weather_futures), total=len(weather_futures), desc=f"Fetching Weather Data for {district_name}"):
                result = future.result()
                if result:
                    weather_data.append(result)

        if lulc_data:
            lulc_df = pd.DataFrame(lulc_data)
            df = pd.merge(df, lulc_df, on='h3_index', how='left')
        if weather_data:
            weather_df = pd.DataFrame(weather_data)
            weather_df['date'] = pd.to_datetime(weather_df['date'], format='%Y-%m-%d', errors='coerce')
            df = pd.merge(df, weather_df, left_on=['Day', 'h3_index'], right_on=['date', 'h3_index'], how='left')

        df.to_csv(os.path.join(result_directory, f'enhanced_{district_name}.csv'), index=False)
