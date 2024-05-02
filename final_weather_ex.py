import pandas as pd
import h3
import ee
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize Earth Engine with the high-volume endpoint
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# Load the dataset
df = pd.read_csv('date_h3_res_7_only.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Drop rows where the date is NaT
df = df.dropna(subset=['date'])


h3_index_column = 'h3_res_7' 
if h3_index_column not in df.columns:
    raise ValueError(f"Missing required H3 index column: {h3_index_column}")


df = df.drop_duplicates(subset=[h3_index_column])

def fetch_weather_data(date, h3_index, scale):
    try:
        center = h3.h3_to_geo(h3_index)
        point = ee.Geometry.Point([center[1], center[0]])
        start_date = date.strftime('%Y-%m-%d')
        end_date = (date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        dataset = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start_date, end_date)
        variables = [
            'total_precipitation_sum', 'temperature_2m_max', 'temperature_2m_min',
            'surface_net_solar_radiation_sum', 'surface_net_thermal_radiation_sum',
            'surface_sensible_heat_flux_sum', 'snow_depth_max', 'snow_depth_min',
            'u_component_of_wind_10m_max', 'u_component_of_wind_10m_min',
            'v_component_of_wind_10m_max', 'v_component_of_wind_10m_min',
            'surface_pressure_max', 'surface_pressure_min',
            'leaf_area_index_low_vegetation_max', 'leaf_area_index_low_vegetation_min'
        ]
        data = dataset.first().select(variables).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=scale  # Dynamic scale based on the resolution
        )
        results = data.getInfo()
        results.update({'date': date.strftime('%Y-%m-%d'), 'h3_index': h3_index})
        return results
    except Exception as e:
        print(f"Error fetching data for index {h3_index}: {str(e)}")
        return None

scale = 10000  # 10km resolution for H3 resolution 7
tasks = [(date, h3_index) for date, h3_index in zip(df['date'], df[h3_index_column])]

weather_data = []
with ThreadPoolExecutor(max_workers=16) as executor:
    future_to_data = {executor.submit(fetch_weather_data, task[0], task[1], scale): task for task in tasks}
    for future in tqdm(as_completed(future_to_data), total=len(tasks), desc="Fetching Weather Data"):
        weather_results = future.result()
        if weather_results:
            weather_data.append(weather_results)

weather_df = pd.DataFrame(weather_data)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
final_df = pd.merge(df, weather_df, how='left', left_on=['h3_res_7', 'date'], right_on=['h3_index', 'date'])


print(final_df.head())
final_df.to_csv('merged_dataset_with_weather.csv', index=False)
