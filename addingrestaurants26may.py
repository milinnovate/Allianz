import os
import pandas as pd
import h3
import requests
from tqdm import tqdm

def fetch_restaurants_in_h3(h3_index):
    """Fetch number of restaurants within a given H3 hexagon using Overpass API."""
    try:
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        boundary_closed = list(boundary) + [boundary[0]]
        polygon_coords = ' '.join([f"{lat} {lon}" for lon, lat in boundary_closed])
        
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        (node["amenity"="restaurant"](poly:"{polygon_coords}"););
        out count;
        """
        
        response = requests.get(overpass_url, params={'data': overpass_query})
        response.raise_for_status()
        data = response.json()
        
        # Debugging output to inspect the response structure
        print(f"Response for index {h3_index}: {data}")
        
        # Extract and return the count of restaurants
        if 'elements' in data and data['elements']:
            count = data['elements'][0].get('tags', {}).get('total', 0)
            return int(count)
        else:
            return 0
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return 0
    except (IndexError, AttributeError, KeyError) as e:
        print(f"Error parsing response for index {h3_index}: {e}")
        return 0

def process_district_data(df):
    unique_h3_indices = df['h3_index'].unique()
    restaurant_counts = {index: fetch_restaurants_in_h3(index) for index in tqdm(unique_h3_indices)}
    df['Restaurant_Count'] = df['h3_index'].map(restaurant_counts)
    return df

def preprocess_data(df):
    # Split date into separate day, month, and year columns
    df['Date'] = pd.to_datetime(df['date'], errors='coerce')
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # Format hour column to only show the hour
    df['Hour'] = pd.to_datetime(df['Hour'], format='%H:%M:%S', errors='coerce').dt.hour
    
    # Convert precipitation from meters to centimeters (1 meter = 100 centimeters)
    df['total_precipitation_sum'] = (df['total_precipitation_sum'] * 100).round(2)
    
    # Convert temperature from Kelvin to Celsius
    df['temperature_2m_max'] = (df['temperature_2m_max'] - 273.15).round(2)
    df['temperature_2m_min'] = (df['temperature_2m_min'] - 273.15).round(2)
    
    # Round wind speed to 2 decimal places
    df['wind_speed'] = df['wind_speed'].round(2)
    
    # Drop u_component_of_wind_10m_max and v_component_of_wind_10m_max columns
    df = df.drop(columns=['u_component_of_wind_10m_max', 'v_component_of_wind_10m_max'])
    
    return df

def rearrange_columns(df):
    # Rename columns to replace spaces with underscores
    df.columns = df.columns.str.replace(' ', '_')
    
    # Define the order of columns
    columns_order = [
        'date', 'File_No', 'h3_index', 'center_lat', 'center_lon', 'lat', 'lon', 'ID_2', 'NAME_2',
        'index', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Event_numerical', 'State_numerical',
        'Reason_Of_Breakdown_numerical', 'Final_Service_numerical', 'Status_Color_numerical',
        'Vehicle_Type__numerical', 'GearBox_numerical', 'class_6_percentage', 'temperature_2m_max',
        'temperature_2m_min', 'total_precipitation_sum', 'wind_speed', 'Holidays', 'Long_Weekend',
        'Restaurant_Count'
    ]
    
    # Ensure only unique columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Reorder columns
    df = df[columns_order]
    
    return df

def process_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.csv'):
            print(f"Processing {file_name}")
            file_path = os.path.join(input_directory, file_name)
            df = pd.read_csv(file_path)
            
            df['h3_index'] = df['h3_index'].astype(str)
            df = process_district_data(df)
            df = preprocess_data(df)
            df = rearrange_columns(df)
            
            output_file_path = os.path.join(output_directory, f"enhanced_{file_name}")
            df.to_csv(output_file_path, index=False)
            
    print("All files processed.")

# Define input and output directories
input_directory = "/Users/milindsoni/Documents/projects/Allianz/presentationdata/output2/processedwithhol"
output_directory = "/Users/milindsoni/Documents/projects/Allianz/presentationdata/output2/upload26may"

# Process all CSV files in the directory
process_files(input_directory, output_directory)
