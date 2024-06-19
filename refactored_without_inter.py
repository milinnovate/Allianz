import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import ee
import ephem
import geopandas as gpd
import h3
import pandas as pd
import requests
from shapely.geometry import Point
from tqdm import tqdm

# Initialize Earth Engine
ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
tqdm.pandas()

# Paths
input_file_path = "data_combined_22_23.csv"
districts_file_path = "india_district.geojson"
holiday_file_path = "Combined_Holiday_Dates_2022_and_2023.csv"
final_output_file_path = "output_1/"

districts_of_interest = ["Ahmadabad", "Delhi"]
variables = [
    "total_precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "u_component_of_wind_10m_max",
    "v_component_of_wind_10m_max",
]

# Load the holidays CSV
holidays_df = pd.read_csv(holiday_file_path)
holidays_df["Date"] = pd.to_datetime(holidays_df["Date"], format="%d-%m-%Y")
holidays_set = set(holidays_df["Date"])


def clear_directory(directory):
    """Remove all files in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)


def is_long_weekend(date, holidays):
    """Check if a date is a long weekend."""
    if date in holidays:
        day_of_week = date.weekday()
        if day_of_week == 4:  # Friday
            return (date + timedelta(days=1) in holidays) and (
                date + timedelta(days=2) in holidays
            )
        elif day_of_week == 5:  # Saturday
            return (date - timedelta(days=1) in holidays) and (
                date + timedelta(days=1) in holidays
            )
        elif day_of_week == 6:  # Sunday
            return (date - timedelta(days=2) in holidays) and (
                date - timedelta(days=1) in holidays
            )
    return False


def preprocess_dataframe(file_path):
    """Load and preprocess the input DataFrame."""
    df = pd.read_csv(file_path, nrows=5000)
    df.columns = df.columns.str.replace(" ", "_", regex=True)
    df.rename(columns={"lat": "temp", "lon": "lat"}, inplace=True)
    df.rename(columns={"temp": "lon"}, inplace=True)
    df["lat"] = df["lat"].fillna(0)
    df["lon"] = df["lon"].fillna(0)
    df["Callin_Date_Time"] = pd.to_datetime(
        df["Callin_Date_Time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    df["Date"] = pd.to_datetime(df["Callin_Date_Time"].dt.date)
    return df


def get_h3_info(lat, lon, resolution=4):
    """Generate H3 index and get the center coordinates."""
    if lat == 0 and lon == 0:
        return None, None, None
    try:
        index = h3.geo_to_h3(lat, lon, resolution)
        center = h3.h3_to_geo(index)
        return index, center[0], center[1]
    except Exception as e:
        print(f"Failed for lat: {lat}, lon: {lon} with error: {e}")
        return None, None, None


def apply_h3_index(df):
    """Apply H3 index function and create new columns."""
    new_cols = df.progress_apply(
        lambda row: pd.Series(get_h3_info(row["lat"], row["lon"])), axis=1
    )
    new_cols.columns = ["h3_index", "center_lat", "center_lon"]
    df = pd.concat([df, new_cols], axis=1)
    df.dropna(subset=["h3_index"], inplace=True)
    return df


def filter_districts(df, districts_gdf, districts_of_interest):
    """Filter the points that fall within the districts of interest."""
    filtered_districts_gdf = districts_gdf[
        districts_gdf["NAME_2"].isin(districts_of_interest)
    ]
    points_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )
    filtered_districts_gdf = filtered_districts_gdf.to_crs(points_gdf.crs)
    filtered_points_gdf = gpd.sjoin(
        points_gdf, filtered_districts_gdf, how="inner", op="intersects"
    )
    return filtered_points_gdf


def fetch_weather_data(date, h3_index, scale=10000):
    """Fetch weather data from Earth Engine."""
    try:
        point = ee.Geometry.Point(h3.h3_to_geo(h3_index)[::-1])
        dataset = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(
            date.strftime("%Y-%m-%d"),
            (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        data = (
            dataset.first()
            .select(variables)
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=scale)
        )
        results = data.getInfo()
        wind_speed = (
            results.get("u_component_of_wind_10m_max", 0) ** 2
            + results.get("v_component_of_wind_10m_max", 0) ** 2
        ) ** 0.5
        results["wind_speed"] = wind_speed
        results.update({"date": date.strftime("%Y-%m-%d"), "h3_index": h3_index})
        return results
    except Exception as e:
        print(f"Error fetching data for index {h3_index} on {date}: {str(e)}")
        return None


def fetch_class_6_percentage(h3_index, start_date, end_date, scale=30):
    """Fetch class 6 (land use/land cover) percentage from Earth Engine."""
    try:
        hex_area = ee.Geometry.Polygon(h3.h3_to_geo_boundary(h3_index, True))
        dw_dataset = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterDate(start_date, end_date)
            .select(["label"])
        )
        dw_image = dw_dataset.median()
        class_6_mask = dw_image.eq(6)
        class_6_area = (
            class_6_mask.multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(), geometry=hex_area, scale=scale, maxPixels=1e13
            )
            .getInfo()
        )
        total_area = (
            ee.Image.pixelArea()
            .reduceRegion(
                reducer=ee.Reducer.sum(), geometry=hex_area, scale=scale, maxPixels=1e13
            )
            .getInfo()
        )
        class_6_percentage = (
            class_6_area.get("label", 0) / total_area.get("area", 1)
        ) * 100
        return {"h3_index": h3_index, "class_6_percentage": class_6_percentage}
    except Exception as e:
        print(f"Error fetching class 6 percentage for index {h3_index}: {str(e)}")
        return None


def process_district_data(df):
    """Process the district data to fetch and combine weather and LULC data."""
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    # Drop rows where Date conversion failed
    df = df.dropna(subset=["Date"])

    if df.empty:
        return df

    start_date = pd.to_datetime(df["Date"].min())
    end_date = pd.to_datetime(df["Date"].max())
    date_range = pd.date_range(start=start_date, end=end_date)
    selected_h3_index = df["h3_index"].iloc[0]  # Only take the first H3 index

    weather_data = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(fetch_weather_data, date, selected_h3_index)
            for date in date_range
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Fetching Weather Data"
        ):
            result = future.result()
            if result:
                weather_data.append(result)

    class_6_data = fetch_class_6_percentage(selected_h3_index, start_date, end_date)

    if weather_data:
        weather_df = pd.DataFrame(weather_data)
        weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

        combined_df = df.merge(
            weather_df,
            left_on=["h3_index", "Date"],
            right_on=["h3_index", "date"],
            how="left",
        )

        if class_6_data:
            combined_df["class_6_percentage"] = class_6_data["class_6_percentage"]

        return combined_df
    return df


def add_holiday_long_weekend_info(df, holidays_set):
    """Add holiday and long weekend information to the DataFrame."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Convert holidays_set to a set of datetime.date objects
    holidays_set = set(pd.to_datetime(list(holidays_set), format="%d-%m-%Y").date)

    # Add 'holidays' column
    df["holidays"] = df["date"].apply(lambda x: 1 if x in holidays_set else 0)

    # Add 'long_weekend' column
    df["long_weekend"] = df["date"].apply(
        lambda x: 1 if is_long_weekend(x, holidays_set) else 0
    )
    return df


def get_moon_phase(date):
    """Calculate the moon phase for a given date."""
    if pd.isna(date):
        return None
    moon = ephem.Moon(date)
    return moon.phase / 100.0


def add_moon_phase(df):
    """Add moon phase information to the DataFrame."""
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["moon_phase"] = df["Date"].apply(lambda x: get_moon_phase(x))
    return df


def fetch_restaurants_in_h3(h3_index):
    """Fetch number of restaurants within a given H3 hexagon using Overpass API."""
    try:
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        boundary_closed = list(boundary) + [boundary[0]]
        polygon_coords = " ".join([f"{lat} {lon}" for lon, lat in boundary_closed])

        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        (node["amenity"="restaurant"](poly:"{polygon_coords}"););
        out count;
        """

        response = requests.get(overpass_url, params={"data": overpass_query})
        response.raise_for_status()
        data = response.json()

        # Extract and return the count of restaurants
        if "elements" in data and data["elements"]:
            count = data["elements"][0].get("tags", {}).get("total", 0)
            return int(count)
        else:
            return 0
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return 0
    except (IndexError, AttributeError, KeyError) as e:
        print(f"Error parsing response for index {h3_index}: {e}")
        return 0


def process_restaurant_data(df):
    """Process the restaurant data for the DataFrame."""
    unique_h3_indices = df["h3_index"].unique()
    restaurant_counts = {
        index: fetch_restaurants_in_h3(index) for index in tqdm(unique_h3_indices)
    }
    df["Restaurant_Count"] = df["h3_index"].map(restaurant_counts)
    return df


def main():
    df = preprocess_dataframe(input_file_path)

    # Apply H3 index and create new columns
    df = apply_h3_index(df)

    # Load district boundaries and filter
    districts_gdf = gpd.read_file(districts_file_path)
    filtered_points_gdf = filter_districts(df, districts_gdf, districts_of_interest)

    # Ensure the DataFrame has a 'district_name' column
    if "NAME_2" not in filtered_points_gdf.columns:
        raise ValueError(
            "The DataFrame must contain a 'NAME_2' column for district names."
        )

    # Process the filtered data for each district and save separately
    districts = filtered_points_gdf["NAME_2"].unique()
    for district in districts:
        district_df = filtered_points_gdf[filtered_points_gdf["NAME_2"] == district]

        processed_df = process_district_data(district_df)
        processed_df = add_holiday_long_weekend_info(processed_df, holidays_set)
        processed_df = add_moon_phase(processed_df)
        processed_df = process_restaurant_data(processed_df)

        output_file_path = os.path.join(
            os.path.dirname(final_output_file_path), f"final_output_{district}.csv"
        )
        processed_df.to_csv(output_file_path, index=False)
        print(f"Final processed data for {district} saved to {output_file_path}")


if __name__ == "__main__":
    main()
