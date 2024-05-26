import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Load district boundaries
districts_gdf = gpd.read_file("/Users/milindsoni/Documents/projects/Allianz/presentationdata/statewise/geojsons/india_district.geojson")

# Define districts of interest
districts_of_interest = [
    "Greater Bombay", "Delhi", "Pune", "Bangalore Rural", "Bangalore Urban",
    "North Goa", "South Goa", "Udaipur", "Chandigarh", "Jaisalmer",
    "Thiruvananthapuram", "Kamrup"
]

# Filter GeoDataFrame for the districts of interest
filtered_districts_gdf = districts_gdf[districts_gdf['NAME_2'].isin(districts_of_interest)]

# Load the points data
points_df = pd.read_csv("final_output_with_h3_centers.csv")

# Remove Unnamed columns and Day_y column before creating GeoDataFrame
unnamed_cols = [col for col in points_df.columns if "Unnamed" in col]
columns_to_drop = unnamed_cols + ['Day_y']
points_df = points_df.drop(columns=columns_to_drop)

# Convert cleaned DataFrame to GeoDataFrame
points_gdf = gpd.GeoDataFrame(points_df, geometry=gpd.points_from_xy(points_df.center_lon, points_df.center_lat), crs="EPSG:4326")

# Ensure district GeoDataFrame is in the same CRS as points GeoDataFrame
filtered_districts_gdf = filtered_districts_gdf.to_crs(points_gdf.crs)

# Perform spatial join to filter only points within the districts
filtered_points_gdf = gpd.sjoin(points_gdf, filtered_districts_gdf, how="inner", op='intersects')

# Save the filtered points to CSV for each district
for district in filtered_points_gdf['NAME_2'].unique():
    district_points = filtered_points_gdf[filtered_points_gdf['NAME_2'] == district]
    district_points.drop(columns='geometry').to_csv(f'output/{district}_points.csv', index=False)
