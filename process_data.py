import cdsapi
import xarray as xr
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import calendar
import os

# Create a folder to store the daily files
os.makedirs("data", exist_ok=True)

print("Calculating target date for ERA5 dataset...")
now = datetime.utcnow()

# Target the previous month safely
target_date = now.replace(day=1) - timedelta(days=1)
target_year = target_date.strftime("%Y")
target_month = target_date.strftime("%m")

num_days = calendar.monthrange(int(target_year), int(target_month))[1]
days_list = [f"{d:02d}" for d in range(1, num_days + 1)]

print("Connecting to CDS API...")
client = cdsapi.Client()

dataset = "reanalysis-era5-pressure-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "geopotential",
        "specific_rain_water_content",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "year": [target_year],
    "month": [target_month], 
    "day": days_list,
    "time": ["00:00"], 
    "pressure_level": ["850"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

output_file = "monthly_data.nc"
print(f"Downloading data for {target_year}-{target_month} (at 00:00)...")
client.retrieve(dataset, request, output_file)

print("Processing NetCDF file and generating daily layers...")
ds = xr.open_dataset(output_file)

# Standardize longitudes to 0-360 for web mapping
ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby('longitude')
time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

# Loop through every day (timestep) in the downloaded file
for i in range(len(ds[time_coord])):
    timestamp = pd.to_datetime(ds[time_coord][i].values)
    file_suffix = f"{timestamp.year}{timestamp.month:02d}{timestamp.day:02d}_{timestamp.hour:02d}"
    print(f"Processing timestamp: {file_suffix}...")

    # --- WIND DATA EXTRACTION FOR WEB RENDERER ---
    u_wind = np.nan_to_num(ds['u'].isel({time_coord: i}).squeeze().values, nan=0.0)
    v_wind = np.nan_to_num(ds['v'].isel({time_coord: i}).squeeze().values, nan=0.0)
    
    header = {
        "lo1": float(ds.longitude.min().values), "la1": float(ds.latitude.max().values),
        "dx": float(abs(ds.longitude[1].values - ds.longitude[0].values)),
        "dy": float(abs(ds.latitude[0].values - ds.latitude[1].values)),
        "nx": int(len(ds.longitude)), "ny": int(len(ds.latitude)),
        "refTime": str(timestamp)
    }
    
    # Save JSON for the Animated Particles AND our new Dynamic Canvas Vectors
    # We no longer export a static PNG for wind, as the browser handles it dynamically!
    output_json = [
        {"header": {**header, "parameterCategory": 2, "parameterNumber": 2}, "data": u_wind.flatten().tolist()},
        {"header": {**header, "parameterCategory": 2, "parameterNumber": 3}, "data": v_wind.flatten().tolist()}
    ]
    with open(f"data/wind_{file_suffix}.json", "w") as f:
        json.dump(output_json, f)

    # --- STATIC IMAGE EXPORTS (Temp, Geo, Rain) ---
    temp_data = ds['t'].isel({time_coord: i}).squeeze().values
    plt.imsave(f'data/temp_{file_suffix}.png', temp_data, cmap='coolwarm')

    geo_data = ds['z'].isel({time_coord: i}).squeeze().values
    plt.imsave(f'data/geo_{file_suffix}.png', geo_data, cmap='viridis')

    rain_data = ds['crwc'].isel({time_coord: i}).squeeze().values
    plt.imsave(f'data/rain_{file_suffix}.png', rain_data, cmap='Blues')

print("All daily processing completed.")
