import cdsapi
import xarray as xr
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

# FIX: Find the previous month directly instead of subtracting 35 days.
# By going to the 1st of the current month and subtracting 1 day, we safely land on the previous month.
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
    "time": ["00:00"], # Maintained optimization: Only downloading 00:00
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

# Detect the correct time dimension name
time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

# Loop through every day (timestep) in the downloaded file
for i in range(len(ds[time_coord])):
    # FIX: Robustly extract datetime integers to guarantee exact YYYYMMDD_HH formatting
    year_val = int(ds[time_coord][i].dt.year)
    month_val = int(ds[time_coord][i].dt.month)
    day_val = int(ds[time_coord][i].dt.day)
    hour_val = int(ds[time_coord][i].dt.hour)
    
    # Format components with leading zeros (e.g., 20260603_00)
    file_suffix = f"{year_val}{month_val:02d}{day_val:02d}_{hour_val:02d}"
    print(f"Processing timestamp: {file_suffix}...")

    # --- 1. WIND (JSON) ---
    u_wind = np.nan_to_num(ds['u'].values[i, 0, :, :], nan=0.0)
    v_wind = np.nan_to_num(ds['v'].values[i, 0, :, :], nan=0.0)
    
    header = {
        "lo1": float(ds.longitude.min()), "la1": float(ds.latitude.max()),
        "dx": float(abs(ds.longitude[1] - ds.longitude[0])),
        "dy": float(abs(ds.latitude[0] - ds.latitude[1])),
        "nx": int(len(ds.longitude)), "ny": int(len(ds.latitude)),
        "refTime": str(ds[time_coord][i].values)
    }
    output_json = [
        {"header": {**header, "parameterCategory": 2, "parameterNumber": 2}, "data": u_wind.flatten().tolist()},
        {"header": {**header, "parameterCategory": 2, "parameterNumber": 3}, "data": v_wind.flatten().tolist()}
    ]
    with open(f"data/wind_{file_suffix}.json", "w") as f:
        json.dump(output_json, f)

    # --- 2. TEMPERATURE (PNG) ---
    temp_data = ds['t'].values[i, 0, :, :]
    plt.imsave(f'data/temp_{file_suffix}.png', temp_data, cmap='coolwarm')

    # --- 3. GEOPOTENTIAL (PNG) ---
    geo_data = ds['z'].values[i, 0, :, :]
    plt.imsave(f'data/geo_{file_suffix}.png', geo_data, cmap='viridis')

    # --- 4. RAIN WATER CONTENT (PNG) ---
    rain_data = ds['crwc'].values[i, 0, :, :]
    plt.imsave(f'data/rain_{file_suffix}.png', rain_data, cmap='Blues')

print("All daily processing complete! Files saved to /data folder.")
