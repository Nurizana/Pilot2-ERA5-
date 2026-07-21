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

print("Calculating safe dates for ERA5 dataset...")
now = datetime.utcnow()
safe_date = now.replace(day=1) - timedelta(days=35)
target_year = safe_date.strftime("%Y")
target_month = safe_date.strftime("%m")
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
    "time": ["00:00"], # OPTIMIZED: Only download 00:00 to save processing time
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

# --- THE FIX: Detect the correct time dimension name ---
# (ERA5 NetCDF often uses 'valid_time' instead of 'time')
time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

# Loop through every day (timestep) in the downloaded file using the correct time coordinate
for i in range(len(ds[time_coord])):
    day_str = str(ds[time_coord][i].dt.day.values).zfill(2)
    print(f"Processing Day {day_str}...")

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
    with open(f"data/wind_{day_str}.json", "w") as f:
        json.dump(output_json, f)

    # --- 2. TEMPERATURE (PNG) ---
    temp_data = ds['t'].values[i, 0, :, :]
    plt.imsave(f'data/temp_{day_str}.png', temp_data, cmap='coolwarm')

    # --- 3. GEOPOTENTIAL (PNG) ---
    geo_data = ds['z'].values[i, 0, :, :]
    plt.imsave(f'data/geo_{day_str}.png', geo_data, cmap='viridis')

    # --- 4. RAIN WATER CONTENT (PNG) ---
    rain_data = ds['crwc'].values[i, 0, :, :]
    plt.imsave(f'data/rain_{day_str}.png', rain_data, cmap='Blues')

print("All daily processing complete! Files saved to /data folder.")
