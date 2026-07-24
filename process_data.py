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

# Detect the correct time dimension name
time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

# Get 2D meshgrids for Lat/Lon to correctly map quiver arrows
lons = ds.longitude.values
lats = ds.latitude.values
lon2d, lat2d = np.meshgrid(lons, lats)

# Loop through every day (timestep) in the downloaded file
for i in range(len(ds[time_coord])):
    timestamp = pd.to_datetime(ds[time_coord][i].values)
    file_suffix = f"{timestamp.year}{timestamp.month:02d}{timestamp.day:02d}_{timestamp.hour:02d}"
    print(f"Processing timestamp: {file_suffix}...")

    # --- WIND DATA EXTRACTION ---
    u_wind = np.nan_to_num(ds['u'].isel({time_coord: i}).squeeze().values, nan=0.0)
    v_wind = np.nan_to_num(ds['v'].isel({time_coord: i}).squeeze().values, nan=0.0)
    
    header = {
        "lo1": float(ds.longitude.min().values), "la1": float(ds.latitude.max().values),
        "dx": float(abs(ds.longitude[1].values - ds.longitude[0].values)),
        "dy": float(abs(ds.latitude[0].values - ds.latitude[1].values)),
        "nx": int(len(ds.longitude)), "ny": int(len(ds.latitude)),
        "refTime": str(timestamp)
    }
    
    # 1. Save JSON for the Animated Wind Overlay (Section 4)
    output_json = [
        {"header": {**header, "parameterCategory": 2, "parameterNumber": 2}, "data": u_wind.flatten().tolist()},
        {"header": {**header, "parameterCategory": 2, "parameterNumber": 3}, "data": v_wind.flatten().tolist()}
    ]
    with open(f"data/wind_{file_suffix}.json", "w") as f:
        json.dump(output_json, f)

    # --- STATIC IMAGE EXPORTS (Section 3) ---
    
    # 2. Save Static Wind PNG using Vector Arrows (Quiver) instead of a color heatmap
    
    # Subsample data to prevent the image from being completely filled with black pixels
    skip = (slice(None, None, 15), slice(None, None, 15)) 
    
    # Create a figure that aligns perfectly with the [[90, 0], [-90, 360]] mapping bounds
    fig = plt.figure(figsize=(14.4, 7.2), dpi=100) 
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off() # Turn off borders and axes
    fig.add_axes(ax)
    
    # Force coordinate limits
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    
    # Plot arrows (cyan color with slight transparency looks great on both dark/ocean maps)
    ax.quiver(lon2d[skip], lat2d[skip], u_wind[skip], v_wind[skip],
              color='#00ffff', pivot='middle', scale=400, alpha=0.9, width=0.002)
    
    # Save with transparent background so it acts as an overlay
    fig.savefig(f'data/wind_{file_suffix}.png', format='png', transparent=True, pad_inches=0)
    plt.close(fig)

    # Save Temperature PNG
    temp_data = ds['t'].isel({time_coord: i}).squeeze().values
    plt.imsave(f'data/temp_{file_suffix}.png', temp_data, cmap='coolwarm')

    # Save Geopotential PNG
    geo_data = ds['z'].isel({time_coord: i}).squeeze().values
    plt.imsave(f'data/geo_{file_suffix}.png', geo_data, cmap='viridis')

    # Save Rain PNG
    rain_data = ds['crwc'].isel({time_coord: i}).squeeze().values
    plt.imsave(f'data/rain_{file_suffix}.png', rain_data, cmap='Blues')

# Fixed the cut-off string from your original file
print("All daily processing completed.")
