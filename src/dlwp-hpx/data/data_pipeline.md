# Data preprocessing pipeline

## Download

Download $n$-yearly data chunks per variable from the [ERA5 dataset](https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset&text=era5).

Retrieve data from ECMWF using `cdsapi` (installation described [here](https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5]). An example script is [era5_download.py](src/dlwp-hpx/data/era5_download.py).

## Prepare Tau300-700

To compute and generate the pressure difference field, use [tau_calculation.py](src/dlwp-hpx/data/tau_calculation.py).

## Convert to HEALPix

Either convert the $n$-yearly data into a single file or load them with `xr.open_mfdataset(/paths/to/n-yearly-files)` when converting them to the HEALPix mesh, following `src/dlwp-hpx/remap/healpix.py`

Optional: Compress data using something like `ncpdq -P all_new uncompressed_file.nc compressed_file.nc`

Copy data to target machine and specify path in `src/dlwp-hpx/configs/data/your_data_configs.yaml`
