## Getting Started

Create an environment with pyhton installed, e.g., with `conda create -n dlwp-hpx python` and activate it with `conda activate dlwp-hpx`.

Install `dlwp-hpx` package with `pip install .`, favorably in a dedicated environment (as shown above). Then cd into `src/dlwp-hpx`

Verify installation with `python remap/healpix.py`, which projects the exemplary `data/era5_z500.nc` file from the equirectangular LatLon convention of shape $[..., H, W]$ (with $H$ the latitudes and $W$ the longitudes) into the HEALPix mesh of shape $[..., F, H, W]$, where $F=12$ the number of faces and $H=W$ the side length of the squares.


## Data

Download data from the [ERA5](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3803) dataset, i.e.,
1. [constants](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form)
	- land-sea mask
	- orography
2. [surface variables](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form)
	- two meter temperature
	- total column water vapor
3. [pressure fields](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form)
	- geopotential 250 hPa
	- geopotential 300 hPa
	- geopotential 500 hPa
	- geopotential 700 hPa
	- geopotential 1000 hPa
	- air temperature 850 hPa

Subsequently, project the data to the HEALPix mesh, using the remapping functions in `src/remapping/healpix.py`


## Train

Multiple exemplary commands to train a DLWP-HPX model are provided in `src/dlwp-hpx/scripts/run_training.sh`. To start a training of the HEALPix model, prepare an according `RUN_CMND` in `run_training.sh` and run `sh scripts/train.py` (from `src/dlwp-hpx/`).


## Forecast

To generate a forecast with a trained model, consider the `src/dlwp-hpx/scripts/forecast.py` script, which requires the model directory as `-m` argument.
