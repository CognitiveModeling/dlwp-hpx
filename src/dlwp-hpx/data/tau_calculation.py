#! /usr/bin/env python3

"""
This script takes two pressure fields, computes their difference, and writes the result to file.
"""


import argparse
import xarray as xr
from dask.diagnostics import ProgressBar


def main(run_args):
    # Load input fields
    f1 = xr.open_dataset(run_args.input_file_name_1, chunks={'time': run_args.time_chunksize})
    f2 = xr.open_dataset(run_args.input_file_name_2, chunks={'time': run_args.time_chunksize})
    
    # Change level so that operation align
    f1 = f1.assign_coords({'level':[500.]})
    f2 = f2.assign_coords({'level':[500.]})

    # Subtract and rename variable
    tau = f1 - f2  # e.g., z300 - z700
    tau = tau.rename({run_args.source_variable_name: run_args.target_variable_name})
    
    # Compute and write in chunks 
    delayed_job = tau.to_netcdf(run_args.output_file_name, compute=False)
    print(f'Computing and writing with chunksize {run_args.time_chunksize} for time...')
    with ProgressBar():
         delayed_job.compute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute difference of two pressure fields and write to file.")
    parser.add_argument("-f1", "--input-file-name-1", type=str, required=True,
                        help="Path to the first .nc file.")
    parser.add_argument("-f2", "--input-file-name-2", type=str, required=True,
                        help="Path to the second .nc file that will be subtracted from the first.")
    parser.add_argument("-o", "--output-file-name", type=str, default="diff.nc",
                        help="The name of the output file.")
    parser.add_argument("-s", "--source-variable-name", type=str, default="z",
                        help="The name of the variables in the input.nc files.")
    parser.add_argument("-t", "--target-variable-name", type=str, default="tau300-700",
                        help="The name of the variable in the output.nc file.")
    parser.add_argument("-c", "--time-chunksize", type=int, default=100,
                        help="The time chunksize for loading and writing the dask arrays.")
    run_args = parser.parse_args()
    main(run_args)
    print('Done.')
