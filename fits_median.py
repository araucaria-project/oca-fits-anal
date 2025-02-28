#!/usr/bin/env python3
"""
FITS Median Calculator - efficiently computes median of multiple FITS files.

This script calculates the pixel-by-pixel median of multiple FITS files
and saves the result to a new FITS file. It's optimized for memory efficiency
by processing the data in chunks, making it suitable for large files.

Usage:
    python fits_median.py output.fits input1.fits input2.fits [input3.fits ...]
"""

import os
import sys
import numpy as np
from astropy.io import fits
from tqdm import tqdm
import argparse
import gc


def get_fits_info(filename):
    """Extract basic information from a FITS file without loading all data."""
    with fits.open(filename, memmap=True) as hdul:
        header = hdul[0].header.copy()
        shape = hdul[0].data.shape
        dtype = hdul[0].data.dtype
    return header, shape, dtype


def calculate_median_chunked(filenames, output_filename, chunk_size=10, overwrite=False):
    """
    Calculate median of FITS files in memory-efficient chunks.
    
    Args:
        filenames (list): List of input FITS filenames
        output_filename (str): Filename for the output median FITS
        chunk_size (int): Number of rows to process at once
        overwrite (bool): Whether to overwrite existing output file
    """
    if not filenames:
        print("Error: No input files provided")
        return

    # Check if all files exist
    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Error: File {filename} not found")
            return

    # Get info from first file to determine dimensions and type
    print(f"Reading information from {len(filenames)} FITS files...")
    header, shape, dtype = get_fits_info(filenames[0])

    # Create output file
    if os.path.exists(output_filename) and not overwrite:
        print(f"Error: Output file {output_filename} already exists. Use --overwrite to replace it.")
        return

    # Create empty output file with same dimensions
    print(f"Creating output file: {output_filename}")
    output_hdu = fits.PrimaryHDU(np.zeros(shape, dtype=dtype))
    output_hdu.header = header.copy()
    output_hdu.header['HISTORY'] = f'Median of {len(filenames)} FITS files'
    for i, filename in enumerate(filenames):
        output_hdu.header['HISTORY'] = f'Input {i+1}: {os.path.basename(filename)}'

    output_hdu.writeto(output_filename, overwrite=True)

    # Process in chunks to save memory
    total_chunks = (shape[0] + chunk_size - 1) // chunk_size
    print(f"Processing files in {total_chunks} chunks...")

    # Open output file for updating
    with fits.open(output_filename, mode='update') as output_hdul:
        # Process each chunk
        for chunk_idx in tqdm(range(total_chunks)):
            start_row = chunk_idx * chunk_size
            end_row = min(start_row + chunk_size, shape[0])

            # Collect chunks from all files
            chunk_data = []
            for filename in filenames:
                with fits.open(filename, memmap=True) as hdul:
                    # Extract chunk and append to list
                    file_chunk = hdul[0].data[start_row:end_row].copy()
                    chunk_data.append(file_chunk)

            # Stack chunks and calculate median
            stacked_chunks = np.stack(chunk_data, axis=0)
            median_chunk = np.median(stacked_chunks, axis=0)

            # Update output file
            output_hdul[0].data[start_row:end_row] = median_chunk

            # Clear memory
            del chunk_data, stacked_chunks, median_chunk
            gc.collect()

    print(f"Median calculation complete. Result saved to: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description='Calculate median of multiple FITS files.')
    parser.add_argument('output', help='Output FITS file')
    parser.add_argument('inputs', nargs='+', help='Input FITS files')
    parser.add_argument('--chunk-size', type=int, default=10,
                        help='Number of rows to process at once (default: 10)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output file if it exists')

    args = parser.parse_args()

    calculate_median_chunked(args.inputs, args.output,
                             chunk_size=args.chunk_size,
                             overwrite=args.overwrite)


if __name__ == '__main__':
    main()