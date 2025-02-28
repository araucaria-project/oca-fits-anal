"""Searches in subdirectories for FLAT file series, and creates shutter map for each series."""


## Imports
import logging
from pathlib import Path
import argparse
from astropy.io import fits as pyfits
import numpy as np
import numpy.ma as ma

import time


log = logging.getLogger('mapper')

def ultra_optimized_masked_lstsq(A, Np_masked):
    """
    Ultra-optimized solution for n=7, k=2 with millions of columns.
    Groups columns by masking patterns and solves each pattern once.

    This approach is significantly faster than processing each column separately
    because it identifies common mask patterns and reuses calculations.

    Args:
        A: design matrix, shape=(n, 2)
        Np_masked: masked data matrix, shape=(n, m)

    Returns:
        X: coefficient matrix, shape=(2, m)
    """
    start_time = time.time()
    n, m = Np_masked.shape
    k = A.shape[1]
    X = np.zeros((k, m))

    # Extract mask and data from masked array
    if ma.is_masked(Np_masked):
        mask = Np_masked.mask
        data = Np_masked.data
    else:
        # If input is not masked, create a dummy mask (all False)
        mask = np.zeros((n, m), dtype=bool)
        data = Np_masked

    # For each unique mask pattern, find all columns that match
    # Convert column masks to integers for efficient lookup
    # Each column's mask pattern is encoded as a single integer
    # with bits representing which rows are masked
    mask_patterns = np.zeros(m, dtype=np.int32)
    for i in range(n):
        # Use bit shifting to create a unique integer for each mask pattern
        # Example: if row 0 and 2 are masked: 1 + 4 = 5 (binary 101)
        mask_patterns += (mask[i, :].astype(np.int32) << i)

    # Find unique patterns and map them to column indices
    unique_patterns, inverse_indices = np.unique(mask_patterns, return_inverse=True)
    # Create a dictionary mapping each pattern to its column indices
    pattern_to_cols = {p: np.where(mask_patterns == p)[0] for p in unique_patterns}

    # print(f"Found {len(unique_patterns)} unique masking patterns")
    # print(f"Pattern identification time: {time.time() - start_time:.4f}s")

    # For each pattern, calculate pseudoinverse once to reuse
    pattern_to_pinv = {}

    for pattern in unique_patterns:
        # Convert integer pattern back to boolean array
        # Example: 5 (binary 101) means rows 0 and 2 are masked
        pattern_array = np.array([bool(pattern & (1 << i)) for i in range(n)])
        # We need valid (unmasked) rows for calculation
        valid_rows = ~pattern_array

        # Skip patterns with insufficient data points
        if np.sum(valid_rows) < k:
            continue

        # Get valid rows from design matrix
        A_valid = A[valid_rows, :]

        # Calculate pseudoinverse for this pattern
        if np.sum(valid_rows) > k:
            # Overdetermined system (more equations than unknowns)
            # Use pseudoinverse for least squares solution
            pattern_to_pinv[pattern] = np.linalg.pinv(A_valid)
        else:
            # Exactly determined system (equations = unknowns)
            # Use regular inverse for direct solution
            try:
                pattern_to_pinv[pattern] = np.linalg.inv(A_valid)
            except np.linalg.LinAlgError:
                # Skip singular matrices that cannot be inverted
                continue

    # print(f"Pseudoinverse calculation time: {time.time() - start_time:.4f}s")

    # Process each column pattern
    for pattern, cols in pattern_to_cols.items():
        # Skip patterns that couldn't be inverted
        if pattern not in pattern_to_pinv:
            continue

        # Get the precomputed pseudoinverse
        pinv = pattern_to_pinv[pattern]
        # Convert pattern back to boolean array
        pattern_array = np.array([bool(pattern & (1 << i)) for i in range(n)])
        valid_rows = ~pattern_array

        # Process columns in batches to avoid memory issues
        # This is crucial for large datasets with millions of columns
        batch_size = 1000000  # Adjust based on available memory
        for i in range(0, len(cols), batch_size):
            batch_cols = cols[i:i+batch_size]
            # Extract data for current batch (only valid rows)
            batch_data = data[valid_rows][:, batch_cols]

            # Calculate solutions: pinv @ data
            # This is the key computation: applying the pseudoinverse
            # to all columns with the same mask pattern at once
            X_batch = pinv @ batch_data
            # Store results in the output matrix
            X[:, batch_cols] = X_batch

            # print(f"Processed batch {i//batch_size + 1}/{(len(cols) + batch_size - 1)//batch_size} "
            #       f"for pattern {pattern}, time: {time.time() - start_time:.4f}s")

    total_time = time.time() - start_time
    # print(f"Total time: {total_time:.4f}s")
    return X


def process_directory(directory: Path, sigmas, output=None, skip_existing=True):
    basename = directory.name
    log = logging.getLogger(basename)
    if output is None:
        output = directory
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shutter_map_file = output / f'{basename}_shutter_map.fits'
    if skip_existing and shutter_map_file.exists():
        log.info(f"Shutter map {shutter_map_file} already exists, skipping")
        return

    start = total = time.time()
    log.info(f"Start")
    flat_files = list(directory.glob(f'????c_????_?????.fits'))
    master_zero_file_pattern = '????c_????_?????_master_z.fits'
    master_dark_file_pattern = '????c_????_?????_master_d.fits'
    master_flat_file_pattern = '????c_????_?????_master_f_*.fits'
    try:
        master_zero_file = list(directory.glob(master_zero_file_pattern))[0]
    except IndexError:
        log.error(f"Master zero file not found")
        raise FileNotFoundError(f"Master zero file not found")
    try:
        master_dark_file = list(directory.glob(master_dark_file_pattern))[0]
    except IndexError:
        log.error(f"Master dark file not found")
        raise FileNotFoundError(f"Master dark file not found")
    try:
        master_flat_file = sorted(directory.glob(master_flat_file_pattern))
    except IndexError:
        log.error(f"Master flat files not found")
        raise FileNotFoundError(f"Master flat file not found")

    zero = pyfits.open(master_zero_file)[0].data
    # dark data and exptime
    dark_hdu = pyfits.open(master_dark_file)[0]
    dark = dark_hdu.data
    dark_exptime = float(dark_hdu.header['EXPTIME'])
    master_hdr = pyfits.open(master_flat_file[0])[0].header

    log.info(f"Zero frame: {master_zero_file.name}")
    log.info(f"Dark frame: {master_dark_file.name}   exptime={dark_exptime}")

    _data = []
    _exptime = []
    for p in flat_files:
        with pyfits.open(p) as hdul:
            _exp  = float(hdul[0].header['EXPTIME'])
            log.debug(f'Loading {p.name} exptime={_exp}')
            d = hdul[0].data - zero - dark * _exp / dark_exptime
            _data.append(d)
            _exptime.append(_exp)
    D = np.array(_data)
    t = np.array(_exptime)
    del _data, _exptime, zero, dark
    log.info(f"Loaded {len(D)} frames in {time.time() - start:.2f}s")

    start = time.time()
    M = np.median(D, axis=(1, 2))
    F = M / t
    for _file, _exp, _flux, _median in zip(flat_files, t, F, M):
        log.debug(f'{_file.name}   median={_median:8.2f} ADU   exp={_exp:7.2f}s   flux={_flux:8.2f} ADU/s')
    N = (D/F[:, None, None])
    Q = D / M[:, None, None]
    Qm = np.median(Q, axis=(0))
    Qs = np.std(Q, axis=(0))

    Np = N.reshape(N.shape[0], -1)
    A = np.vstack([t, np.ones_like(t)]).T

    if sigmas > 0:
        R = np.logical_and(Qm - sigmas * Qs < Q, Q < Qm + sigmas * Qs)
        Rp = R.reshape(R.shape[0], -1)
        Npm = ma.array(Np, mask=~Rp)
        X = ultra_optimized_masked_lstsq(A, Npm)
        del Npm, Rp, R
    else:
        X = np.linalg.lstsq(A, Np, rcond=None)[0]

    # extract map from reshaped X
    map = X[1].reshape(N.shape[1:])
    del N, Q, Qm, Qs, Np, A, X

    log.info(f"Calculated shutter map in {time.time() - start:.2f}s")

    # save shutter map
    log.info(f"Saving shutter map to {shutter_map_file}")
    # copy header from master_flat_file, set some header values
    hdr = master_hdr.copy()
    hdr['HISTORY'] = 'Created by shutter_mapper.py'
    hdr['HISTORY'] = f'Processed directory {directory}'
    hdr['HISTORY'] = f'Sigma clipping: {sigmas}'
    hdr['HISTORY'] = f'Zero frame: {master_zero_file.name}'
    hdr['HISTORY'] = f'Dark frame: {master_dark_file.name}   exptime={dark_exptime}'
    hdr['HISTORY'] = f'Flat frames: {len(flat_files)}'
    hdr['IMAGETYP'] = 'shutter'
    hdr['BUNIT'] = 's'
    if 'SATURATE' in hdr:
        del hdr['SATURATE']
    if 'EXPTIME' in hdr:
        del hdr['EXPTIME']
    hdr['SHUT-ERA'] = 1
    # save fits
    pyfits.writeto(shutter_map_file, map, hdr, overwrite=True)
    log.info(f"Saved shutter map to {shutter_map_file}")
    log.info(f"Total time: {time.time() - total:.2f}s")












def main():
    # command line parsing using argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='Root directory to scan for subdirectories with FLAT series')
    parser.add_argument('-o', '--output', help='Path to save the shutter maps')
    parser.add_argument('-k', '--skip', action='store_true', help='Skip existing shutter maps')
    parser.add_argument('-s', '--sigma', type=float, default=2.0, help='Number of sigmas for sigma clipping, 0 - no clipping')
    # parser.add_argument("-j", "--jobs", type=int, default=4, help="Number of parallel workers")
    parser.add_argument('-l', '--log', default='INFO', help='Log level')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=args.log.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    basedir = Path(args.directory)
    if not basedir.exists():
        log.error(f"Directory {basedir} does not exist")
        return
    for subdir in basedir.glob('????c_????_?????'):
        if subdir.is_dir():
            try:
                process_directory(subdir, args.sigma, args.output, args.skip)
            except FileNotFoundError as e:
                log.error(f"Error processing {subdir}: {e}")
            except Exception as e:
                log.error(f"Error processing directory {subdir}: {e}")
                raise e


if __name__ == "__main__":
    main()






