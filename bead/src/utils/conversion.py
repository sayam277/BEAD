# Description: This script contains functions to convert CSV files to HDF5 and .npy files in parallel.
import os
os.environ['KMP_WARNINGS'] = 'off'
import csv
import numpy as np
import h5py
from numba import njit
from dask import delayed, compute

# Define structured dtypes for memory efficiency
event_dtype = np.dtype([
    ('evt_id', np.int32),
    ('evt_weight', np.float32),
    ('met', np.float32),
    ('met_phi', np.float32),
    ('num_jets', np.int16)
])

jet_dtype = np.dtype([
    ('evt_id', np.int32),
    ('jet_index', np.int16),
    ('num_constits', np.int16),
    ('b_tagged', np.int8),
    ('jet_pt', np.float32),
    ('jet_eta', np.float32),
    ('jet_phi', np.float32)
])

constituent_dtype = np.dtype([
    ('evt_id', np.int32),
    ('jet_index', np.int16),
    ('constituent_index', np.int16),
    ('particle_id', np.int32),
    ('pt', np.float32),
    ('eta', np.float32),
    ('phi', np.float32)
])


@njit
def calculate_jet_properties_numba(pt_arr, eta_arr, phi_arr):
    """
    Numba-accelerated calculation of jet properties using vectorized NumPy operations.
    Processes all constituents in a jet simultaneously.
    
    """
    px = 0.0
    py = 0.0
    pz = 0.0
    for i in range(pt_arr.shape[0]):
        pt = pt_arr[i]
        eta = eta_arr[i]
        phi = phi_arr[i]
        px += pt * np.cos(phi)
        py += pt * np.sin(phi)
        pz += pt * np.sinh(eta)
    jet_pt = np.sqrt(px * px + py * py)
    jet_phi = np.arctan2(py, px)
    jet_eta = np.arcsinh(pz / jet_pt) if jet_pt != 0.0 else 0.0
    return jet_pt, jet_eta, jet_phi


def process_event(evt_id, row):
    """
    Process a single event row:
      - Convert string data to floats.
      - Extract event-level data.
      - Loop over jets and their constituents.
      - Use vectorized (Numba) calculation for jet properties.
    
    """
    try:
        row_floats = [float(x) for x in row]
    except ValueError:
        return None  # Skip events with conversion issues.
    
    row_np = np.array(row_floats, dtype=np.float32)
    evt_weight = row_np[0]
    met = row_np[1]
    met_phi = row_np[2]
    num_jets = int(row_np[3])
    event_data = [evt_id, evt_weight, met, met_phi, num_jets]

    jet_offset = 4  # First 4 columns are event-level variables.
    jets = []
    constituents = []
    
    for i in range(num_jets):
        num_constits = int(row_np[jet_offset])
        b_tagged = int(row_np[jet_offset + 1])
        jet_constituents = []
        for j in range(num_constits):
            base = jet_offset + 2 + j * 4
            pid = row_np[base]
            pt = row_np[base + 1]
            eta = row_np[base + 2]
            phi = row_np[base + 3]
            jet_constituents.append((pt, eta, phi))
            constituents.append((evt_id, i, j, pid, pt, eta, phi))
        if jet_constituents:
            arr = np.array(jet_constituents, dtype=np.float32)
            pt_arr = arr[:, 0]
            eta_arr = arr[:, 1]
            phi_arr = arr[:, 2]
            jet_pt, jet_eta, jet_phi = calculate_jet_properties_numba(pt_arr, eta_arr, phi_arr)
        else:
            jet_pt, jet_eta, jet_phi = 0.0, 0.0, 0.0
        jets.append((evt_id, i, num_constits, b_tagged, jet_pt, jet_eta, jet_phi))
        jet_offset += 2 + num_constits * 4

    return event_data, jets, constituents


def csv_chunk_generator(csv_file, chunk_size=10000):
    """
    Generator yielding CSV file chunks (lists of rows) to avoid loading the entire file.
    Skips empty rows and rows with 'evtwt' in the first column.
    
    """
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header row.
        chunk = []
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue
            if "evtwt" in row[0].lower():
                continue
            chunk.append(tuple(row))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def process_chunk(chunk, start_evt_id):
    """
    Process a chunk of CSV rows:
      - Iterates over each row in the chunk.
      - Applies process_event to extract event, jet, and constituent data.
      - Converts lists to structured NumPy arrays.
    
    """
    events = []
    jets = []
    constituents = []
    evt_id = start_evt_id
    for row in chunk:
        res = process_event(evt_id, row)
        if res is not None:
            ev, js, cons = res
            events.append(tuple(ev))
            jets.extend([tuple(j) for j in js])
            constituents.extend([tuple(c) for c in cons])
        evt_id += 1
    events_arr = np.array(events, dtype=event_dtype) if events else np.empty(0, dtype=event_dtype)
    jets_arr = np.array(jets, dtype=jet_dtype) if jets else np.empty(0, dtype=jet_dtype)
    constituents_arr = np.array(constituents, dtype=constituent_dtype) if constituents else np.empty(0, dtype=constituent_dtype)
    return events_arr, jets_arr, constituents_arr


def append_to_hdf5(h5file, dataset_name, data_chunk):
    """
    Append a data chunk to an existing resizable HDF5 dataset.
    """
    ds = h5file[dataset_name]
    current_size = ds.shape[0]
    new_size = current_size + data_chunk.shape[0]
    ds.resize((new_size,))
    ds[current_size:new_size] = data_chunk


def convert_csv_to_hdf5_npy_parallel(csv_file, output_prefix, out_path, file_type="h5",
                                     chunk_size=10000, n_workers=4, verbose=False):
    """
    
    Convert a CSV file to HDF5 and/or .npy files using chunked processing and Dask for parallelism.
    - Reads the CSV in configurable chunks.
    - Processes chunks in parallel via Dask delayed.
    - For HDF5 output, writes results incrementally with memory-mapped, resizable datasets.
    - For .npy output, accumulates results then saves once processing is complete.
    
    """
    if verbose:
        print(f"Processing {csv_file}...")
        print(f"Processing in parallel using {n_workers} workers...")
    # Schedule processing for each CSV chunk.
    delayed_tasks = []
    start_evt_id = 1
    for chunk in csv_chunk_generator(csv_file, chunk_size):
        delayed_tasks.append(delayed(process_chunk)(chunk, start_evt_id))
        start_evt_id += len(chunk)

    # Compute chunk results in parallel.
    if verbose:
        print(f"Processing {start_evt_id - 1} events in {len(delayed_tasks)} chunks...")
    results = compute(*delayed_tasks, num_workers=n_workers)

    if file_type == "h5":
        h5_filepath = f"{out_path}/{output_prefix}.h5"
        with h5py.File(h5_filepath, 'w') as h5file:
            # Create resizable datasets with gzip compression.
            h5file.create_dataset("events", shape=(0,), maxshape=(None,),
                                  dtype=event_dtype, chunks=True, compression="gzip")
            h5file.create_dataset("jets", shape=(0,), maxshape=(None,),
                                  dtype=jet_dtype, chunks=True, compression="gzip")
            h5file.create_dataset("constituents", shape=(0,), maxshape=(None,),
                                  dtype=constituent_dtype, chunks=True, compression="gzip")
            # Append each processed chunk.
            for events_arr, jets_arr, constituents_arr in results:
                if events_arr.size > 0:
                    append_to_hdf5(h5file, "events", events_arr)
                if jets_arr.size > 0:
                    append_to_hdf5(h5file, "jets", jets_arr)
                if constituents_arr.size > 0:
                    append_to_hdf5(h5file, "constituents", constituents_arr)
        if verbose:
            print(f"Data saved to HDF5 file at {h5_filepath}")
    
    elif file_type == "npy":
        # Accumulate results in memory then save.
        events_list, jets_list, constituents_list = [], [], []
        for events_arr, jets_arr, constituents_arr in results:
            if events_arr.size > 0:
                events_list.append(events_arr)
            if jets_arr.size > 0:
                jets_list.append(jets_arr)
            if constituents_arr.size > 0:
                constituents_list.append(constituents_arr)
        if events_list:
            events_all = np.concatenate(events_list)
            np.save(f"{out_path}/{output_prefix}_events.npy", events_all)
        if jets_list:
            jets_all = np.concatenate(jets_list)
            np.save(f"{out_path}/{output_prefix}_jets.npy", jets_all)
        if constituents_list:
            constituents_all = np.concatenate(constituents_list)
            np.save(f"{out_path}/{output_prefix}_constituents.npy", constituents_all)
        if verbose:
            print(f"Data saved to .npy files with prefix {output_prefix} at {out_path}")


# import sys
# import csv
# import numpy as np
# import h5py
# from loky import get_reusable_executor


# def calculate_jet_properties(constituents):
#     """
#     Calculate jet pT, eta, and phi from constituent properties.

#     Parameters:
#         constituents (list of dicts): Each dict contains constituent properties:
#                                       {'pt': ..., 'eta': ..., 'phi': ...}

#     Returns:
#         dict: Jet properties {'jet_pt': ..., 'jet_eta': ..., 'jet_phi': ...}
#     """
#     px, py, pz, energy = 0.0, 0.0, 0.0, 0.0

#     for c in constituents:
#         pt = c["pt"]
#         eta = c["eta"]
#         phi = c["phi"]

#         px += pt * np.cos(phi)
#         py += pt * np.sin(phi)
#         pz += pt * np.sinh(eta)
#         energy += pt * np.cosh(eta)

#     jet_pt = np.sqrt(px**2 + py**2)
#     jet_phi = np.arctan2(py, px)
#     jet_eta = np.arcsinh(pz / jet_pt) if jet_pt != 0 else 0.0

#     return {
#         "jet_pt": jet_pt,
#         "jet_eta": jet_eta,
#         "jet_phi": jet_phi,
#     }


# def process_event(evt_id, row):
#     """
#     Process a single event, calculating jet and constituent data.

#     Parameters:
#         evt_id (int): Event ID.
#         row (tuple): Row from the csv corresponding to the event.

#     Returns:
#         tuple: (event_data, jets, constituents) for the event.
#     """
#     row = np.array(row, dtype=np.float32)
#     # Extract event-level variables
#     evt_weight = row[0]
#     met = row[1]
#     met_phi = row[2]
#     num_jets = int(row[3])
#     event_data = [evt_id, evt_weight, met, met_phi, num_jets]

#     jet_offset = 4  # First 4 columns are event-level variables
#     jets = []  # Temporary list to hold jet-level data for this event
#     constituents = []

#     for i in range(num_jets):
#         # Extract jet-level variables
#         num_constits = int(row[jet_offset])
#         b_tagged = int(row[jet_offset + 1])

#         # Collect constituent data for this jet
#         jet_constituents = []
#         for j in range(num_constits):
#             pid = row[jet_offset + 2 + j * 4]
#             pt = row[jet_offset + 3 + j * 4]
#             eta = row[jet_offset + 4 + j * 4]
#             phi = row[jet_offset + 5 + j * 4]
#             jet_constituents.append({"pt": pt, "eta": eta, "phi": phi})
#             constituents.append([evt_id, i, j, pid, pt, eta, phi])

#         # Calculate jet properties
#         jet_properties = calculate_jet_properties(jet_constituents)
#         jets.append(
#             [
#                 evt_id,
#                 i,
#                 num_constits,
#                 b_tagged,
#                 jet_properties["jet_pt"],
#                 jet_properties["jet_eta"],
#                 jet_properties["jet_phi"],
#             ]
#         )

#         # Update offset to next jet
#         jet_offset += 2 + num_constits * 4

#     return event_data, jets, constituents


# def convert_csv_to_hdf5_npy_parallel(
#     csv_file,
#     output_prefix,
#     out_path,
#     file_type="h5",
#     n_workers=4,
#     verbose: bool = False,
# ):
#     """
#     Convert a CSV file to HDF5 and .npy files in parallel,
#     adding event ID (evt_id) and jet-level properties calculated from constituents.

#     Parameters:
#         csv_file (str): Path to the input CSV file.
#         output_prefix (str): Prefix for the output files.
#         file_type (str): Output file type ('h5' or 'npy').
#         out_path (str): Path to save output files.
#         n_workers (int): Number of parallel workers.
#         verbose (bool): Print progress if True.
#     """
#     # # Read the CSV file
#     # data = pd.read_csv(csv_file, sep=";")
#     # # Remove rows with all NaN values
#     # data = data.dropna(how="all")

#     # # Parallel processing of events
#     # if verbose:
#     #     # print the file path being parsed
#     #     print(f"Processing {csv_file}...")
#     #     print(f"Processing {len(data)} events in parallel using {n_workers} workers...")
    
#     # Read the CSV file
#     data = []
#     with open(csv_file, 'r', newline="") as f:
#         reader = csv.reader(f, delimiter=",")
#         # Skip header row (column names aren't needed)
#         next(reader, None)
#         for row in reader:
#             # Check if the row is "all NaN" (here, treating empty strings as NaN)
#             if all(cell.strip() == "" for cell in row):
#                 continue
#             # Skip rows where the first column contains the string 'evtwt' anywhere
#             if "evtwt" in row[0].lower():
#                 continue
#             # Convert the row to a tuple so that process_event can access via row[0], row[1], etc.
#             data.append(tuple(row))

#     if verbose:
#         print(f"Processing {csv_file}...")
#         print(f"Processing {len(data)} events in parallel using {n_workers} workers...")


#     # Process events in parallel
#     event_results = []
#     with get_reusable_executor(max_workers=n_workers) as executor:
#         futures = [
#             executor.submit(process_event, evt_id - 1, row)
#             for evt_id, row in enumerate(data, start=1)
#         ]
#         for future in futures:
#             event_results.append(future.result())

#     # Combine results
#     event_data = []
#     jet_data = []
#     constituent_data = []
#     for event, jets, constituents in event_results:
#         event_data.append(event)
#         jet_data.extend(jets)
#         constituent_data.extend(constituents)

#     # Convert to NumPy arrays
#     event_data = np.array(event_data, dtype=np.float32)
#     jet_data = np.array(jet_data, dtype=np.float32)
#     constituent_data = np.array(constituent_data, dtype=np.float32)

#     if file_type == "npy":
#         # Save to .npy files
#         np.save(out_path + f"/{output_prefix}_events.npy", event_data)
#         np.save(out_path + f"/{output_prefix}_jets.npy", jet_data)
#         np.save(out_path + f"/{output_prefix}_constituents.npy", constituent_data)

#     if file_type == "h5":
#         # Save to HDF5 file
#         with h5py.File(out_path + "/" + output_prefix + ".h5", "w") as h5file:
#             h5file.create_dataset("events", data=event_data)
#             h5file.create_dataset("jets", data=jet_data)
#             h5file.create_dataset("constituents", data=constituent_data)

#     if verbose:
#         print(f"Data saved to files with prefix {output_prefix} at {out_path}/")
