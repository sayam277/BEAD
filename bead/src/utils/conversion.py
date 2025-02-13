import pandas as pd
import numpy as np
import h5py
from loky import get_reusable_executor


def calculate_jet_properties(constituents):
    """
    Calculate jet pT, eta, and phi from constituent properties.

    Parameters:
        constituents (list of dicts): Each dict contains constituent properties:
                                      {'pt': ..., 'eta': ..., 'phi': ...}

    Returns:
        dict: Jet properties {'jet_pt': ..., 'jet_eta': ..., 'jet_phi': ...}
    """
    px, py, pz, energy = 0.0, 0.0, 0.0, 0.0

    for c in constituents:
        pt = c["pt"]
        eta = c["eta"]
        phi = c["phi"]

        px += pt * np.cos(phi)
        py += pt * np.sin(phi)
        pz += pt * np.sinh(eta)
        energy += pt * np.cosh(eta)

    jet_pt = np.sqrt(px**2 + py**2)
    jet_phi = np.arctan2(py, px)
    jet_eta = np.arcsinh(pz / jet_pt) if jet_pt != 0 else 0.0

    return {
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_phi": jet_phi,
    }


def process_event(evt_id, row):
    """
    Process a single event, calculating jet and constituent data.

    Parameters:
        evt_id (int): Event ID.
        row (pandas.Series): Row from the DataFrame corresponding to the event.

    Returns:
        tuple: (event_data, jets, constituents) for the event.
    """
    # Extract event-level variables
    evt_weight = row[1]
    met = row[2]
    met_phi = row[3]
    num_jets = int(row[4])
    event_data = [evt_id, evt_weight, met, met_phi, num_jets]

    # #print row[4] to debug
    # print(row[4])

    jet_offset = 5  # First 4 columns are event-level variables
    jets = []  # Temporary list to hold jet-level data for this event
    constituents = []

    for i in range(num_jets):
        # Extract jet-level variables
        num_constits = int(row[jet_offset])
        b_tagged = int(row[jet_offset + 1])

        # Collect constituent data for this jet
        jet_constituents = []
        for j in range(num_constits):
            pid = row[jet_offset + 2 + j * 4]
            pt = row[jet_offset + 3 + j * 4]
            eta = row[jet_offset + 4 + j * 4]
            phi = row[jet_offset + 5 + j * 4]
            jet_constituents.append({"pt": pt, "eta": eta, "phi": phi})
            constituents.append([evt_id, i, j, pid, pt, eta, phi])

        # Calculate jet properties
        jet_properties = calculate_jet_properties(jet_constituents)
        jets.append(
            [
                evt_id,
                i,
                num_constits,
                b_tagged,
                jet_properties["jet_pt"],
                jet_properties["jet_eta"],
                jet_properties["jet_phi"],
            ]
        )

        # Update offset to next jet
        jet_offset += 2 + num_constits * 4

    # Reorder jets, constituents in decreasing order of pT
    # jets.sort(key=lambda x: -x[4])  # Sort by jet_pt (index 4)
    # constituents.sort(key=lambda x: -x[4])  # Sort by constit_pt (index 4)
    return event_data, jets, constituents


def convert_csv_to_hdf5_npy_parallel(
    csv_file,
    output_prefix,
    out_path,
    file_type="h5",
    n_workers=4,
    verbose: bool = False,
):
    """
    Convert a CSV file to HDF5 and .npy files in parallel,
    adding event ID (evt_id) and jet-level properties calculated from constituents.

    Parameters:
        csv_file (str): Path to the input CSV file.
        output_prefix (str): Prefix for the output files.
        file_type (str): Output file type ('h5' or 'npy').
        out_path (str): Path to save output files.
        n_workers (int): Number of parallel workers.
        verbose (bool): Print progress if True.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)  # , on_bad_lines='skip')
    # Remove rows with all NaN values
    data = data.dropna(how="all")

    # Parallel processing of events
    if verbose:
        # print the file path being parsed
        print(f"Processing {csv_file}...")
        print(f"Processing {len(data)} events in parallel using {n_workers} workers...")
    event_results = []
    with get_reusable_executor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_event, evt_id - 1, row)
            for evt_id, row in enumerate(data.itertuples(), start=1)
        ]
        for future in futures:
            event_results.append(future.result())

    # Combine results
    event_data = []
    jet_data = []
    constituent_data = []
    for event, jets, constituents in event_results:
        event_data.append(event)
        jet_data.extend(jets)
        constituent_data.extend(constituents)

    # Convert to NumPy arrays
    event_data = np.array(event_data, dtype=np.float32)
    jet_data = np.array(jet_data, dtype=np.float32)
    constituent_data = np.array(constituent_data, dtype=np.float32)

    if file_type == "npy":
        # Save to .npy files
        np.save(out_path + f"/{output_prefix}_events.npy", event_data)
        np.save(out_path + f"/{output_prefix}_jets.npy", jet_data)
        np.save(out_path + f"/{output_prefix}_constituents.npy", constituent_data)

    if file_type == "h5":
        # Save to HDF5 file
        with h5py.File(out_path + "/" + output_prefix + ".h5", "w") as h5file:
            h5file.create_dataset("events", data=event_data)
            h5file.create_dataset("jets", data=jet_data)
            h5file.create_dataset("constituents", data=constituent_data)

    if verbose:
        print(f"Data saved to files with prefix {output_prefix} at {out_path}/")
