import json


def load_electrode_positions_and_fiducials(scanID=None):
    """Load electrode and fiducial data from JSON files."""
    if scanID is None:
        raise ValueError("scanID must be provided to load the correct files.")

    with open(f"data/json/electrode_positions_{scanID}.json") as f:
        electrodes = json.load(f)

    with open(f"data/json/fiducials_{scanID}.json") as f:
        fiducials = json.load(f)

    return electrodes, fiducials
