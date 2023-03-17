from typing import Any, Dict

from pydicom.dataset import Dataset


def dictify_dicom(ds: Dataset) -> Dict[str, Any]:
    """Extract key, value pairs from Pydicom dataset.

    Args:
        ds: Pydicom dataset.

    Returns:
        A dictionary with DICOM metadata.
    """
    output = dict()
    for elem in ds:
        if elem.VR != "SQ":
            output[elem.name] = str(elem.value)
        else:
            output[elem.name] = [dictify_dicom(item) for item in elem]
    return output
