from os_tools import file_list
from typing import List, Union
import os
from os.path import isfile, isdir
import glob
import logging


def resolve_base_structures(paths: Union[str, List[str]], extension: str = ".xyz") -> List[str]:
    """
    Resolve base structure paths, handling file lists and directory scanning.
    
    Args:
        paths: Path(s) to resolve - can be:
               - A single .lst file containing paths (one per line)
               - A directory to scan for files with specified extension
               - A direct path to a file
               - A list of any of the above
        extension: File extension to filter by when scanning directories
    
    Returns:
        List of resolved file paths
    """
    if not paths:
        return []

    if isinstance(paths, str):
        paths = [paths]

    resolved_paths = []

    for path in paths:
        if isfile(path):
            try:
                with open(path, "r") as f:
                    list_paths = [line.strip() for line in f.readlines()]
                    list_paths = [p for p in list_paths if p and not p.startswith("#")]
                    resolved_paths.extend(list_paths)

            except Exception as e:
                logging.error(f"Error reading list file {path}: {str(e)}")
                break
        elif isdir(path):
            list_paths = glob.glob(os.path.join(path, f"*{extension}"))
            resolved_paths.extend(list_paths)
        else:
            logging.warning(f"Path not found: {path}")

    return resolved_paths

