"""JSON serialization for data models."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from ..logging_config import get_logger

logger = get_logger(__name__)


def to_json_dict(obj: Any) -> Any:
    """Convert a dataclass object to a JSON-serializable dictionary.
    
    Recursively handles nested dataclasses, lists, and dictionaries.
    
    Args:
        obj: Object to convert (typically a dataclass instance)
        
    Returns:
        JSON-serializable dictionary
    """
    if is_dataclass(obj):
        return {k: to_json_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [to_json_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_json_dict(v) for k, v in obj.items()}
    else:
        return obj


def write_json(data: Any, output_path: Union[str, Path], indent: int = 2) -> None:
    """Write data to a JSON file.
    
    Automatically handles dataclass objects by converting them to dictionaries.
    
    Args:
        data: Data to write (dataclass, dict, list, etc.)
        output_path: Path to output JSON file
        indent: Indentation level for pretty printing (default: 2)
    """
    output_path = Path(output_path)
    logger.info("Writing JSON data to: %s", output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_data = to_json_dict(data)
    
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=indent)
    
    logger.info("Successfully wrote JSON file: %s", output_path)


def read_json(input_path: Union[str, Path]) -> Any:
    """Read JSON data from a file.
    
    Args:
        input_path: Path to input JSON file
        
    Returns:
        Parsed JSON data (dict, list, etc.)
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    input_path = Path(input_path)
    logger.info("Reading JSON file: %s", input_path)
    
    if not input_path.exists():
        logger.error("JSON file not found: %s", input_path)
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    
    with open(input_path, "r") as f:
        data = json.load(f)
    
    logger.info("Successfully read JSON file: %s", input_path)
    return data
