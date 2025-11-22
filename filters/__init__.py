"""Filter registry and management."""

from .base import BaseFilter
from .sobel import SobelEdgeFilter
from .cartoon import CartoonFilter
from .thermal import ThermalVisionFilter
from .pixelation import PixelationFilter


# Filter registry - maps filter name to filter class
FILTERS = {
    'sobel': SobelEdgeFilter,
    'cartoon': CartoonFilter,
    'thermal': ThermalVisionFilter,
    'pixel': PixelationFilter,
}


def get_filter_class(name: str):
    """
    Get filter class by name.

    Args:
        name: Filter name

    Returns:
        Filter class

    Raises:
        KeyError: If filter name not found
    """
    if name not in FILTERS:
        raise KeyError(f"Unknown filter: {name}. Available filters: {list_filters()}")
    return FILTERS[name]


def list_filters():
    """
    Get list of available filter names.

    Returns:
        List of filter names
    """
    return list(FILTERS.keys())


def get_filter_info(name: str):
    """
    Get information about a filter.

    Args:
        name: Filter name

    Returns:
        Dictionary with filter metadata
    """
    filter_class = get_filter_class(name)
    return {
        'name': filter_class.name,
        'description': filter_class.description,
        'parameters': filter_class.get_parameters()
    }


def create_filter(name: str, **kwargs):
    """
    Create a filter instance with given parameters.

    Args:
        name: Filter name
        **kwargs: Filter parameters

    Returns:
        Filter instance
    """
    filter_class = get_filter_class(name)
    return filter_class(**kwargs)


__all__ = [
    'BaseFilter',
    'SobelEdgeFilter',
    'CartoonFilter',
    'ThermalVisionFilter',
    'PixelationFilter',
    'FILTERS',
    'get_filter_class',
    'list_filters',
    'get_filter_info',
    'create_filter',
]
