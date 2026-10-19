"""Preprocessing and dataset preparation module.

Provides dataset loaders, image pairing, validation, and controlled
synthetic haze synthesis based on Koschmieder's physical scattering model.
"""

from src.preprocessing.dataset import PairedHazeDataset, create_paired_dataloader
from src.preprocessing.ingestion import ImagePair, IngestionReport, ResideDatasetIngestor
from src.preprocessing.synthetic import (
    create_transmission_map,
    generate_synthetic_haze,
)

__all__ = [
    "PairedHazeDataset",
    "create_paired_dataloader",
    "create_transmission_map",
    "generate_synthetic_haze",
    "ResideDatasetIngestor",
    "ImagePair",
    "IngestionReport",
]
