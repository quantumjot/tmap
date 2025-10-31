"""Temporal UMAP (tmap) - Temporal sequence embedding library.

tmap provides tools for creating low-dimensional embeddings of temporal sequence
data, respecting the temporal ordering within sequences while aligning similar
sequences across the dataset.

Main classes:
    - TemporalMAP: Temporal extension of UMAP for sequence data
    - DefaultUMAP: Standard UMAP wrapper for comparison
"""
from tmap.temporal import DefaultUMAP, TemporalMAP
