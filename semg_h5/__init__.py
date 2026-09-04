"""Canonical HDF5 conversion tools for public Meta sEMG datasets."""

from .canonical import CanonicalRecord, validate_record

__all__ = ["CanonicalRecord", "validate_record"]
