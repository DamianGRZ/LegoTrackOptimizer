"""Decoder package: chromosome → MultiPathLayout.

Public API re-exported here so callers can keep using
``from src.decoder import decode_chromosome, DecoderConfig``.
"""

from src.decoder.construction import decode_chromosome
from src.decoder.types import DecoderConfig

__all__ = ["decode_chromosome", "DecoderConfig"]
