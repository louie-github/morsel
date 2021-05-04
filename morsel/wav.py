#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# TODO: Add docstrings and __repr__() methods
# Currently, this script only supports exporting s16le, s24le, and s32le
# wav files.
SUPPORTED_BIT_DEPTHS = 16, 24, 32


def _int_to_field(i: int, length: int) -> None:
    """Convert int objects to RIFF fields (unsigned little-endian)

    Args:
        i (int): The integer to convert.
        length (int): The number of bytes used to represent the number.

    Returns:
        None
    """
    return i.to_bytes(length, "little", signed=False)


def generate_pcm_wave_file_header(
    *,
    num_samples: int,
    num_channels: int = 2,
    sample_rate: int = 48000,
    bits_per_sample: int = 16,
):
    if bits_per_sample not in {16, 24, 32}:
        raise ValueError(
            " ".join(
                (
                    "Bit depths other than 16, 24, or 32 are currently",
                    f"not supported (got {bits_per_sample})",
                )
            )
        )
    # TODO: Add error-checking (validation) for values such as
    # bounds checking, and multiple-of-8 verification.
    # Calculate some fields which depend on other fields' values
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    subchunk_2_size = num_samples * num_channels * bits_per_sample // 8
    chunk_size = 36 + subchunk_2_size
    # Set RIFF fields
    # fmt: off
    chunk_id        = b"RIFF"
    chunk_size      = _int_to_field(chunk_size, 4)
    format_         = b"WAVE"  # format_ to avoid conflict with builtin "format"
    subchunk_1_id   = b"fmt "
    subchunk_1_size = _int_to_field(16, 4)
    audio_format    = _int_to_field(1, 2)  # PCM = 1
    num_channels    = _int_to_field(num_channels, 2)
    sample_rate     = _int_to_field(sample_rate, 4)
    byte_rate       = _int_to_field(byte_rate, 4)
    block_align     = _int_to_field(block_align, 2)
    bits_per_sample = _int_to_field(bits_per_sample, 2)
    subchunk_2_id   = b"data"
    subchunk_2_size = _int_to_field(subchunk_2_size, 4)
    # fmt: on
    return b"".join(
        (
            chunk_id,
            chunk_size,
            format_,
            subchunk_1_id,
            subchunk_1_size,
            audio_format,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            subchunk_2_id,
            subchunk_2_size,
        )
    )
