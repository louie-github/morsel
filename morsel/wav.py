#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import io
import itertools
from typing import Iterable, Iterator

# TODO: Add docstrings, __repr__() methods, and unittests

# Currently, this script only supports exporting s16le, s24le, and s32le
# wav files.
SUPPORTED_BIT_DEPTHS = {16, 24, 32}


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
) -> bytes:
    if bits_per_sample not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(
            " ".join(
                (
                    "Bit depths other than 16, 24, or 32 are currently",
                    f"not supported (requested {bits_per_sample})",
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


def _map_floats_to_ints(floats: Iterable[float], bits_per_sample: int):
    max_signed = 2 ** (bits_per_sample - 1) - 1
    min_signed = ~max_signed
    # Every x increase from the minimum float value maps to this value
    # above the minimum int value
    step_size = (max_signed - min_signed) / (1.0 - (-1.0))
    return (round((value - (-1.0)) * step_size + min_signed) for value in floats)


def _generate_sine_wave(
    *, angular_frequency: float, amplitude: float, num_samples: int, sample_rate: int
):
    return (
        amplitude * math.sin(angular_frequency * t / sample_rate)
        for t in range(num_samples)
    )


def generate_sine_wave(
    frequency: int = 500,
    amplitude: float = 0.75,
    num_samples: int = 48000,
    num_channels: int = 2,
    sample_rate: int = 48000,
    bits_per_sample: int = 16,
    allow_clipping=True,
) -> Iterator:
    if bits_per_sample not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(
            " ".join(
                (
                    "Bit depths other than 16, 24, or 32 are currently",
                    f"not supported (requested {bits_per_sample})",
                )
            )
        )

    if amplitude < 0.0:
        raise ValueError("Given amplitude {amplitude} is smaller than 0.0.")
    elif amplitude > 1.0 and not allow_clipping:
        raise ValueError(
            " ".join(
                (
                    f"Given amplitude {amplitude} is larger than 1.0",
                    "and clipping was not allowed.",
                )
            )
        )
    # Algorithm: Calculate the minimum amount of cycles whose duration
    # can be represented exactly by the given sample rate.
    # This avoids extra rounding errors and is faster. Otherwise, values
    # that map to 0 might instead map to -1 due to said errors.
    min_cycles = frequency // math.gcd(frequency, sample_rate)
    min_samples = sample_rate * min_cycles // frequency
    # Don't generate extra samples if duration is shorter than minimum
    # exactly representable duration
    min_samples = min(min_samples, num_samples)
    angular_frequency = 2 * math.pi * frequency
    exact_cycle = _generate_sine_wave(
        angular_frequency=angular_frequency,
        amplitude=amplitude,
        num_samples=min_samples,
        sample_rate=sample_rate,
    )
    # Convert raw float data to integer PCM data with interleaved channels
    exact_cycle = _map_floats_to_ints(exact_cycle, bits_per_sample=bits_per_sample)
    bytes_per_sample = bits_per_sample // 8
    exact_cycle = (
        i.to_bytes(bytes_per_sample, "little", signed=True) for i in exact_cycle
    )
    exact_cycle = (
        data for value in exact_cycle for data in itertools.repeat(value, num_channels)
    )
    output = itertools.islice(itertools.cycle(exact_cycle), num_samples * num_channels)
    return output


def generate_sine_wave_file(
    filename: str,
    buffer_size: int = 8192,
    frequency: int = 500,
    amplitude: float = 0.75,
    num_samples: int = 48000,
    num_channels: int = 2,
    sample_rate: int = 48000,
    bits_per_sample: int = 16,
    allow_clipping=True,
) -> int:
    header = generate_pcm_wave_file_header(
        num_samples=num_samples,
        num_channels=num_channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
    )
    data = generate_sine_wave(
        frequency=frequency,
        amplitude=amplitude,
        num_samples=num_samples,
        num_channels=num_channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
    )
    bytes_per_sample = bits_per_sample // 8
    chunk_size = buffer_size // bytes_per_sample
    data_iter = iter(data)
    bytes_written = 0
    with open(filename, "wb") as f:
        bytes_written += f.write(header)
        finished = False
        while not finished:
            buffer = bytearray()
            for _ in itertools.repeat(None, chunk_size):
                try:
                    buffer.extend(next(data_iter))
                except StopIteration:
                    finished = True
                    break
            bytes_written += f.write(buffer)
    return bytes_written


if __name__ == "__main__":
    import time

    BUFSIZE = io.DEFAULT_BUFFER_SIZE
    filename = "test-5min-512hz-sr48khz-s24le.wav"
    frequency = 512
    sample_rate = 48000
    duration = 5 * 60 * sample_rate  # 5 minutes
    bit_depth = 24

    start_time = time.time()
    generate_sine_wave_file(
        filename=filename,
        frequency=frequency,
        sample_rate=sample_rate,
        num_samples=duration,
        bits_per_sample=bit_depth,
    )
    end_time = time.time()

    print(f"Time taken: {end_time - start_time}")
