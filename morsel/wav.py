#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import io
import itertools

from typing import Iterable

# TODO: Add module docstrings, __repr__() methods, and unit tests

# Currently, this script only supports exporting s16le, s24le, and s32le
# wav files.
DEFAULT_BIT_DEPTH = 16
SUPPORTED_BIT_DEPTHS = {16, 24, 32}


def _int_to_riff(i: int, length: int) -> bytes:
    """Convert an int to its byte representation in a RIFF file.

    Represents integers as unsigned integers in *length* bytes encoded
    in little-endian.

    Args:
        i (int):
            The integer to represent.
        length (int):
            The number of bytes used to represent the integer.

    Returns:
        bytes:
            The bytes representation of the specified integer.
    """
    return i.to_bytes(length, "little", signed=False)


def generate_pcm_wave_file_header(
    *,
    num_samples: int,
    num_channels: int = 2,
    sample_rate: int = 48000,
    bits_per_sample: int = DEFAULT_BIT_DEPTH,
) -> bytes:
    """Generates a PCM WAVE file header.

    Args:
        num_samples (int):
            The number of samples in the PCM data.
        num_channels (int, optional):
            The number of channels in the PCM data. Defaults to 2.
        sample_rate (int, optional):
            The sample rate of the PCM data. Defaults to 48000.
        bits_per_sample (int, optional):
            The bit depth or number of bits per sample in the PCM data.
            Defaults to DEFAULT_BIT_DEPTH.

    Raises:
        ValueError:
            An unsupported bit depth was specified.

    Returns:
        bytes:
            The PCM wave file header.
    """
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
    chunk_size      = _int_to_riff(chunk_size, 4)
    format_         = b"WAVE"  # format_ to avoid conflict with builtin "format"
    subchunk_1_id   = b"fmt "
    subchunk_1_size = _int_to_riff(16, 4)
    audio_format    = _int_to_riff(1, 2)  # PCM = 1
    num_channels    = _int_to_riff(num_channels, 2)
    sample_rate     = _int_to_riff(sample_rate, 4)
    byte_rate       = _int_to_riff(byte_rate, 4)
    block_align     = _int_to_riff(block_align, 2)
    bits_per_sample = _int_to_riff(bits_per_sample, 2)
    subchunk_2_id   = b"data"
    subchunk_2_size = _int_to_riff(subchunk_2_size, 4)
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


def _map_floats_to_ints(floats: Iterable[float], bits_per_sample: int) -> Iterable[int]:
    """Maps float values in the range -1.0 to 1.0 to integer values.

    The integer range is determined by *bits_per_sample*. The range is
    defined as the minimum and maximum values representable using
    signed integers of size *bits_per_sample*.

    Args:
        floats (Iterable[float]):
            The float values to map to integers.
        bits_per_sample (int):
            The bit depth used to determine the output integer range.

    Returns:
        Iterable[int]:
            The mapped integer values.
    """
    max_signed = 2 ** (bits_per_sample - 1) - 1
    min_signed = ~max_signed
    # Every x increase from the minimum float value maps to this value
    # above the minimum int value
    step_size = (max_signed - min_signed) / (1.0 - (-1.0))
    return (round((value - (-1.0)) * step_size + min_signed) for value in floats)


def _clamp_floats(floats: Iterable[float]):
    return (max(min(value, 1.0), -1.0) for value in floats)


def _generate_sine_wave(
    *, angular_frequency: float, amplitude: float, num_samples: int, sample_rate: int
) -> Iterable[float]:
    """Generates samples of a sine wave function.

    Args:
        angular_frequency (float):
            The angular frequency of the function or rate of change in
            radians per second.
        amplitude (float):
            The amplitude of the function or the maximum deviation from
            zero.
        num_samples (int):
            The number of samples to generate.
        sample_rate (int):
            The sample rate of the generated data, in cycles per second
            or Hertz (Hz).

    Returns:
        Iterable[float]:
            The generated samples of the sine wave function.
    """
    return (
        amplitude * math.sin(angular_frequency * t / sample_rate)
        for t in range(num_samples)
    )


def generate_sine_wave(
    *,
    frequency: int = 500,
    amplitude: float = 0.75,
    num_samples: int = 48000,
    num_channels: int = 2,
    sample_rate: int = 48000,
    bits_per_sample: int = DEFAULT_BIT_DEPTH,
    allow_clipping=True,
) -> Iterable[bytes]:
    """Generates PCM data for a sine wave.
    All channels are filled with the same data.

    If `allow_clipping = True`, then any floating point values that
    fall outside the allowed range (-1.0 to 1.0) will be clamped to
    either -1.0 or 1.0, whichever is appropriate.

    Args:
        frequency (int, optional):
            The frequency of the sine wave in cycles per second or
            Hertz (Hz). Defaults to 500.
        amplitude (float, optional):
            The amplitude of the sine wave or maximum deviation from
            zero. Defaults to 0.75.
        num_samples (int, optional):
            The number of samples to generate. Defaults to 48000.
        num_channels (int, optional):
            The number of channels of the PCM data. Defaults to 2.
        sample_rate (int, optional):
            The sample rate of the PCM data. Defaults to 48000.
        bits_per_sample (int, optional):
            The bit depth / number of bits per sample in the PCM data.
            Defaults to DEFAULT_BIT_DEPTH.
        allow_clipping (bool, optional):
            Whether to allow clipping of output values. Defaults to True.

    Raises:
        ValueError:
            An unsupported bit depth was specified.
        ValueError:
            An amplitude smaller than 0.0 was specified.
        ValueError:
            An amplitude larger than 1.0 was specified while clipping
            was not allowed.

    Returns:
        Iterable[bytes]:
            An iterable of the PCM data of the sine wave.
    """
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
    if allow_clipping:
        exact_cycle = _clamp_floats(exact_cycle)
    exact_cycle = _map_floats_to_ints(exact_cycle, bits_per_sample=bits_per_sample)
    bytes_per_sample = bits_per_sample // 8
    exact_cycle = (
        i.to_bytes(bytes_per_sample, "little", signed=True) for i in exact_cycle
    )
    exact_cycle = (
        channel_data
        for data in exact_cycle
        for channel_data in itertools.repeat(data, num_channels)
    )
    output = itertools.islice(itertools.cycle(exact_cycle), num_samples * num_channels)
    return output


def generate_sine_wave_file(
    filename: str,
    buffer_size: int = 8192,
    *,
    frequency: int = 500,
    amplitude: float = 0.75,
    num_samples: int = 48000,
    num_channels: int = 2,
    sample_rate: int = 48000,
    bits_per_sample: int = DEFAULT_BIT_DEPTH,
    allow_clipping=True,
) -> int:
    """Generates a PCM WAVE (.wav) file containing a sine wave.

    See `generate_sine_wave()` for more details.

    Args:
        filename (str):
            The filename of the output file.
        buffer_size (int, optional):
            The number of bytes to write to the file at a time.
            Defaults to 8192.
        frequency (int, optional):
            The frequency of the sine wave in cycles per second or
            Hertz (Hz). Defaults to 500.
        amplitude (float, optional):
            The amplitude of the sine wave or maximum deviation from
            zero. Defaults to 0.75.
        num_samples (int, optional):
            The number of samples to generate. Defaults to 48000.
        num_channels (int, optional):
            The number of channels of the PCM data. Defaults to 2.
        sample_rate (int, optional):
            The sample rate of the PCM data. Defaults to 48000.
        bits_per_sample (int, optional):
            The bit depth / number of bits per sample in the PCM data.
            Defaults to DEFAULT_BIT_DEPTH.
        allow_clipping (bool, optional):
            Whether to allow clipping of output values. Defaults to True.


    Returns:
        int:
            The number of bytes written.
    """
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
        allow_clipping=allow_clipping,
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


# TODO: Add generate_silence


if __name__ == "__main__":
    import time

    BUFSIZE = io.DEFAULT_BUFFER_SIZE
    filename = "test-5min-512hz-sr48khz-s24le-3.wav"
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
