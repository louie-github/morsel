#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import gcd, pi, sin
from itertools import cycle, islice, repeat
from typing import Iterable, Iterator

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
            The number of channels in each sample of the PCM data.
            Defaults to 2.
        sample_rate (int, optional):
            The sample rate of the PCM data measured in cycles per
            second or Hertz (Hz). Defaults to 48000.
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
    header_chunk_id        = b"RIFF"
    header_chunk_size      = _int_to_riff(chunk_size, 4)
    header_format_         = b"WAVE"  # format_ to avoid conflict with builtin "format"
    header_subchunk_1_id   = b"fmt "
    header_subchunk_1_size = _int_to_riff(16, 4)
    header_audio_format    = _int_to_riff(1, 2)  # PCM = 1
    header_num_channels    = _int_to_riff(num_channels, 2)
    header_sample_rate     = _int_to_riff(sample_rate, 4)
    header_byte_rate       = _int_to_riff(byte_rate, 4)
    header_block_align     = _int_to_riff(block_align, 2)
    header_bits_per_sample = _int_to_riff(bits_per_sample, 2)
    header_subchunk_2_id   = b"data"
    header_subchunk_2_size = _int_to_riff(subchunk_2_size, 4)
    # fmt: on
    return b"".join(
        (
            header_chunk_id,
            header_chunk_size,
            header_format_,
            header_subchunk_1_id,
            header_subchunk_1_size,
            header_audio_format,
            header_num_channels,
            header_sample_rate,
            header_byte_rate,
            header_block_align,
            header_bits_per_sample,
            header_subchunk_2_id,
            header_subchunk_2_size,
        )
    )


def _clamp_floats(floats: Iterable[float]):
    return (max(min(value, 1.0), -1.0) for value in floats)


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
        amplitude * sin(angular_frequency * t / sample_rate) for t in range(num_samples)
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
            An iterable of bytes containing PCM data.
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
    min_cycles = frequency // gcd(frequency, sample_rate)
    min_samples = sample_rate * min_cycles // frequency
    # Don't generate extra samples if duration is shorter than minimum
    # exactly representable duration
    min_samples = min(min_samples, num_samples)
    angular_frequency = 2 * pi * frequency
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
        for channel_data in repeat(data, num_channels)
    )
    output = islice(cycle(exact_cycle), num_samples * num_channels)
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

    See `generate_sine_wave()` for more details on generating sine wave
    data.

    The function will attempt to buffer the individual samples
    generated by `generate_sine_wave()` into *buffer_size* bytes chunks
    before writing them into the file.

    However, *buffer_size* is not guaranteed to be accurate. The
    function will attempt to write up to *buffer_size* bytes at a time
    into the file, but this is not always possible.

    This is because the bit depth of the PCM data might, and commonly
    does, result in individual samples being larger than one byte
    (8 bits). The number of bytes for each sample might not divide
    *buffer_size* exactly.

    If this is the case, the function will use the nearest buffer size
    which can be divided exactly by the number of bytes per sample.


    Args:
        filename (str):
            The filename of the output file.
        buffer_size (int, optional):
            The number of bytes to write to the file at a time. Is not
            guaranteed to be accurate. Defaults to 8192.
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
    samples_per_buffer = buffer_size // bytes_per_sample
    data_iter = iter(data)
    bytes_written = 0
    with open(filename, "wb") as f:
        bytes_written += f.write(header)
        buffer = b"".join(islice(data_iter, samples_per_buffer))
        while buffer:
            bytes_written += f.write(buffer)
            buffer = b"".join(islice(data_iter, samples_per_buffer))
    return bytes_written


# TODO: Add generate_silence


if __name__ == "__main__":
    import io
    import time

    BUFSIZE = io.DEFAULT_BUFFER_SIZE
    filename = "test-5min-512hz-sr48khz-s24le-3b.wav"
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
