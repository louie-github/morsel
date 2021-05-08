from .generators import write_sine_wave_wav_file


def test_sine():
    import io
    import time

    buffer_size = io.DEFAULT_BUFFER_SIZE
    filename = "test-5min-12khz-sr48khz-s24le-pcmdatagen.wav"
    frequency = 12_000
    sample_rate = 48000
    duration = 5 * 60 * sample_rate  # 5 minutes
    bit_depth = 24

    start_time = time.time()
    with open(filename, "wb") as fp:
        write_sine_wave_wav_file(
            fp=fp,
            frequency=frequency,
            buffer_size=buffer_size,
            sample_rate=sample_rate,
            num_samples=duration,
            bits_per_sample=bit_depth,
        )
    end_time = time.time()

    print(f"Time taken: {end_time - start_time}")


def main():
    return test_sine()


if __name__ == "__main__":
    main()
