import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import base64
    import io
    import math
    import os
    import marimo as mo
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile

    return base64, io, math, mo, np, os, signal, wavfile


@app.cell
def _(mo):
    mo.md("""
    # Resampling Studio
    Select a pre-provided sample track or upload a WAV audio file, choose a target sampling rate, and listen to the resampled output using pure Python/SciPy DSP.
    """)
    return


@app.cell
def _(mo):
    sample_tracks = {
        "Classical Strings": "samples/classical_strings.wav",
        "Drum Beat": "samples/drums_beat.wav",
        "Solo Trumpet": "samples/solo_trumpet.wav",
        "Speech Voice": "samples/speech_voice.wav",
        "Jazz Vibes": "samples/jazz_vibes.wav",
    }

    sample_rate_options = {
        "44,100 Hz (44.1 kHz - CD Standard)": 44100,
        "32,000 Hz (32 kHz - Broadcast Quality)": 32000,
        "22,050 Hz (22.05 kHz - FM Radio Quality)": 22050,
        "16,000 Hz (16 kHz - Wideband Speech)": 16000,
        "11,025 Hz (11.025 kHz - Low Bitrate Audio)": 11025,
        "8,000 Hz (8 kHz - Telephone Quality)": 8000,
        "4,000 Hz (4 kHz - Vintage Lo-Fi / Walkie-Talkie)": 4000,
    }

    source_type = mo.ui.radio(
        options=["Pre-provided Sample", "Upload Audio File"],
        value="Pre-provided Sample",
        label="**Audio Source**",
    )

    sample_selector = mo.ui.dropdown(
        options=sample_tracks,
        value="Classical Strings",
        label="**Sample Track**",
    )

    file_uploader = mo.ui.file(
        filetypes=[".wav"],
        multiple=False,
        label="Upload Audio File (.wav)",
    )

    sample_rate_selector = mo.ui.dropdown(
        options=sample_rate_options,
        value="44,100 Hz (44.1 kHz - CD Standard)",
        label="**Target Sampling Rate**",
    )
    return (
        file_uploader,
        sample_rate_selector,
        sample_selector,
        sample_tracks,
        source_type,
    )


@app.cell
def _(file_uploader, mo, sample_rate_selector, sample_selector, source_type):
    if source_type.value == "Pre-provided Sample":
        _track_input = sample_selector
    else:
        _track_input = file_uploader

    _controls = mo.vstack([
        source_type,
        _track_input,
        sample_rate_selector,
    ], gap=1)

    mo.callout(
        _controls,
        kind="neutral",
    )
    return


@app.cell
def _(
    base64,
    file_uploader,
    io,
    math,
    mo,
    np,
    os,
    sample_rate_selector,
    sample_selector,
    sample_tracks,
    signal,
    source_type,
    wavfile,
):
    def _extract_audio_metadata(input_bytes_or_path):
        if isinstance(input_bytes_or_path, str):
            with open(input_bytes_or_path, "rb") as _f:
                _raw_bytes = _f.read()
        else:
            _raw_bytes = input_bytes_or_path

        _bio = io.BytesIO(_raw_bytes)
        _sr, _data = wavfile.read(_bio)

        _channels = 1 if _data.ndim == 1 else _data.shape[1]
        _num_samples = len(_data)
        _duration = _num_samples / _sr
        _bit_depth = _data.dtype.itemsize * 8
        _bitrate = int((_sr * _channels * _bit_depth) / 1000)

        return {
            "sample_rate": _sr,
            "channels": _channels,
            "duration": _duration,
            "bit_depth": _bit_depth,
            "bitrate_kbps": _bitrate,
            "size_bytes": len(_raw_bytes),
            "format_name": "WAV (PCM)",
        }


    def _resample_audio(input_bytes_or_path, target_sr):
        if isinstance(input_bytes_or_path, str):
            with open(input_bytes_or_path, "rb") as _f:
                _bio_in = io.BytesIO(_f.read())
        else:
            _bio_in = io.BytesIO(input_bytes_or_path)

        _orig_sr, _data = wavfile.read(_bio_in)

        if _orig_sr == target_sr:
            _bio_in.seek(0)
            return _bio_in.read()

        _orig_dtype = _data.dtype
        if np.issubdtype(_orig_dtype, np.integer):
            _max_val = np.iinfo(_orig_dtype).max
            _audio_float = _data.astype(np.float32) / _max_val
        else:
            _audio_float = _data.astype(np.float32)

        _gcd = math.gcd(int(target_sr), int(_orig_sr))
        _up = int(target_sr // _gcd)
        _down = int(_orig_sr // _gcd)

        _resampled_float = signal.resample_poly(_audio_float, _up, _down, axis=0)
        _clipped = np.clip(_resampled_float, -1.0, 1.0)
        _out_pcm16 = (_clipped * 32767).astype(np.int16)

        _bio_out = io.BytesIO()
        wavfile.write(_bio_out, target_sr, _out_pcm16)
        return _bio_out.getvalue()


    if source_type.value == "Pre-provided Sample":
        _audio_source = sample_selector.value
        _track_title = next((k for k, v in sample_tracks.items() if v == _audio_source), "Sample Track")
        _has_audio = bool(_audio_source and os.path.exists(_audio_source))
    else:
        _audio_source = file_uploader.contents(0)
        _track_title = file_uploader.name(0) or "Uploaded Audio"
        _has_audio = bool(_audio_source)

    if not _has_audio:
        player_view = mo.callout(
            mo.md("Please select or upload a WAV audio file to begin playback."),
            kind="info",
        )
    else:
        try:
            _target_sr = sample_rate_selector.value
            _meta = _extract_audio_metadata(_audio_source)
            _orig_sr = _meta["sample_rate"]
            _orig_size_bytes = _meta["size_bytes"]
            _orig_size_kb = _orig_size_bytes / 1024
            _orig_bitrate = _meta["bitrate_kbps"]

            # Resample audio using pure SciPy polyphase filtering
            _resampled_bytes = _resample_audio(_audio_source, _target_sr)
            _new_size_bytes = len(_resampled_bytes)
            _new_size_kb = _new_size_bytes / 1024
            _resampled_meta = _extract_audio_metadata(_resampled_bytes)
            _new_bitrate = _resampled_meta["bitrate_kbps"]

            # Compute percent compression vs original file
            _file_compression_pct = (1.0 - (_new_size_bytes / _orig_size_bytes)) * 100
            if _file_compression_pct > 0.05:
                _size_compression_text = f"{_file_compression_pct:.1f}% smaller"
            elif _file_compression_pct < -0.05:
                _size_compression_text = f"+{abs(_file_compression_pct):.1f}% larger"
            else:
                _size_compression_text = "0.0% (same size)"

            # Compute sample rate reduction ratio
            _sr_reduction_pct = (1.0 - (_target_sr / _orig_sr)) * 100
            if _sr_reduction_pct > 0.05:
                _sr_reduction_text = f"{_sr_reduction_pct:.1f}% reduction"
            elif _sr_reduction_pct < -0.05:
                _sr_reduction_text = f"+{abs(_sr_reduction_pct):.1f}% upsampled"
            else:
                _sr_reduction_text = "0.0% (original rate)"

            _orig_nyquist = _orig_sr / 2000.0
            _target_nyquist = _target_sr / 2000.0

            # Base64 Data URI for HTML5 audio playback
            _b64 = base64.b64encode(_resampled_bytes).decode("ascii")
            _data_uri = f"data:audio/wav;base64,{_b64}"
            _player = mo.audio(src=_data_uri)

            _metadata_md = f"""
    ### Now Playing: {_track_title}

    | Metric | Original Audio | Resampled Output (Pure SciPy DSP) |
    | :--- | :--- | :--- |
    | **Sampling Rate** | {_orig_sr:,} Hz | **{_target_sr:,} Hz** |
    | **Frequency Bandwidth (Nyquist Cutoff)** | {_orig_nyquist:.1f} kHz | **{_target_nyquist:.2f} kHz** |
    | **Bitrate (Uncompressed PCM)** | {_orig_bitrate} kbps | **{_new_bitrate} kbps** |
    | **File Size** | {_orig_size_kb:.1f} KB | **{_new_size_kb:.1f} KB (WAV)** |
    | **Sample Rate Reduction** | Baseline (1.0x) | **{_sr_reduction_text}** |
    | **File Size Comparison** | Baseline | **{_size_compression_text}** |
    | **Channels & Duration** | {_meta['channels']} channel(s) | {_meta['duration']:.1f}s |
    """

            player_view = mo.vstack([
                mo.md(_metadata_md),
                _player,
            ], gap=1)
        except Exception as _e:
            player_view = mo.callout(
                mo.md(f"**Error processing audio**: `{_e}`\n\nPlease ensure the uploaded file is a valid standard WAV audio file."),
                kind="danger",
            )

    player_view
    return


if __name__ == "__main__":
    app.run()
