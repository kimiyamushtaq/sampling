# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.16",
#     "numpy==2.2.6",
#     "scipy==1.15.3",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import base64
    import io
    import math
    import os
    import urllib.request
    import marimo as mo
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile

    return base64, io, math, mo, np, os, signal, urllib, wavfile


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
    urllib,
    wavfile,
):
    def _generate_procedural_sample(name):
        sr = 44100
        dur = 4.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        name_lower = str(name).lower()

        if "string" in name_lower:
            chords = [
                ([261.63, 329.63, 392.00, 523.25], 1.0),
                ([196.00, 246.94, 293.66, 392.00], 1.0),
                ([220.00, 261.63, 329.63, 440.00], 1.0),
                ([174.61, 220.00, 261.63, 349.23], 1.0),
            ]
            audio = np.zeros_like(t)
            step = int(sr * 1.0)
            vibrato = 1.0 + 0.008 * np.sin(2 * np.pi * 5.5 * t)
            for i, (freqs, _) in enumerate(chords):
                start = i * step
                end = min(start + step, len(t))
                sub_t = t[start:end] - t[start]
                env = np.sin(np.pi * sub_t / 1.0) ** 0.5
                sig = np.zeros(len(sub_t))
                for f in freqs:
                    sig += 0.35 * np.sin(2 * np.pi * f * vibrato[start:end] * sub_t)
                    sig += 0.20 * np.sin(2 * np.pi * (2 * f) * vibrato[start:end] * sub_t)
                    sig += 0.12 * np.sin(2 * np.pi * (3 * f) * sub_t)
                audio[start:end] = sig * env

        elif "drum" in name_lower:
            audio = np.zeros_like(t)
            beat_len = int(sr * 0.5)
            for beat in range(8):
                start = beat * beat_len
                hh_len = int(sr * 0.06)
                hh_t = np.linspace(0, 0.06, hh_len, endpoint=False)
                hh = np.random.randn(hh_len) * np.exp(-hh_t * 60) * 0.2
                audio[start:start + hh_len] += hh
                if beat % 2 == 0:
                    k_len = int(sr * 0.25)
                    k_t = np.linspace(0, 0.25, k_len, endpoint=False)
                    k_freq = 150 * np.exp(-k_t * 25) + 40
                    kick = np.sin(2 * np.pi * np.cumsum(k_freq) / sr) * np.exp(-k_t * 12) * 0.8
                    audio[start:start + k_len] += kick
                else:
                    s_len = int(sr * 0.2)
                    s_t = np.linspace(0, 0.2, s_len, endpoint=False)
                    snare = (np.sin(2 * np.pi * 180 * s_t) * 0.4 + np.random.randn(s_len) * 0.5) * np.exp(-s_t * 18)
                    audio[start:start + s_len] += snare

        elif "jazz" in name_lower:
            chords = [
                [261.63, 329.63, 392.00, 493.88],
                [220.00, 261.63, 329.63, 392.00],
                [293.66, 349.23, 440.00, 523.25],
                [196.00, 246.94, 293.66, 349.23],
            ]
            audio = np.zeros_like(t)
            step = int(sr * 1.0)
            tremolo = 1.0 + 0.15 * np.sin(2 * np.pi * 6.0 * t)
            for i, chord in enumerate(chords):
                start = i * step
                sub_t = np.linspace(0, 1.0, step, endpoint=False)
                decay = np.exp(-sub_t * 2.2)
                sig = np.zeros(step)
                for f in chord:
                    sig += 0.3 * np.sin(2 * np.pi * f * sub_t) + 0.15 * np.sin(2 * np.pi * 2.76 * f * sub_t)
                audio[start:start + step] = sig * decay * tremolo[start:start + step]

        elif "trumpet" in name_lower:
            notes = [(392.00, 0.6), (440.00, 0.4), (523.25, 0.8), (587.33, 0.6), (659.25, 1.2), (523.25, 0.4)]
            audio = np.zeros_like(t)
            pos = 0
            vibrato = 1.0 + 0.012 * np.sin(2 * np.pi * 5.0 * t)
            for f, dur_n in notes:
                n_len = min(int(sr * dur_n), len(t) - pos)
                if n_len <= 0:
                    break
                sub_t = np.linspace(0, dur_n, n_len, endpoint=False)
                env = np.minimum(sub_t / 0.05, 1.0) * np.minimum((dur_n - sub_t) / 0.05, 1.0)
                vib = vibrato[pos:pos + n_len]
                sig = 0.4 * np.sin(2 * np.pi * f * vib * sub_t) + 0.3 * np.sin(2 * np.pi * 2 * f * vib * sub_t) + 0.2 * np.sin(2 * np.pi * 3 * f * vib * sub_t)
                audio[pos:pos + n_len] = sig * env
                pos += n_len

        else:
            sr = 16000
            dur = 4.0
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            f0 = 130.0
            vowel = np.zeros_like(t)
            for h in range(1, 18):
                vowel += (1.0 / (h ** 0.85)) * np.sin(2 * np.pi * (h * f0) * t)
            env = 0.5 * (1 - np.cos(2 * np.pi * 0.5 * t))
            audio = vowel * env * 0.4

        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio / max_val * 0.9
        pcm = (audio * 32767).astype(np.int16)
        bio = io.BytesIO()
        wavfile.write(bio, sr, pcm)
        return bio.getvalue()


    def _load_audio_bytes(input_bytes_or_path):
        if not isinstance(input_bytes_or_path, str):
            return input_bytes_or_path

        path = input_bytes_or_path

        # 1. Try reading local file
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = f.read()
                if data.startswith(b"RIFF") and b"WAVE" in data[:16]:
                    wavfile.read(io.BytesIO(data))
                    return data
            except Exception:
                pass

        # 2. Try fetching clean binary from GitHub raw (resolves Pyodide WASM UTF-8 MEMFS corruption)
        filename = os.path.basename(path)
        raw_url = f"https://raw.githubusercontent.com/kimiyamushtaq/sampling/main/samples/{filename}"
        try:
            try:
                import pyodide.http
                resp = pyodide.http.open_url(raw_url)
                data = resp.read()
                if isinstance(data, str):
                    data = data.encode("latin1")
                wavfile.read(io.BytesIO(data))
                return data
            except ImportError:
                pass

            with urllib.request.urlopen(raw_url, timeout=5) as resp:
                data = resp.read()
            wavfile.read(io.BytesIO(data))
            return data
        except Exception:
            pass

        # 3. Infallible fallback: Generate procedural audio
        return _generate_procedural_sample(filename)


    def _extract_audio_metadata(input_bytes):
        _bio = io.BytesIO(input_bytes)
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
            "size_bytes": len(input_bytes),
            "format_name": "WAV (PCM)",
        }


    def _resample_audio(input_bytes, target_sr):
        _bio_in = io.BytesIO(input_bytes)
        _orig_sr, _data = wavfile.read(_bio_in)

        if _orig_sr == target_sr:
            return input_bytes

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
        _audio_source_input = sample_selector.value
        _track_title = next((k for k, v in sample_tracks.items() if v == _audio_source_input), "Sample Track")
        _has_audio = bool(_audio_source_input)
    else:
        _audio_source_input = file_uploader.contents(0)
        _track_title = file_uploader.name(0) or "Uploaded Audio"
        _has_audio = bool(_audio_source_input)

    if not _has_audio:
        player_view = mo.callout(
            mo.md("Please select or upload a WAV audio file to begin playback."),
            kind="info",
        )
    else:
        try:
            _raw_audio_bytes = _load_audio_bytes(_audio_source_input)
            _target_sr = sample_rate_selector.value
            _meta = _extract_audio_metadata(_raw_audio_bytes)
            _orig_sr = _meta["sample_rate"]
            _orig_size_bytes = _meta["size_bytes"]
            _orig_size_kb = _orig_size_bytes / 1024
            _orig_bitrate = _meta["bitrate_kbps"]

            # Resample audio using pure SciPy polyphase filtering
            _resampled_bytes = _resample_audio(_raw_audio_bytes, _target_sr)
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
