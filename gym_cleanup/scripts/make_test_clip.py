#!/usr/bin/env python3
"""Build a synthetic reverberant voice+guitar test video for pipeline validation.

Synthesizes a vowel-like "sung" vocal line with known pitch defects (ground
truth below), a Karplus-Strong fingerpicked guitar arpeggio, drenches the mix
in a big-room reverb (sox), and muxes it under a test-card video (ffmpeg).

Ground truth vocal line (A440 grid):
    0.5-2.0s   A3  in tune               -> no action
    2.3-4.0s   C4  35 cents FLAT         -> should be corrected
    4.3-6.0s   E4  30c sharp + wide 5.5 Hz vibrato -> should be FLAGGED (vibrato)
    6.3-7.6s   G3->B3 slide              -> should be FLAGGED (slide) or ignored
    7.9-9.6s   D4  28 cents SHARP        -> should be corrected
    9.9-10.4s  A3  40c flat but SHORT    -> ignored (< min duration)
   10.7-12.4s  E3  in tune               -> no action

Outputs (in --outdir): vocal_dry.wav, guitar_dry.wav, vocal_rev.wav,
guitar_rev.wav, mix_rev.wav, test_clip.mp4
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def note_hz(name_midi, cents=0.0):
    return 440.0 * 2 ** ((name_midi - 69 + cents / 100.0) / 12.0)


def vowel(f0_curve, sr=SR):
    """Sawtooth source + 'ah' formant resonators, soft attack/release."""
    from scipy.signal import lfilter
    n = len(f0_curve)
    phase = np.cumsum(2 * np.pi * f0_curve / sr)
    src = 0.0
    for k in range(1, 30):  # band-limited saw
        if k * f0_curve.max() > sr / 2 - 200:
            break
        src = src + np.sin(k * phase) / k
    out = np.zeros(n)
    for fc, bw, g in [(700, 110, 1.0), (1220, 130, 0.5), (2600, 180, 0.25)]:
        r = np.exp(-np.pi * bw / sr)
        th = 2 * np.pi * fc / sr
        b, a = [1 - r], [1, -2 * r * np.cos(th), r * r]
        out = out + g * lfilter(b, a, src)
    env = np.ones(n)
    fade = int(0.06 * sr)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return out * env


def ks_pluck(f0, dur, sr=SR):
    """Karplus-Strong plucked string."""
    n = int(dur * sr)
    period = int(sr / f0)
    buf = np.random.uniform(-1, 1, period)
    out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % period]
        buf[i % period] = 0.996 * 0.5 * (buf[i % period] + buf[(i + 1) % period])
    return out * np.exp(-np.linspace(0, 3.0, n) / dur)


def build_vocal():
    total = 13.0
    x = np.zeros(int(total * SR))
    t = lambda n: int(n * SR)

    def add(start, end, midi, cents=0.0, vib_pp=0.0, vib_hz=5.5, slide_to=None):
        n = t(end) - t(start)
        tt = np.arange(n) / SR
        c = np.full(n, cents)
        if slide_to is not None:
            midi_curve = np.linspace(midi, slide_to, n)
        else:
            midi_curve = np.full(n, float(midi))
        if vib_pp > 0:
            c = c + (vib_pp / 2.0) * np.sin(2 * np.pi * vib_hz * tt)
        f0 = 440.0 * 2 ** ((midi_curve - 69 + c / 100.0) / 12.0)
        x[t(start):t(end)] += vowel(f0)

    add(0.5, 2.0, 57)                       # A3 in tune
    add(2.3, 4.0, 60, cents=-35)            # C4 flat -> correct
    add(4.3, 6.0, 64, cents=+30, vib_pp=60) # E4 sharp + vibrato -> flag
    add(6.3, 7.6, 55, slide_to=59)          # G3->B3 slide -> flag/ignore
    add(7.9, 9.6, 62, cents=+28)            # D4 sharp -> correct
    add(9.9, 10.4, 57, cents=-40)           # short -> ignore
    add(10.7, 12.4, 52)                     # E3 in tune
    return 0.25 * x / (np.abs(x).max() + 1e-9)


def build_guitar():
    total = 13.0
    x = np.zeros(int(total * SR))
    pattern = [45, 52, 57, 60, 57, 52]      # A-minor-ish arpeggio (A2 E3 A3 C4...)
    step = 0.32
    pos = 0.2
    i = 0
    while pos + 1.2 < total:
        f = note_hz(pattern[i % len(pattern)])
        pl = ks_pluck(f, 1.2)
        a = int(pos * SR)
        x[a:a + len(pl)] += pl
        pos += step
        i += 1
    return 0.22 * x / (np.abs(x).max() + 1e-9)


def sox_gym(in_path, out_path):
    subprocess.run(["sox", str(in_path), str(out_path),
                    "reverb", "80", "40", "100", "100", "30", "0",
                    "gain", "-3"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="testclip")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    v, g = build_vocal(), build_guitar()
    sf.write(out / "vocal_dry.wav", v, SR)
    sf.write(out / "guitar_dry.wav", g, SR)
    sox_gym(out / "vocal_dry.wav", out / "vocal_rev.wav")
    sox_gym(out / "guitar_dry.wav", out / "guitar_rev.wav")
    vr, _ = sf.read(out / "vocal_rev.wav")
    gr, _ = sf.read(out / "guitar_rev.wav")
    n = max(len(vr), len(gr))
    vr = np.pad(vr, (0, n - len(vr)))
    gr = np.pad(gr, (0, n - len(gr)))
    mix = vr + gr
    mix = 0.7 * mix / (np.abs(mix).max() + 1e-9)
    sf.write(out / "mix_rev.wav", mix, SR)

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={n / SR:.3f}",
        "-i", out / "mix_rev.wav",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        out / "test_clip.mp4"], check=True)
    print(f"wrote {out / 'test_clip.mp4'}")


if __name__ == "__main__":
    main()
