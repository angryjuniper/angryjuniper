#!/usr/bin/env python3
"""
Gym-recording cleanup pipeline.

Gentle cleanup of a reverberant live recording (one performer: voice +
fingerpicked acoustic guitar, single iPhone mic):

  1. extract   - pull audio out of the video to WAV (original file untouched)
  2. separate  - demucs htdemucs two-stem split: vocals / accompaniment
  3. dereverb  - reduce reverb on the VOCAL stem only (strength is a mix knob);
                 guitar can optionally get a much lighter pass
  4. analyze   - pYIN pitch tracking on the vocal, note-grid plot (PNG),
                 candidate off-pitch segments with cents deviations;
                 vibrato / slides / unstable notes are flagged, not corrected
  5. correct   - per-segment corrective pitch shift with rubberband
                 (partial correction, short crossfades, no time-stretch)
  6. mix       - vocal + guitar, gentle 80 Hz high-pass, light glue
                 compression, small shared room-reverb return
  7. master    - loudness normalize (pure gain) to target LUFS / dBTP
  8. remux     - copy the original video stream, mux in AAC 256k audio

Every stage writes its outputs into --workdir and is skipped on re-run if its
outputs already exist; use --force-from STAGE to redo a stage and everything
after it. All tuning knobs are CLI flags (see README.md).

No stage time-stretches: durations are preserved end to end, so A/V sync is
maintained by construction.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import soundfile as sf

STAGES = ["extract", "separate", "dereverb", "analyze", "correct", "mix", "master", "remux"]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------- util

def log(msg):
    print(f"[pipeline] {msg}", flush=True)


def run(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def read_wav(path):
    data, sr = sf.read(str(path), always_2d=True)
    return data.astype(np.float64), sr


def write_wav(path, data, sr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="FLOAT")


def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


def midi_to_name(m):
    m = int(round(m))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.asarray(f) / 440.0)


def db_to_lin(db):
    return 10.0 ** (db / 20.0)


# --------------------------------------------------------------------------- config

@dataclass
class Params:
    # dereverb
    dereverb_backend: str = "auto"       # auto | deepfilternet | wpe | none
    dereverb_mix: float = 0.6            # 0=dry stem .. 1=fully dereverbed (vocal)
    guitar_dereverb_mix: float = 0.0     # same, for the guitar stem (default: untouched)
    wpe_taps: int = 10
    wpe_delay: int = 3
    wpe_iterations: int = 3

    # pitch analysis
    fmin: float = 70.0                   # Hz, pYIN range (high baritone default)
    fmax: float = 600.0
    pyin_frame: int = 1536               # shorter frame tracks vibrato better
    pyin_resolution: float = 0.05        # semitones per pYIN bin (5 cents)
    cents_threshold: float = 25.0        # candidate if |median dev| >= this
    min_segment_dur: float = 0.25        # s, ignore shorter notes
    stability_max_std: float = 35.0      # cents, detrended std above this -> "unstable", flag only
    slide_max_slope: float = 120.0       # cents/sec median slope above this -> "slide", flag only
    vibrato_min_extent: float = 20.0     # cents p-p of 3-9 Hz modulation -> "vibrato", flag only.
                                         # NB: heavy reverb + dereverb attenuate measured vibrato
                                         # ~2x vs the true sung extent; threshold is set low on purpose.

    # correction
    correction_strength: float = 0.7     # fraction of deviation to remove (0..1)
    max_shift_cents: float = 60.0        # safety clamp on any single shift
    crossfade_ms: float = 40.0
    segment_pad_ms: float = 30.0         # analysis-to-audio padding around each segment

    # mix
    highpass_hz: float = 80.0
    comp_threshold_db: float = -20.0     # glue compressor (RMS, soft)
    comp_ratio: float = 1.5
    comp_attack_ms: float = 30.0
    comp_release_ms: float = 250.0
    reverb_return_db: float = -20.0      # wet-only room return level relative to full scale mix
    reverb_reverberance: float = 35.0    # sox reverb params (small shared room)
    reverb_room_scale: float = 25.0
    reverb_predelay_ms: float = 12.0
    vocal_gain_db: float = 0.0           # stem balance trims
    guitar_gain_db: float = 0.0

    # master
    target_lufs: float = -14.0
    target_tp_db: float = -1.0

    # remux
    audio_bitrate: str = "256k"

    # runtime
    sample_rate: int = 44100
    demucs_model: str = "htdemucs"


# --------------------------------------------------------------------------- stages

def stage_extract(args, P, W):
    out = W / "01_audio_raw.wav"
    if out.exists() and not args.redo("extract"):
        return out
    run(["ffmpeg", "-y", "-i", args.input, "-vn", "-ac", "2",
         "-ar", str(P.sample_rate), "-c:a", "pcm_f32le", out])
    return out


def stage_separate(args, P, W, raw_wav):
    """demucs two-stem split. Honors --vocal-stem/--guitar-stem overrides."""
    vdir = W / "02_stems"
    voc, acc = vdir / "vocals.wav", vdir / "guitar.wav"
    if args.vocal_stem and args.guitar_stem:
        vdir.mkdir(parents=True, exist_ok=True)
        for src, dst in [(args.vocal_stem, voc), (args.guitar_stem, acc)]:
            x, sr = read_wav(src)
            if sr != P.sample_rate:
                x = resample_audio(x, sr, P.sample_rate)
            write_wav(dst, x, P.sample_rate)
        log("using externally provided stems (demucs skipped)")
        return voc, acc
    if voc.exists() and acc.exists() and not args.redo("separate"):
        return voc, acc
    tmp = W / "02_demucs_out"
    try:
        run([sys.executable, "-m", "demucs.separate", "-n", P.demucs_model,
             "--two-stems", "vocals", "-o", tmp, "-d", "cpu", "--float32", raw_wav])
    except subprocess.CalledProcessError:
        log("demucs failed. If the log above shows a download error, the "
            "model weights host (dl.fbaipublicfiles.com) is unreachable -- "
            "allow it in the environment's network policy, or drop a "
            "pre-downloaded checkpoint into the torch hub cache "
            "(~/.cache/torch/hub/checkpoints/), or run with --dereverb-only.")
        raise
    got = tmp / P.demucs_model / Path(raw_wav).stem
    vdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(got / "vocals.wav", voc)
    shutil.copy(got / "no_vocals.wav", acc)
    report_stem_quality(voc, acc, W)
    return voc, acc


def report_stem_quality(voc_path, acc_path, W):
    """Cheap heuristics to help judge whether the vocal stem survived the reverb.

    These cannot hear 'watery' artifacts -- always listen to 02_stems/vocals.wav
    before trusting pitch correction. But wildly low energy share or very low
    voiced-frame coverage is an early warning.
    """
    v, sr = read_wav(voc_path)
    a, _ = read_wav(acc_path)
    v, a = to_mono(v), to_mono(a)
    ev, ea = float(np.mean(v ** 2)), float(np.mean(a ** 2))
    share = ev / (ev + ea + 1e-12)
    q = {"vocal_energy_share": round(share, 4),
         "vocal_rms_db": round(10 * math.log10(ev + 1e-12), 1),
         "warning": share < 0.02}
    (W / "02_stem_quality.json").write_text(json.dumps(q, indent=2))
    log(f"stem quality heuristics: {q}")
    if q["warning"]:
        log("WARNING: vocal stem energy is very low -- separation may have "
            "failed on this material. LISTEN to 02_stems/vocals.wav before "
            "proceeding; consider --dereverb-only fallback (no pitch correction).")


def resample_audio(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    import librosa
    y = librosa.resample(x.T, orig_sr=sr_in, target_sr=sr_out)
    return y.T


def dereverb_deepfilternet(in_path, out_path, W):
    """DeepFilterNet CLI. Runs at 48 kHz internally; we resample back."""
    tmpdir = W / "03_dfn_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    x, sr = read_wav(in_path)
    p48 = tmpdir / "in48.wav"
    sf.write(str(p48), resample_audio(x, sr, 48000), 48000, subtype="FLOAT")
    run(["deepFilter", str(p48), "-o", str(tmpdir)])
    outs = [p for p in tmpdir.glob("*.wav") if p.name != "in48.wav"]
    if not outs:
        raise RuntimeError("deepFilter produced no output")
    y, ysr = read_wav(sorted(outs, key=lambda p: p.stat().st_mtime)[-1])
    y = resample_audio(y, ysr, sr)
    y = match_length(y, len(x))
    write_wav(out_path, y, sr)


def dereverb_wpe(in_path, out_path, P):
    """Single-channel WPE (nara_wpe): solid moderate dereverb, no ML voicing bias."""
    from nara_wpe.wpe import wpe
    import librosa
    x, sr = read_wav(in_path)
    chans = [librosa.stft(x[:, c].astype(np.float32), n_fft=1024, hop_length=256)
             for c in range(x.shape[1])]
    Y = np.stack(chans, axis=1)  # (F, D, T)
    Z = wpe(Y, taps=P.wpe_taps, delay=P.wpe_delay, iterations=P.wpe_iterations,
            statistics_mode="full")
    y = np.stack([librosa.istft(Z[:, c, :], hop_length=256, length=len(x))
                  for c in range(x.shape[1])], axis=1)
    y = match_length(y, len(x))
    write_wav(out_path, y, sr)


def match_length(y, n):
    if len(y) < n:
        y = np.pad(y, [(0, n - len(y))] + [(0, 0)] * (y.ndim - 1))
    return y[:n]


def pick_dereverb_backend(P):
    if P.dereverb_backend != "auto":
        return P.dereverb_backend
    have_cli = shutil.which("deepFilter") is not None
    have_model = any(Path(d).expanduser().exists() for d in
                     ["~/.cache/DeepFilterNet", "~/Library/Caches/DeepFilterNet"])
    return "deepfilternet" if (have_cli and have_model) else "wpe"


def stage_dereverb(args, P, W, voc, acc):
    voc_out = W / "03_vocal_dereverbed.wav"
    acc_out = W / "03_guitar.wav"
    if voc_out.exists() and acc_out.exists() and not args.redo("dereverb"):
        return voc_out, acc_out
    backend = pick_dereverb_backend(P)
    log(f"dereverb backend: {backend}, vocal mix={P.dereverb_mix}, "
        f"guitar mix={P.guitar_dereverb_mix}")

    def process(in_path, out_path, mix):
        dry, sr = read_wav(in_path)
        if mix <= 0 or backend == "none":
            write_wav(out_path, dry, sr)
            return
        wet_path = out_path.with_suffix(".wet.wav")
        if backend == "deepfilternet":
            try:
                dereverb_deepfilternet(in_path, wet_path, W)
            except Exception as e:  # e.g. model weights unreachable
                log(f"deepfilternet failed ({e}); falling back to WPE")
                dereverb_wpe(in_path, wet_path, P)
        else:
            dereverb_wpe(in_path, wet_path, P)
        wet, _ = read_wav(wet_path)
        wet = match_length(wet, len(dry))
        write_wav(out_path, mix * wet + (1.0 - mix) * dry, sr)

    process(voc, voc_out, P.dereverb_mix)
    process(acc, acc_out, P.guitar_dereverb_mix)
    return voc_out, acc_out


# ----------------------------------------------------------------- pitch analysis

def track_pitch(path, P):
    import librosa
    x, sr = read_wav(path)
    mono = to_mono(x)
    hop = 256
    f0, voiced, prob = librosa.pyin(
        mono.astype(np.float32), fmin=P.fmin, fmax=P.fmax, sr=sr,
        frame_length=P.pyin_frame, hop_length=hop,
        resolution=P.pyin_resolution, fill_na=np.nan)
    t = librosa.times_like(f0, sr=sr, hop_length=hop)
    return t, f0, voiced


def analyze_segments(t, f0, voiced, P):
    """Group voiced frames into note segments and classify each."""
    midi = np.full_like(f0, np.nan)
    ok = np.isfinite(f0) & voiced
    midi[ok] = hz_to_midi(f0[ok])

    # median-smooth to remove pYIN jitter before segmentation
    from scipy.signal import medfilt
    sm = midi.copy()
    idx = np.where(np.isfinite(sm))[0]
    if len(idx) == 0:
        return []
    smf = sm[idx]
    if len(smf) >= 9:
        smf = medfilt(smf, 9)
    sm[idx] = smf

    dt = float(t[1] - t[0])
    segs, start = [], None
    for i in range(len(sm) + 1):
        val = sm[i] if i < len(sm) else np.nan
        new_note = (start is not None and i < len(sm) and np.isfinite(val)
                    and abs(val - np.nanmedian(sm[start:i])) > 0.8)
        if np.isfinite(val) and start is None:
            start = i
        elif start is not None and (not np.isfinite(val) or new_note):
            if (i - start) * dt >= P.min_segment_dur:
                segs.append((start, i))
            start = i if (i < len(sm) and np.isfinite(val)) else None

    out = []
    for a, b in segs:
        seg_c = (midi[a:b] - np.round(np.nanmedian(midi[a:b]))) * 100.0
        seg_c = seg_c[np.isfinite(seg_c)]
        if len(seg_c) < 8:
            continue
        nearest = int(round(np.nanmedian(midi[a:b])))
        dev = float(np.median(seg_c))
        tt = np.arange(len(seg_c)) * dt
        slope = float(np.polyfit(tt, seg_c, 1)[0])
        detr = seg_c - np.polyval(np.polyfit(tt, seg_c, 1), tt)
        std = float(np.std(detr))
        vib_extent, vib_rate = vibrato_extent(detr, dt)

        action, reason = "ok", ""
        if abs(dev) >= P.cents_threshold:
            if vib_extent >= P.vibrato_min_extent:
                action, reason = "flag", f"vibrato ({vib_extent:.0f}c p-p @ {vib_rate:.1f} Hz)"
            elif abs(slope) >= P.slide_max_slope:
                action, reason = "flag", f"slide (slope {slope:+.0f} c/s)"
            elif std >= P.stability_max_std:
                action, reason = "flag", f"unstable (detrended std {std:.0f}c)"
            else:
                action = "correct"
        out.append({
            "start": round(float(t[a]), 3), "end": round(float(t[b - 1]), 3),
            "note": midi_to_name(nearest), "midi": nearest,
            "cents_dev": round(dev, 1), "slope_cents_per_s": round(slope, 1),
            "detrended_std_cents": round(std, 1),
            "vibrato_extent_cents": round(vib_extent, 1),
            "action": action, "reason": reason,
        })
    return out


def vibrato_extent(detrended_cents, dt):
    """Peak-to-peak extent of 3-9 Hz modulation.

    Rate from the rFFT peak in the vibrato band, then amplitude from a
    quadrature sine fit at that rate (matched filter -- much less sensitive
    to window scalloping and dropped frames than the raw FFT magnitude).
    """
    n = len(detrended_cents)
    if n < 16:
        return 0.0, 0.0
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(detrended_cents * w))
    freqs = np.fft.rfftfreq(n, dt)
    band = (freqs >= 3.0) & (freqs <= 9.0)
    if not band.any():
        return 0.0, 0.0
    rate = float(freqs[np.argmax(spec * band)])
    tt = np.arange(n) * dt
    amp = 2.0 * math.hypot(
        float(np.mean(detrended_cents * np.sin(2 * np.pi * rate * tt))),
        float(np.mean(detrended_cents * np.cos(2 * np.pi * rate * tt))))
    return float(2.0 * amp), rate


def plot_pitch(t, f0, segments, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    midi = np.full_like(f0, np.nan)
    ok = np.isfinite(f0)
    midi[ok] = hz_to_midi(f0[ok])
    if not np.isfinite(midi).any():
        log(f"plot: no voiced frames, skipping {out_png}")
        return
    lo = int(np.floor(np.nanmin(midi))) - 1
    hi = int(np.ceil(np.nanmax(midi))) + 1
    fig, ax = plt.subplots(figsize=(min(60.0, max(12.0, t[-1] * 0.9)), 7))
    for m in range(lo, hi + 1):
        ax.axhline(m, color="0.85", lw=0.6, zorder=0)
    ax.plot(t, midi, ".", ms=2.0, color="#2b6cb0", zorder=2, label="f0 (pYIN)")
    colors = {"correct": "#c05621", "flag": "#b83280"}
    seen = set()
    for s in segments:
        if s["action"] == "ok":
            continue
        lab = {"correct": "correction candidate", "flag": "flagged (skipped)"}[s["action"]]
        ax.axvspan(s["start"], s["end"], color=colors[s["action"]], alpha=0.18,
                   label=lab if lab not in seen else None, zorder=1)
        seen.add(lab)
        ax.annotate(f"{s['note']} {s['cents_dev']:+.0f}c",
                    ((s["start"] + s["end"]) / 2, s["midi"] + 0.55),
                    ha="center", fontsize=8, color=colors[s["action"]])
    ax.set_yticks(range(lo, hi + 1))
    ax.set_yticklabels([midi_to_name(m) for m in range(lo, hi + 1)])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("note")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, t[-1])
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    log(f"wrote {out_png}")


def stage_analyze(args, P, W, voc_path):
    seg_json = W / "04_segments.json"
    png = W / "04_pitch_pre.png"
    if seg_json.exists() and png.exists() and not args.redo("analyze"):
        return json.loads(seg_json.read_text())
    t, f0, voiced = track_pitch(voc_path, P)
    segments = analyze_segments(t, f0, voiced, P)
    seg_json.write_text(json.dumps(segments, indent=2))
    np.savez(W / "04_pitch_track.npz", t=t, f0=f0, voiced=voiced)
    plot_pitch(t, f0, segments, png, "vocal pitch - BEFORE correction")
    log("--- segment report ---")
    for s in segments:
        if s["action"] != "ok":
            log(f"  {s['start']:7.2f}-{s['end']:7.2f}s  {s['note']:<4} "
                f"{s['cents_dev']:+5.1f}c  -> {s['action']}  {s['reason']}")
    return segments


# --------------------------------------------------------------------- correction

def parse_manual_corrections(spec):
    """--corrections '12.3-14.1:+18,22.0-23.5:-25' -> [(start, end, cents), ...]"""
    out = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        rng, cents = part.rsplit(":", 1)
        a, b = rng.split("-")
        out.append((float(a), float(b), float(cents)))
    return out


def build_correction_plan(segments, args, P):
    """Auto plan from analysis, minus --exclude ranges, plus --corrections overrides."""
    plan = []
    if not args.no_auto_corrections:
        for s in segments:
            if s["action"] != "correct":
                continue
            shift = -s["cents_dev"] * P.correction_strength
            shift = float(np.clip(shift, -P.max_shift_cents, P.max_shift_cents))
            plan.append({"start": s["start"], "end": s["end"], "cents": round(shift, 1),
                         "note": s["note"], "source": "auto",
                         "orig_dev": s["cents_dev"]})
    for a, b, _ in (parse_manual_corrections(args.exclude) if args.exclude else []):
        plan = [p for p in plan if not (p["start"] < b and p["end"] > a)]
    if args.corrections:
        for a, b, c in parse_manual_corrections(args.corrections):
            plan = [p for p in plan if not (p["start"] < b and p["end"] > a)]
            plan.append({"start": a, "end": b, "cents": c, "note": "?",
                         "source": "manual", "orig_dev": None})
    return sorted(plan, key=lambda p: p["start"])


def rubberband_shift(seg, sr, cents, W, tag):
    """Pitch-shift a segment with rubberband CLI, preserving duration exactly."""
    tmp_in = W / f"05_tmp_{tag}_in.wav"
    tmp_out = W / f"05_tmp_{tag}_out.wav"
    write_wav(tmp_in, seg, sr)
    cmd = ["rubberband", "--fine", "-F", "-t", "1", "-p", f"{cents / 100.0:.5f}",
           tmp_in, tmp_out]
    try:
        run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # older rubberband without --fine (R3 engine)
        run(["rubberband", "-F", "-t", "1", "-p", f"{cents / 100.0:.5f}",
             tmp_in, tmp_out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    y, _ = read_wav(tmp_out)
    y = match_length(y, len(seg))
    tmp_in.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)
    return y


def stage_correct(args, P, W, voc_path, segments):
    out = W / "05_vocal_corrected.wav"
    plan_path = W / "05_correction_plan.json"
    plan = build_correction_plan(segments, args, P)
    plan_path.write_text(json.dumps(plan, indent=2))
    if out.exists() and not args.redo("correct"):
        return out, plan
    x, sr = read_wav(voc_path)
    if not plan:
        log("no corrections to apply")
        write_wav(out, x, sr)
        return out, plan
    y = x.copy()
    fade = int(P.crossfade_ms / 1000.0 * sr)
    pad = int(P.segment_pad_ms / 1000.0 * sr)
    for k, p in enumerate(plan):
        a = max(0, int(p["start"] * sr) - pad)
        b = min(len(x), int(p["end"] * sr) + pad)
        if b - a < 4 * fade:
            log(f"  skip too-short segment {p}")
            continue
        log(f"  shifting {p['start']:.2f}-{p['end']:.2f}s by {p['cents']:+.1f} cents "
            f"({p['source']}, note {p['note']})")
        shifted = rubberband_shift(x[a:b], sr, p["cents"], W, f"{k}")
        piece = shifted.copy()
        ramp = np.linspace(0.0, 1.0, fade)[:, None]
        piece[:fade] = x[a:a + fade] * (1 - ramp) + shifted[:fade] * ramp
        piece[-fade:] = x[b - fade:b] * ramp + shifted[-fade:] * (1 - ramp)
        y[a:b] = piece
    write_wav(out, y, sr)
    # post-correction plot
    t, f0, voiced = track_pitch(out, P)
    post_segments = analyze_segments(t, f0, voiced, P)
    plot_pitch(t, f0, post_segments, W / "05_pitch_post.png",
               "vocal pitch - AFTER correction")
    return out, plan


# --------------------------------------------------------------------------- mix

def glue_compress(x, sr, P):
    """Very light feed-forward RMS compressor (soft glue, ~1-2 dB GR max)."""
    mono = to_mono(x)
    att = math.exp(-1.0 / (P.comp_attack_ms / 1000.0 * sr))
    rel = math.exp(-1.0 / (P.comp_release_ms / 1000.0 * sr))
    env = np.zeros(len(mono))
    e = 1e-9
    sq = mono ** 2
    for i in range(len(mono)):  # smoothed RMS envelope
        coef = att if sq[i] > e else rel
        e = coef * e + (1 - coef) * sq[i]
        env[i] = e
    lvl_db = 10.0 * np.log10(env + 1e-12)
    over = np.maximum(0.0, lvl_db - P.comp_threshold_db)
    gr_db = over * (1.0 - 1.0 / P.comp_ratio)
    gain = db_to_lin(-gr_db)
    log(f"  glue comp: max GR {gr_db.max():.2f} dB")
    return x * gain[:, None], float(gr_db.max())


def sox_room_wet(in_path, out_path, P):
    run(["sox", in_path, out_path, "reverb", "-w",
         f"{P.reverb_reverberance:.0f}", "50", f"{P.reverb_room_scale:.0f}",
         "100", f"{P.reverb_predelay_ms:.0f}", "0"])


def stage_mix(args, P, W, voc_path, gtr_path):
    out = W / "06_mix.wav"
    if out.exists() and not args.redo("mix"):
        return out
    from scipy.signal import butter, sosfilt
    v, sr = read_wav(voc_path)
    g, _ = read_wav(gtr_path)
    n = max(len(v), len(g))
    v, g = match_length(v, n), match_length(g, n)
    mix = v * db_to_lin(P.vocal_gain_db) + g * db_to_lin(P.guitar_gain_db)
    if P.highpass_hz > 0:
        sos = butter(2, P.highpass_hz, "highpass", fs=sr, output="sos")
        mix = sosfilt(sos, mix, axis=0)
    # shared small-room return to re-glue the stems (wet-only, low in the mix)
    dry_path = W / "06_premix_dry.wav"
    write_wav(dry_path, mix, sr)
    if P.reverb_return_db > -60:
        wet_path = W / "06_room_wet.wav"
        sox_room_wet(dry_path, wet_path, P)
        wet, _ = read_wav(wet_path)
        wet = match_length(wet, n)
        mix = mix + wet * db_to_lin(P.reverb_return_db)
    mix, _ = glue_compress(mix, sr, P)
    write_wav(out, mix, sr)
    return out


# ------------------------------------------------------------------------- master

def true_peak_db(x, sr):
    from scipy.signal import resample_poly
    up = resample_poly(x, 4, 1, axis=0)
    return 20.0 * math.log10(np.abs(up).max() + 1e-12)


def stage_master(args, P, W, mix_path):
    out = W / "07_master.wav"
    if out.exists() and not args.redo("master"):
        return out
    import pyloudnorm as pyln
    x, sr = read_wav(mix_path)
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(x)
    gain_db = P.target_lufs - lufs
    tp = true_peak_db(x, sr)
    headroom = P.target_tp_db - tp
    applied = min(gain_db, headroom)  # pure gain; never push TP over target
    y = x * db_to_lin(applied)
    info = {"input_lufs": round(lufs, 2), "input_tp_db": round(tp, 2),
            "gain_applied_db": round(applied, 2),
            "output_lufs": round(lufs + applied, 2),
            "output_tp_db": round(tp + applied, 2),
            "tp_limited": bool(applied < gain_db)}
    (W / "07_loudness.json").write_text(json.dumps(info, indent=2))
    log(f"loudness: {info}")
    if info["tp_limited"]:
        log("NOTE: gain capped by true-peak target; integrated loudness is "
            "below target rather than risk limiter artifacts.")
    write_wav(out, y, sr)
    return out


def stage_remux(args, P, W, master_path):
    ext = Path(args.input).suffix.lower()
    container = ext if ext in (".mp4", ".mov", ".m4v") else ".mp4"
    out = Path(args.output) if args.output else W / f"08_output{container}"
    if out.exists() and not args.redo("remux"):
        return out
    run(["ffmpeg", "-y", "-i", args.input, "-i", master_path,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", P.audio_bitrate,
         "-movflags", "+faststart", out])
    return out


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input video (never modified)")
    ap.add_argument("--workdir", default="work", help="stage outputs / cache dir")
    ap.add_argument("--output", default=None, help="final video path")
    ap.add_argument("--force-from", choices=STAGES, default=None,
                    help="redo this stage and all later ones")
    ap.add_argument("--stop-after", choices=STAGES, default=None,
                    help="stop after this stage (e.g. 'analyze' to review plots first)")
    ap.add_argument("--dereverb-only", action="store_true",
                    help="fallback mode: skip separation & pitch correction, "
                         "mild dereverb on the combined track only")
    ap.add_argument("--vocal-stem", default=None, help="skip demucs, use this vocal stem")
    ap.add_argument("--guitar-stem", default=None, help="skip demucs, use this guitar stem")
    ap.add_argument("--corrections", default=None,
                    help="manual override 'start-end:cents,...' e.g. '12.3-14.1:+18'")
    ap.add_argument("--exclude", default=None,
                    help="drop auto corrections overlapping 'start-end:0,...' ranges")
    ap.add_argument("--no-auto-corrections", action="store_true",
                    help="apply only --corrections, ignore the auto plan")
    for f in Params.__dataclass_fields__.values():
        flag = "--" + f.name.replace("_", "-")
        ftype = f.type if isinstance(f.type, type) else {"str": str, "int": int,
                                                         "float": float}[f.type]
        ap.add_argument(flag, type=ftype, default=f.default)
    args = ap.parse_args()
    P = Params(**{f: getattr(args, f) for f in Params.__dataclass_fields__})

    W = Path(args.workdir)
    W.mkdir(parents=True, exist_ok=True)
    (W / "params_used.json").write_text(json.dumps(asdict(P), indent=2))

    force_idx = STAGES.index(args.force_from) if args.force_from else None
    args.redo = lambda stage: force_idx is not None and STAGES.index(stage) >= force_idx

    def done(stage, result):
        log(f"stage '{stage}' done -> {result}")
        if args.stop_after == stage:
            log("stopping early as requested")
            sys.exit(0)

    raw = stage_extract(args, P, W)
    done("extract", raw)

    if args.dereverb_only:
        # fallback path: no separation, no pitch correction
        drv = W / "03_combined_dereverbed.wav"
        if not drv.exists() or args.redo("dereverb"):
            backend = pick_dereverb_backend(P)
            wet = W / "03_combined.wet.wav"
            (dereverb_deepfilternet if backend == "deepfilternet"
             else lambda i, o, _: dereverb_wpe(i, o, P))(raw, wet, W)
            dry, sr = read_wav(raw)
            w, _ = read_wav(wet)
            write_wav(drv, P.dereverb_mix * match_length(w, len(dry))
                      + (1 - P.dereverb_mix) * dry, sr)
        done("dereverb", drv)
        master = stage_master(args, P, W, drv)
        done("master", master)
        final = stage_remux(args, P, W, master)
        done("remux", final)
        return

    voc, gtr = stage_separate(args, P, W, raw)
    done("separate", (voc, gtr))
    voc_d, gtr_d = stage_dereverb(args, P, W, voc, gtr)
    done("dereverb", (voc_d, gtr_d))
    segments = stage_analyze(args, P, W, voc_d)
    done("analyze", W / "04_segments.json")
    voc_c, plan = stage_correct(args, P, W, voc_d, segments)
    done("correct", voc_c)
    mixed = stage_mix(args, P, W, voc_c, gtr_d)
    done("mix", mixed)
    master = stage_master(args, P, W, mixed)
    done("master", master)
    final = stage_remux(args, P, W, master)
    done("remux", final)
    log(f"FINAL: {final}")


if __name__ == "__main__":
    main()
