# gym_cleanup — gentle live-recording cleanup pipeline

Cleans up a reverberant single-mic live recording (voice + fingerpicked
acoustic guitar) from a video file: moderate vocal dereverb, subtle
per-segment pitch correction, gentle bus processing, loudness normalize,
and remux back into the original video. **The original file is never
modified and nothing in the chain time-stretches — A/V sync is preserved
by construction.**

## Usage

```bash
/opt/audioenv/bin/python gym_cleanup/pipeline.py INPUT.mov \
    --workdir work --output cleaned.mov
```

Stages: `extract → separate → dereverb → analyze → correct → mix → master → remux`

Each stage caches its output in `--workdir`. Re-running skips finished
stages; `--force-from dereverb` redoes dereverb and everything after it
(cheap iteration — demucs won't re-run). `--stop-after analyze` lets you
review the pitch plot and segment list before anything is corrected.

### Recommended workflow

```bash
# 1st pass: analyze only, look at the plot + segment report
python pipeline.py in.mov --workdir work --stop-after analyze

# inspect work/04_pitch_pre.png and work/04_segments.json, then run full:
python pipeline.py in.mov --workdir work

# after listening, override the plan and re-render from 'correct' on:
python pipeline.py in.mov --workdir work --force-from correct \
    --exclude "41.2-43.0:0" --corrections "12.3-14.1:+18"
```

## Tunable parameters

### Dereverb
| flag | default | meaning |
|---|---|---|
| `--dereverb-backend` | `auto` | `deepfilternet` \| `wpe` \| `none`. `auto` prefers DeepFilterNet if its CLI is installed, else WPE (nara_wpe). |
| `--dereverb-mix` | `0.6` | Vocal wet/dry blend. 0 = untouched, 1 = fully dereverbed. **The main strength knob.** |
| `--guitar-dereverb-mix` | `0.0` | Same for the guitar stem — default leaves the guitar's ambience alone. Try 0.15–0.3 if guitar wash is a problem. |
| `--wpe-taps / --wpe-delay / --wpe-iterations` | 10 / 3 / 3 | WPE backend detail; more taps = stronger dereverb, slower. |

### Pitch analysis / segment selection
| flag | default | meaning |
|---|---|---|
| `--fmin` / `--fmax` | 70 / 600 Hz | pYIN search range (covers high baritone comfortably). |
| `--cents-threshold` | `25` | A note becomes a candidate when its median deviation from the nearest semitone ≥ this. |
| `--min-segment-dur` | `0.25` s | Shorter notes are never touched. |
| `--stability-max-std` | `35` cents | Detrended pitch std above this ⇒ flagged "unstable", skipped. |
| `--slide-max-slope` | `120` cents/s | Median slope above this ⇒ flagged "slide", skipped. |
| `--vibrato-min-extent` | `20` cents p-p | 3–9 Hz modulation above this ⇒ flagged "vibrato", skipped. Threshold is deliberately low: reverb + dereverb attenuate measured vibrato ~2× vs the sung extent. |

### Correction
| flag | default | meaning |
|---|---|---|
| `--correction-strength` | `0.7` | Fraction of the deviation removed (0.6–0.8 keeps it human; 1.0 = fully quantized, not recommended). |
| `--max-shift-cents` | `60` | Safety clamp per segment. |
| `--crossfade-ms` | `40` | Raised-cosine crossfade at segment boundaries. |
| `--corrections` | – | Manual overrides: `"start-end:cents,..."`, e.g. `"12.3-14.1:+18,22.0-23.5:-25"`. Overlapping auto segments are replaced. |
| `--exclude` | – | Drop auto corrections overlapping ranges: `"41.2-43.0:0"` (cents value ignored). |
| `--no-auto-corrections` | off | Apply only `--corrections`. |

### Mix bus
| flag | default | meaning |
|---|---|---|
| `--highpass-hz` | `80` | Gentle 12 dB/oct high-pass on the summed mix. |
| `--comp-threshold-db` / `--comp-ratio` | −20 / 1.5 | Light RMS glue compression (attack/release also exposed). |
| `--reverb-return-db` | `−20` | Level of the shared small-room return (sox, wet-only) that re-glues the stems and masks separation artifacts. −60 or lower = off. |
| `--reverb-reverberance` / `--reverb-room-scale` / `--reverb-predelay-ms` | 35 / 25 / 12 | Character of that room. |
| `--vocal-gain-db` / `--guitar-gain-db` | 0 / 0 | Stem balance trims before summing. |

### Master / output
| flag | default | meaning |
|---|---|---|
| `--target-lufs` | `−14` | Integrated loudness target (pure gain — no limiter artifacts; if the true-peak cap binds, output lands slightly quieter and this is reported in `07_loudness.json`). |
| `--target-tp-db` | `−1` | True-peak ceiling (4× oversampled estimate). |
| `--audio-bitrate` | `256k` | AAC bitrate at remux. |

### Escape hatches
| flag | meaning |
|---|---|
| `--dereverb-only` | Fallback mode if the demucs vocal stem comes back watery/degraded: mild dereverb on the **combined** track, no separation, no pitch correction, then master + remux. |
| `--vocal-stem X --guitar-stem Y` | Skip demucs and use stems you provide. |

## Artifacts kept in `--workdir`

| file | what |
|---|---|
| `01_audio_raw.wav` | extracted audio, untouched |
| `02_stems/vocals.wav`, `02_stems/guitar.wav` | raw demucs stems — **listen to vocals.wav before trusting corrections** |
| `02_stem_quality.json` | crude separation-health heuristics |
| `03_vocal_dereverbed.wav` / `03_guitar.wav` | post-dereverb stems |
| `04_pitch_pre.png`, `04_segments.json`, `04_pitch_track.npz` | pre-correction pitch plot (note grid), segment report |
| `05_correction_plan.json`, `05_vocal_corrected.wav`, `05_pitch_post.png` | what was shifted and by how much; post-correction plot |
| `06_mix.wav`, `07_master.wav` | bus mix and loudness-normalized master |
| `07_loudness.json` | measured/applied loudness numbers |
| `08_output.*` | final remuxed video (video stream bit-copied) |
| `params_used.json` | full parameter set of the last run |

## Notes / known trade-offs

- **Segment classification is conservative by design**: vibrato, slides, and
  unstable notes are flagged in the report and plot but never touched.
  Everything is overridable via `--corrections` / `--exclude`.
- Stem-separation quality on very reverberant material is the weakest link.
  If `02_stems/vocals.wav` sounds watery, use `--dereverb-only`.
- rubberband runs with `--fine` (R3 engine) `-F` (formant preserve) `-t 1`
  (duration preserved exactly; we also sample-match lengths after).
- Loudness normalize is pure gain on purpose — no dynamics processing that
  could pump on a sparse acoustic arrangement.

## Environment

`scripts/setup_env.sh` recreates the whole toolchain (venv at
`/opt/audioenv`): `demucs` (torch CPU), `librosa`, `soundfile`,
`pyloudnorm`, `matplotlib`, `nara_wpe`, optionally `deepfilternet`;
system `ffmpeg`, `rubberband-cli`, `sox`.

Model weights are downloaded on first use and need network access to
`dl.fbaipublicfiles.com` (demucs htdemucs) and `github.com/Rikorose`
(DeepFilterNet). In a restricted environment, allow those hosts in the
network policy or pre-seed `~/.cache/torch/hub/checkpoints/` and
`~/.cache/DeepFilterNet/`.

`scripts/make_test_clip.py` builds a synthetic reverberant test video with
known pitch defects for validating the pipeline without the real footage
(inject its ground-truth stems via `--vocal-stem/--guitar-stem`).
