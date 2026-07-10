#!/usr/bin/env bash
# Recreate the toolchain for gym_cleanup/pipeline.py.
# Tested on Ubuntu 24.04, Python 3.11, CPU-only.
set -euo pipefail

VENV="${1:-/opt/audioenv}"

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    ffmpeg rubberband-cli sox libsox-fmt-all

# NOTE: use a venv -- the Debian-patched system setuptools fails to build
# some sdist deps of demucs (antlr4) with an 'install_layout' AttributeError.
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip setuptools wheel
"$VENV/bin/pip" install demucs soundfile librosa pyloudnorm matplotlib nara_wpe

# Optional: DeepFilterNet backend. Its 0.5.6 release predates torchaudio 2.2's
# removal of torchaudio.backend, and torchaudio >= 2.9 needs torchcodec for I/O.
"$VENV/bin/pip" install deepfilternet torchcodec || true
python3 - "$VENV" <<'EOF'
import glob, sys
hits = glob.glob(sys.argv[1] + "/lib/python*/site-packages/df/io.py")
if not hits:
    sys.exit(0)
p = hits[0]
s = open(p).read()
if "torchaudio >= 2.2 removed" in s:
    sys.exit(0)  # already patched
s = s.replace(
    "from torchaudio.backend.common import AudioMetaData",
    """try:  # torchaudio < 2.2
    from torchaudio.backend.common import AudioMetaData
except ImportError:  # torchaudio >= 2.2 removed the backend module
    class AudioMetaData:
        def __init__(self, sample_rate, num_frames=0, num_channels=0,
                     bits_per_sample=0, encoding=""):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding""")
s = s.replace(
    "info: AudioMetaData = ta.info(file, **ikwargs)",
    """if hasattr(ta, "info"):
        info: AudioMetaData = ta.info(file, **ikwargs)
    else:  # torchaudio >= 2.9: use soundfile for metadata
        import soundfile as _sf
        _i = _sf.info(file)
        info = AudioMetaData(_i.samplerate, _i.frames, _i.channels)""")
open(p, "w").write(s)
print("patched", p)
EOF

echo "Done. Run the pipeline with: $VENV/bin/python gym_cleanup/pipeline.py ..."
echo "First demucs / deepFilter run downloads model weights from"
echo "dl.fbaipublicfiles.com / github.com/Rikorose -- make sure those hosts"
echo "are allowed by the network policy, or pre-seed the caches:"
echo "  ~/.cache/torch/hub/checkpoints/   (demucs)"
echo "  ~/.cache/DeepFilterNet/           (deepFilter)"
