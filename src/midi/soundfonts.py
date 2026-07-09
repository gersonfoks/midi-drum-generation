"""Managed SoundFonts for synthesis.

The synthesizer defaults to this SoundFont when none is supplied. To keep results
reproducible across machines without committing a ~30 MB binary, it is downloaded
on first use from a pinned URL and verified against a known checksum.

Default font: **GeneralUser GS** by S. Christian Collins — freely redistributable,
with realistic acoustic drum kits. See https://schristiancollins.com/generaluser.php
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Downloaded SoundFonts live here; the directory is git-ignored (see .gitignore).
SOUNDFONT_DIR = Path("assets/soundfonts")


@dataclass(frozen=True)
class ManagedSoundFont:
    """A SoundFont fetched on demand from a pinned URL and checksum-verified."""

    filename: str
    # Pinned to a specific commit so every machine fetches identical bytes.
    url: str
    sha256: str
    size: int

    @property
    def path(self) -> Path:
        return SOUNDFONT_DIR / self.filename


# GeneralUser GS, pinned to a specific commit of mrbumpy409/GeneralUser-GS.
GENERALUSER_GS = ManagedSoundFont(
    filename="GeneralUser-GS.sf2",
    url=(
        "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/"
        "684543d5e5efaef08d02be50dcda8d552478fa60/GeneralUser-GS.sf2"
    ),
    sha256="9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe",
    size=32319396,
)

DEFAULT_SOUNDFONT = GENERALUSER_GS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_soundfont(soundfont: ManagedSoundFont = DEFAULT_SOUNDFONT) -> Path:
    """Return the local path to ``soundfont``, downloading it once if needed.

    A present-but-corrupt file (wrong checksum) is re-downloaded. Raises
    :class:`RuntimeError` if the freshly downloaded bytes fail verification.
    """
    dest = soundfont.path
    if dest.is_file() and _sha256(dest) == soundfont.sha256:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SoundFont {soundfont.filename} (~{soundfont.size // 1_000_000} MB)...")
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(soundfont.url, tmp_path)
        actual = _sha256(tmp_path)
        if actual != soundfont.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {soundfont.filename}: "
                f"expected {soundfont.sha256}, got {actual}"
            )
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    return dest
