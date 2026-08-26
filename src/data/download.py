"""Explicit-license dataset downloader with checksum and safe-extraction guards."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tarfile
import urllib.request
import zipfile


VCTK_URL = "https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full"
VCTK_MD5 = "8a6ba2946b36fcbef0212cad601f4bfa"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
MUSAN_MD5 = "0c472d4fc0c5141eca47ad1ffeb2a7df"


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract_archive(archive: Path, destination: Path, *, required_prefixes: tuple[str, ...] = ()) -> None:
    """Extract only safe members, optionally restricting extraction to needed paths."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    def allowed(name: str) -> bool:
        clean = Path(name)
        if clean.is_absolute() or ".." in clean.parts:
            raise ValueError(f"Unsafe archive member: {name}")
        return not required_prefixes or any(name.startswith(prefix) for prefix in required_prefixes)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if allowed(member.filename):
                    target = (destination / member.filename).resolve()
                    if not target.is_relative_to(root):
                        raise ValueError(f"Unsafe archive member: {member.filename}")
                    zipped.extract(member, destination)
    else:
        with tarfile.open(archive, "r:*") as tarred:
            for member in tarred.getmembers():
                if member.issym() or member.islnk():
                    raise ValueError(f"Refusing archive link: {member.name}")
                if allowed(member.name):
                    target = (destination / member.name).resolve()
                    if not target.is_relative_to(root):
                        raise ValueError(f"Unsafe archive member: {member.name}")
                    tarred.extract(member, destination, filter="data")


def download_datasets(dataset_root: Path, *, accept_data_licenses: bool) -> dict[str, Path]:
    if not accept_data_licenses:
        raise PermissionError("Pass --accept-data-licenses after reviewing VCTK and MUSAN licenses.")
    dataset_root.mkdir(parents=True, exist_ok=True)
    archives = {"vctk": (VCTK_URL, VCTK_MD5, dataset_root / "VCTK-Corpus-0.92.zip", ("VCTK-Corpus-0.92/wav48_silence_trimmed/", "VCTK-Corpus-0.92/COPYING")), "musan": (MUSAN_URL, MUSAN_MD5, dataset_root / "musan.tar.gz", ("musan/noise/",))}
    completed: dict[str, Path] = {}
    for name, (url, checksum, archive, prefixes) in archives.items():
        if not archive.exists():
            urllib.request.urlretrieve(url, archive)
        actual = file_md5(archive)
        if actual != checksum:
            raise ValueError(f"Checksum mismatch for {archive.name}: expected {checksum}, got {actual}")
        safe_extract_archive(archive, dataset_root, required_prefixes=prefixes)
        completed[name] = archive
    validate_dataset_layout(dataset_root)
    return completed


def validate_dataset_layout(dataset_root: Path) -> None:
    vctk_root = dataset_root / "VCTK-Corpus-0.92"
    vctk = vctk_root / "wav48_silence_trimmed"
    if not vctk.is_dir():
        vctk = vctk_root / "wav48"
    musan_noise = dataset_root / "musan" / "noise"
    mic1 = list(vctk.glob("p*/*_mic1.flac")) if vctk.is_dir() else []
    if not mic1:
        raise FileNotFoundError(f"Expected VCTK 0.92 mic1 FLAC files below {vctk}")
    if not any(path.suffix.lower() in {".wav", ".flac"} for path in musan_noise.rglob("*")):
        raise FileNotFoundError(f"Expected MUSAN noise audio below {musan_noise}")
