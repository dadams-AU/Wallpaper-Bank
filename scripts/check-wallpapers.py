#!/usr/bin/env python3
"""
Lint the wallpaper bank. Exits non-zero if anything fails.

  * every still and video must cover a 4K screen: 3840x2160 (landscape)
    or 2160x3840 (portrait)
  * file names must be unique without their extension (WallpaperSelect.sh
    resolves a pick with `find -iname "<name>.*"`, so duplicates collide)
  * no leading/trailing spaces in names, image extensions must match content
  * videos should be H.264 / yuv420p so mpvpaper can decode them anywhere

    scripts/check-wallpapers.py            # check wallpapers/
    scripts/check-wallpapers.py --verbose  # also list every passing file

Needs Pillow; videos are probed with ffprobe when it is installed.
"""
import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1] / "wallpapers"
IMAGES = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".gif": "GIF"}
VIDEOS = {".mp4", ".mkv", ".mov", ".webm"}


def covers_4k(w, h):
    return (w >= 3840 and h >= 2160) if w >= h else (w >= 2160 and h >= 3840)


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,codec_name,pix_fmt", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", type=Path, default=ROOT)
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()

    problems, names = [], defaultdict(list)
    have_ffprobe = shutil.which("ffprobe") is not None
    files = sorted(p for p in a.root.rglob("*") if p.is_file())
    for p in files:
        rel = p.relative_to(a.root)
        ext = p.suffix.lower()
        if p.name != p.name.strip():
            problems.append(f"{rel}: leading/trailing whitespace in name")
        names[p.stem.strip().lower()].append(rel)
        if ext in IMAGES:
            with Image.open(p) as im:
                w, h, fmt = *im.size, im.format
            if fmt != IMAGES[ext]:
                problems.append(f"{rel}: extension {ext} but content is {fmt}")
        elif ext in VIDEOS:
            if not have_ffprobe:
                print(f"skip {rel}: ffprobe not installed", file=sys.stderr)
                continue
            try:
                st = probe(p)
            except (subprocess.CalledProcessError, KeyError, IndexError):
                problems.append(f"{rel}: unreadable video")
                continue
            w, h = st["width"], st["height"]
            if st["codec_name"] != "h264" or st["pix_fmt"] != "yuv420p":
                problems.append(f"{rel}: {st['codec_name']}/{st['pix_fmt']} (want h264/yuv420p)")
        else:
            problems.append(f"{rel}: not an image or video")
            continue
        if not covers_4k(w, h):
            problems.append(f"{rel}: {w}x{h} is below 4K")
        elif a.verbose:
            print(f"ok  {w}x{h}  {rel}")
    for stem, paths in names.items():
        if len(paths) > 1:
            problems.append("duplicate name '" + stem + "': " + ", ".join(map(str, paths)))

    for msg in problems:
        print("FAIL " + msg)
    print(f"{len(files)} files checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
