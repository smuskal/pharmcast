"""Give a checkpoint a release name.

A trained checkpoint records what it was trained on but not what it is *called*.
That gap matters once a model is published: a file can be renamed, copied or
re-downloaded, and a name that lives only in the filename is not evidence of
anything. `stamp` writes the release label into the blob itself, so
`PharmCast.load(...).version` reports the same string no matter what the file on
disk happens to be called.

    python -m pharmcast.stamp in.pt --release PharmCastSP.21August2026 --out out.pt

Stamping never touches the weights. It is refused on a checkpoint that already
carries a release name unless `--force` is given, because silently renaming a
published model is how two different sets of weights end up sharing one name.
"""
from __future__ import annotations

import argparse
import hashlib

import torch


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stamp(src, dst, release, force=False):
    blob = torch.load(src, map_location="cpu", weights_only=False)
    existing = blob.get("release")
    if existing and existing != release and not force:
        raise SystemExit(
            "refusing to restamp: %s is already released as %r (use --force)"
            % (src, existing))
    blob["release"] = release
    blob["released_from_sha256"] = sha256(src)
    torch.save(blob, dst)
    return blob


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("--release", required=True,
                   help="e.g. PharmCastSP.21August2026")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    stamp(a.src, a.out, a.release, a.force)
    print("%s -> %s\nrelease  %s\nsha256   %s"
          % (a.src, a.out, a.release, sha256(a.out)))


if __name__ == "__main__":
    main()
