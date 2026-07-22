#!/usr/bin/env python3
# Usage: from fr3_real/, run: python data-pipeline/stage_fr3_merged_recordings.py --help
"""Create a symlink-only raw recording set from one or more source roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile


MARKER = ".fr3_merged_recordings.json"


def episode_is_usable(path: Path, include_failed: bool) -> bool:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    return include_failed or bool(metadata.get("success", False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument(
        "--replace",
        action="store_true",
        help=f"Replace an existing staging directory only if it contains {MARKER}.",
    )
    args = parser.parse_args()

    sources = [source.expanduser().resolve() for source in args.source]
    output = args.output.expanduser().resolve()
    if output in sources:
        raise SystemExit("--output must not be one of the source directories")
    for source in sources:
        if not source.is_dir():
            raise SystemExit(f"Missing source directory: {source}")

    episodes: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for source in sources:
        count = 0
        for episode in sorted(path for path in source.iterdir() if path.is_dir()):
            if not episode_is_usable(episode, args.include_failed):
                continue
            if episode.name in episodes:
                raise SystemExit(
                    f"Duplicate episode name {episode.name!r}: {episodes[episode.name]} and {episode}"
                )
            episodes[episode.name] = episode
            count += 1
        counts[str(source)] = count
    if not episodes:
        raise SystemExit("No usable episodes found")

    if output.exists():
        if not args.replace:
            raise SystemExit(f"{output} already exists; pass --replace to rebuild the staging directory")
        if not (output / MARKER).is_file():
            raise SystemExit(f"Refusing to replace unmarked directory: {output}")
        shutil.rmtree(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, episode in sorted(episodes.items()):
            (temporary / name).symlink_to(episode, target_is_directory=True)
        manifest = {
            "version": 1,
            "sources": counts,
            "total_episodes": len(episodes),
            "include_failed": args.include_failed,
        }
        (temporary / MARKER).write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"[stage] episodes={len(episodes)} output={output}")
    for source, count in counts.items():
        print(f"[stage] source={source} episodes={count}")


if __name__ == "__main__":
    main()
