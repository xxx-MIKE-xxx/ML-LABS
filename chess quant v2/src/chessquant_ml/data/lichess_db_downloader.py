from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


def build_parquet_endpoint(dataset: str) -> str:
    return f"https://datasets-server.huggingface.co/parquet?dataset={dataset}"


def fetch_parquet_metadata(dataset: str) -> dict[str, Any]:
    url = build_parquet_endpoint(dataset)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "parquet_files" not in payload:
        raise RuntimeError(f"Unexpected response from {url}: missing 'parquet_files'")
    return payload


def filter_shards(
    parquet_files: list[dict[str, Any]],
    config: str,
    split: str,
) -> list[dict[str, Any]]:
    shards = [
        item
        for item in parquet_files
        if item.get("config") == config and item.get("split") == split
    ]
    if not shards:
        raise RuntimeError(f"No shards found for config='{config}', split='{split}'")
    return shards


def dedupe_shards_by_url(shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_urls: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for shard in shards:
        url = str(shard["url"])
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(shard)

    return deduped


def build_unique_local_filenames(
    shards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Ensure that every selected shard gets a unique local filename.

    HF metadata can contain repeated bare filenames. If two different URLs share
    the same filename, later downloads would overwrite earlier ones. We avoid
    that by adding a short URL hash suffix only when needed.
    """
    filename_counts: dict[str, int] = {}
    for shard in shards:
        filename = str(shard["filename"])
        filename_counts[filename] = filename_counts.get(filename, 0) + 1

    result: list[dict[str, Any]] = []
    used_local_names: set[str] = set()

    for shard in shards:
        filename = str(shard["filename"])
        url = str(shard["url"])

        if filename_counts[filename] == 1:
            local_filename = filename
        else:
            path = Path(filename)
            short_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
            local_filename = f"{path.stem}__{short_hash}{path.suffix}"

        if local_filename in used_local_names:
            raise RuntimeError(
                f"Local filename collision remained after disambiguation: {local_filename}"
            )

        used_local_names.add(local_filename)
        enriched = dict(shard)
        enriched["local_filename"] = local_filename
        result.append(enriched)

    return result


def pick_tail_under_cap(
    shards: list[dict[str, Any]],
    max_download_bytes: int,
    max_shards: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total = 0

    # Take from the tail as a "recent-ish" heuristic.
    for shard in reversed(shards):
        size = int(shard["size"])
        if total + size > max_download_bytes:
            continue
        selected.append(shard)
        total += size
        if max_shards is not None and len(selected) >= max_shards:
            break

    selected.reverse()
    return selected


def download_file(url: str, out_path: Path, max_retries: int = 5) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0))

                with tmp_path.open("wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=out_path.name,
                    leave=False,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            tmp_path.replace(out_path)
            return

        except requests.exceptions.RequestException as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

            if attempt == max_retries:
                raise

            tqdm.write(
                f"Retry {attempt}/{max_retries} failed for {out_path.name}: {e}"
            )
            time.sleep(min(10 * attempt, 60))


def gb_to_bytes(value_gb: float) -> int:
    return int(value_gb * 1024 * 1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face parquet shards with a strict size cap."
    )
    parser.add_argument(
        "--dataset",
        default="Lichess/standard-chess-games",
        help="HF dataset name, e.g. Lichess/standard-chess-games",
    )
    parser.add_argument(
        "--config",
        default="default",
        help="Dataset config name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/lichess_db",
        help="Directory to store downloaded parquet shards",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=20.0,
        help="Maximum total parquet bytes to download, in GB",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Optional maximum number of shards to download",
    )
    parser.add_argument(
        "--manifest-json",
        default="data/registry/hf_shard_manifest.json",
        help="Path to save the selected shard manifest",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = fetch_parquet_metadata(args.dataset)
    parquet_files = payload["parquet_files"]

    filtered = filter_shards(parquet_files, args.config, args.split)
    deduped = dedupe_shards_by_url(filtered)
    deduped_with_names = build_unique_local_filenames(deduped)

    selected = pick_tail_under_cap(
        shards=deduped_with_names,
        max_download_bytes=gb_to_bytes(args.max_gb),
        max_shards=args.max_shards,
    )

    if not selected:
        raise RuntimeError("No shards fit under the requested cap.")

    total_bytes = sum(int(s["size"]) for s in selected)

    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "max_gb": args.max_gb,
        "filtered_count": len(filtered),
        "deduped_count": len(deduped_with_names),
        "selected_count": len(selected),
        "selected_total_bytes": total_bytes,
        "selected_total_gb": round(total_bytes / (1024**3), 3),
        "partial": payload.get("partial"),
        "selected_shards": [
            {
                "filename": s["filename"],
                "local_filename": s["local_filename"],
                "size": int(s["size"]),
                "url": s["url"],
            }
            for s in selected
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    removed_duplicates = len(filtered) - len(deduped_with_names)
    print(
        f"Filtered {len(filtered)} shard entries, removed {removed_duplicates} duplicate URLs."
    )
    print(
        f"Selected {len(selected)} unique shards "
        f"({manifest['selected_total_gb']} GB total) "
        f"for {args.dataset} / {args.config} / {args.split}"
    )
    print(f"Manifest: {manifest_path}")

    for shard in tqdm(selected, desc="Shards", unit="shard"):
        filename = shard["filename"]
        local_filename = shard["local_filename"]
        url = shard["url"]
        out_path = out_dir / local_filename

        if args.skip_existing and out_path.exists():
            tqdm.write(f"Skip existing: {local_filename}")
            continue

        tqdm.write(f"Downloading: {local_filename} (source: {filename})")

        try:
            download_file(url, out_path)
        except requests.exceptions.RequestException as e:
            tqdm.write(f"FAILED: {local_filename} -> {e}")
            continue

    print(f"Done. Files saved in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())