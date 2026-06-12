"""
Process REVISOR-25k dataset and upload to Hugging Face.

Reads 4 parquet training files, extracts metadata, organizes videos
into per-subset subdirectories, and pushes to HuggingFace Hub.

Usage:
    pip install datasets huggingface_hub pandas pyarrow
    huggingface-cli login
    python scripts/upload_to_huggingface.py
"""

import os
import json
import shutil
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, DatasetDict, Features, Value, Sequence

# ============ Configuration ============
HF_REPO_ID = "williamljz/REVISOR-25k"  # TODO: Change to your HF username
OUTPUT_DIR = "/workspace/images-ks3-starfs-hd/workspace/lijiaze/projects/DeepEyes-video-xiaomi/REVISOR-25k"

PARQUET_FILES = {
    "video_r1": "/workspace/images-ks3-starfs-hd/dataset/omni/Video-R1-data/Video-R1-video_sample_ratio_0p2.parquet",
    "time_r1": "/workspace/images-ks3-starfs-hd/dataset/omni/Video-R1-data/Time_r1_coc.parquet",
    "cg_bench": "/workspace/images-ks3-starfs-hd/dataset/omni/CG-Bench/CG-Bench-timeintervl-bigger-4seconds.parquet",
    "rextime": "/workspace/images-ks3-starfs-hd/dataset/omni/ReXTime/data/rextime-val-temporal_grounding.parquet",
}


def parse_value(val):
    """Parse a value that might be a string, dict, list, or numpy array."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return val
    return val


def extract_conversations(prompt_raw):
    """Extract conversation messages from the prompt field."""
    prompt = parse_value(prompt_raw)
    if not isinstance(prompt, list):
        return [], ""

    conversations = []
    system_prompt = ""
    for msg in prompt:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Flatten content list to text (ignore image/video placeholders)
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") in ("image", "video"):
                    text_parts.append("<video>")
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)

        if role == "system":
            system_prompt = content
        conversations.append({"role": role, "content": content})

    return conversations, system_prompt


def extract_record(row, subset_name):
    """Extract a standardized record from a raw parquet row."""
    # Parse videos field
    videos_raw = parse_value(row["videos"])
    video_path = ""
    video_fps = 1
    if isinstance(videos_raw, list) and len(videos_raw) > 0:
        v = videos_raw[0]
        if isinstance(v, dict):
            video_path = v.get("video", "")
            video_fps = v.get("fps", 1)

    # Parse reward_model
    reward_model = parse_value(row["reward_model"])
    ground_truth = ""
    if isinstance(reward_model, dict):
        ground_truth = str(reward_model.get("ground_truth", ""))

    # Parse extra_info
    extra_info = parse_value(row["extra_info"])
    duration = ""
    video_id = ""
    query = ""
    original_fps = ""
    if isinstance(extra_info, dict):
        duration = str(extra_info.get("duration", ""))
        video_id = str(extra_info.get("id", ""))
        query = str(extra_info.get("query", ""))
        original_fps = str(extra_info.get("fps", ""))

    # Parse conversations
    conversations, system_prompt = extract_conversations(row["prompt"])

    # Video filename (relative path for dataset)
    video_filename = os.path.basename(video_path) if video_path else ""

    return {
        "video": f"videos/{subset_name}/{video_filename}" if video_filename else "",
        "video_path_original": video_path,
        "video_filename": video_filename,
        "sample_fps": video_fps,
        "original_fps": original_fps,
        "duration": duration,
        "video_id": video_id,
        "conversations": json.dumps(conversations, ensure_ascii=False),
        "system_prompt": system_prompt,
        "query": query,
        "ground_truth": ground_truth,
        "data_source": str(row.get("data_source", "")),
        "env_name": str(row.get("env_name", "")),
        "ability": str(row.get("ability", "")),
        "subset": subset_name,
    }


def process_subset(subset_name, filepath):
    """Process one parquet file into a list of records."""
    print(f"\n[{subset_name}] Reading: {filepath}")
    df = pd.read_parquet(filepath)
    print(f"  Rows: {len(df)}")

    records = []
    errors = 0
    for idx, row in df.iterrows():
        try:
            record = extract_record(row, subset_name)
            records.append(record)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Warning row {idx}: {e}")
    if errors > 0:
        print(f"  Total errors: {errors}")
    print(f"  Successfully extracted: {len(records)} records")
    return records


MAX_FILES_PER_DIR = 9000  # HuggingFace limit is 10000, use 9000 for safety


def get_shard_subdir(index):
    """Get shard subdirectory name like part_00, part_01, etc."""
    shard_id = index // MAX_FILES_PER_DIR
    return f"part_{shard_id:02d}"


def copy_videos(records, subset_name, output_dir, mode="symlink", num_workers=32):
    """
    Organize video files into the output directory structure.
    Automatically shards into subdirectories if >9000 files (HF limit is 10000/dir).

    Args:
        mode: "symlink" (fastest, creates symbolic links),
              "hardlink" (fast, same filesystem only),
              "copy" (multi-threaded copy, slowest but portable)
        num_workers: number of threads for copy mode
    """
    base_video_dir = os.path.join(output_dir, "videos", subset_name)

    # Deduplicate by video filename
    tasks = {}
    for r in records:
        src = r["video_path_original"]
        if not src:
            continue
        filename = r["video_filename"]
        if filename not in tasks:
            tasks[filename] = src

    # Determine if sharding is needed
    need_shard = len(tasks) > MAX_FILES_PER_DIR
    if need_shard:
        n_shards = (len(tasks) + MAX_FILES_PER_DIR - 1) // MAX_FILES_PER_DIR
        print(f"    Sharding into {n_shards} subdirectories ({len(tasks)} files, max {MAX_FILES_PER_DIR}/dir)")

    # Assign each file a shard subdirectory
    file_list = list(tasks.items())  # [(filename, src), ...]
    file_to_shard = {}
    for i, (filename, src) in enumerate(file_list):
        if need_shard:
            shard_dir = get_shard_subdir(i)
            video_dir = os.path.join(base_video_dir, shard_dir)
            file_to_shard[filename] = shard_dir
        else:
            video_dir = base_video_dir
            file_to_shard[filename] = ""
        os.makedirs(video_dir, exist_ok=True)

    done = 0
    skipped = 0
    missing = 0
    errors = 0

    def process_one(filename, src):
        if need_shard:
            video_dir = os.path.join(base_video_dir, file_to_shard[filename])
        else:
            video_dir = base_video_dir
        dst = os.path.join(video_dir, filename)
        if os.path.exists(dst) or os.path.islink(dst):
            return "skipped"
        if not os.path.exists(src):
            return "missing"
        try:
            if mode == "symlink":
                os.symlink(os.path.abspath(src), dst)
            elif mode == "hardlink":
                os.link(src, dst)
            else:  # copy
                shutil.copy2(src, dst)
            return "done"
        except Exception as e:
            return f"error: {e}"

    if mode == "copy" and num_workers > 1:
        # Multi-threaded copy
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_one, fname, src): fname
                for fname, src in tasks.items()
            }
            for future in as_completed(futures):
                result = future.result()
                if result == "done":
                    done += 1
                elif result == "skipped":
                    skipped += 1
                elif result == "missing":
                    missing += 1
                else:
                    errors += 1
                # Progress
                total_processed = done + skipped + missing + errors
                if total_processed % 500 == 0:
                    print(f"    progress: {total_processed}/{len(tasks)}")
    else:
        # Symlink/hardlink are fast enough single-threaded
        for filename, src in tasks.items():
            result = process_one(filename, src)
            if result == "done":
                done += 1
            elif result == "skipped":
                skipped += 1
            elif result == "missing":
                missing += 1
            else:
                errors += 1

    print(f"  Videos ({mode}): done={done}, skipped={skipped}, missing={missing}, errors={errors}")
    return file_to_shard if need_shard else None


def main():
    parser = argparse.ArgumentParser(description="Process REVISOR-25k dataset for HuggingFace")
    parser.add_argument("--video-mode", choices=["symlink", "hardlink", "copy"], default="symlink",
                        help="How to handle video files: symlink (fastest), hardlink, or copy (default: symlink)")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="Number of threads for copy mode (default: 32)")
    parser.add_argument("--skip-videos", action="store_true",
                        help="Skip video file processing, only generate parquet metadata")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = {}
    total = 0

    # Step 1: Process each subset
    for subset_name, filepath in PARQUET_FILES.items():
        records = process_subset(subset_name, filepath)
        all_records[subset_name] = records
        total += len(records)

    print(f"\n{'='*60}")
    print(f"Total records: {total}")
    print(f"{'='*60}")

    # Step 2: Organize videos into per-subset subdirectories
    shard_maps = {}  # subset_name -> {filename: shard_subdir} or None
    if not args.skip_videos:
        print(f"\n[Step 2] Organizing video files (mode={args.video_mode})...")
        for subset_name, records in all_records.items():
            print(f"\n  [{subset_name}]")
            shard_map = copy_videos(records, subset_name, OUTPUT_DIR, mode=args.video_mode, num_workers=args.num_workers)
            shard_maps[subset_name] = shard_map
    else:
        print("\n[Step 2] Skipped video processing (--skip-videos)")

    # Step 3: Create HuggingFace Dataset with per-subset splits
    # Update video paths to reflect sharding
    print("\n[Step 3] Creating HuggingFace dataset...")

    dataset_dict = {}
    for subset_name, records in all_records.items():
        shard_map = shard_maps.get(subset_name)
        # Remove the original absolute path and update video path with shard info
        clean_records = []
        for r in records:
            clean = {k: v for k, v in r.items() if k != "video_path_original"}
            # Update relative video path if sharded
            if shard_map and r["video_filename"] in shard_map:
                shard_subdir = shard_map[r["video_filename"]]
                clean["video"] = f"videos/{subset_name}/{shard_subdir}/{r['video_filename']}"
            clean_records.append(clean)
        df = pd.DataFrame(clean_records)
        dataset_dict[subset_name] = Dataset.from_pandas(df, preserve_index=False)

    hf_dataset = DatasetDict(dataset_dict)
    print(f"\nDataset splits:")
    for split_name, ds in hf_dataset.items():
        print(f"  {split_name}: {len(ds)} samples")

    # Step 4: Save locally
    # Save as parquet files in each subset directory
    for subset_name, ds in hf_dataset.items():
        subset_dir = os.path.join(OUTPUT_DIR, "data", subset_name)
        os.makedirs(subset_dir, exist_ok=True)
        parquet_path = os.path.join(subset_dir, "train.parquet")
        ds.to_parquet(parquet_path)
        print(f"  Saved: {parquet_path}")

    # Also save the full DatasetDict
    hf_dataset.save_to_disk(os.path.join(OUTPUT_DIR, "dataset_arrow"))
    print(f"\nArrow dataset saved to: {OUTPUT_DIR}/dataset_arrow")

    # Step 5: Push to Hub (uncomment when ready)
    # hf_dataset.push_to_hub(HF_REPO_ID, private=False)
    # print(f"\nPushed to: https://huggingface.co/datasets/{HF_REPO_ID}")

    print(f"""
{'='*60}
DONE! Dataset prepared at: {OUTPUT_DIR}
{'='*60}

Directory structure:
  REVISOR-25k/
  ├── data/
  │   ├── video_r1/train.parquet      (20,855 samples - Video QA)
  │   ├── time_r1/train.parquet       (2,500 samples - Temporal Grounding)
  │   ├── cg_bench/train.parquet      (1,167 samples - Temporal Grounding)
  │   └── rextime/train.parquet       (837 samples - Temporal Grounding)
  ├── videos/
  │   ├── video_r1/                   (video files)
  │   ├── time_r1/                    (video files)
  │   ├── cg_bench/                   (video files)
  │   └── rextime/                    (video files)
  └── dataset_arrow/                  (HF arrow format)

To upload to HuggingFace:
  1. huggingface-cli login
  2. Option A (metadata only - fast):
       python -c "
       from datasets import load_from_disk
       ds = load_from_disk('{OUTPUT_DIR}/dataset_arrow')
       ds.push_to_hub('{HF_REPO_ID}')
       "
  3. Option B (with videos - use git lfs):
       huggingface-cli repo create REVISOR-25k --type dataset
       cd {OUTPUT_DIR}
       git init && git lfs install
       git lfs track "*.mp4"
       huggingface-cli upload {HF_REPO_ID} . . --repo-type dataset
""")


if __name__ == "__main__":
    main()
