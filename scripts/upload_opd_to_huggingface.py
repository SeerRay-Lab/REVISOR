"""
Process Video-opd-Dataset and upload to Hugging Face.

Reads the OPD JSON file, extracts metadata, organizes videos
into the output directory, and prepares for HuggingFace upload.

Usage:
    pip install datasets huggingface_hub pandas pyarrow
    huggingface-cli login
    python scripts/upload_opd_to_huggingface.py [--video-mode symlink]
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset

# ============ Configuration ============
HF_REPO_ID = "williamljz/Video-opd-Dataset"
OUTPUT_DIR = "/workspace/images-ks3-starfs-hd/workspace/lijiaze/projects/DeepEyes-video-xiaomi/Video-opd-Dataset"

JSON_FILE = "/workspace/images-ks3-starfs-hd/workspace/lijiaze/projects/dataprocess/scripts/opd/temp_results_timelens_qwen3vl32binstruct_timelenfiltered2p5knothinkinfer_grpo_1epoch_llmtrained_newdataset_maxtoken768_no_think_vs_qwen3vl8bnothink_2k5_diff_top_k_desc_iou_diff_reward_msswift_opd_nothink.json"


def extract_records(data):
    """Extract standardized records from the JSON data."""
    records = []
    for i, item in enumerate(data):
        # Get video path
        video_paths = item.get("videos", [])
        video_path = video_paths[0] if video_paths else ""
        video_filename = os.path.basename(video_path) if video_path else ""

        # Determine subset from video path
        # e.g., .../TimeLens-100K/videos/cosmo_cap/xxx.mp4 -> cosmo_cap
        subset = ""
        if video_path:
            parts = video_path.split("/")
            # Find "videos" in path and take the next part as subset
            for j, part in enumerate(parts):
                if part == "videos" and j + 1 < len(parts):
                    subset = parts[j + 1]
                    break

        # Extract messages
        messages = item.get("messages", [])
        system_prompt = ""
        user_content = ""
        assistant_content = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content

        records.append({
            "video": f"videos/{subset}/{video_filename}" if video_filename else "",
            "video_path_original": video_path,
            "video_filename": video_filename,
            "subset": subset,
            "system_prompt": system_prompt,
            "user_query": user_content,
            "assistant_response": assistant_content,
            "messages": json.dumps(messages, ensure_ascii=False),
        })

    return records


def copy_videos(records, output_dir, mode="symlink", num_workers=32):
    """
    Organize video files into per-subset subdirectories.

    Args:
        mode: "symlink" (fastest), "hardlink", or "copy" (multi-threaded)
        num_workers: number of threads for copy mode
    """
    # Deduplicate by (subset, filename)
    tasks = {}  # (subset, filename) -> src_path
    for r in records:
        src = r["video_path_original"]
        if not src:
            continue
        key = (r["subset"], r["video_filename"])
        if key not in tasks:
            tasks[key] = src

    # Create directories
    subsets = set(k[0] for k in tasks.keys())
    for subset in subsets:
        os.makedirs(os.path.join(output_dir, "videos", subset), exist_ok=True)

    print(f"  Total unique videos: {len(tasks)} across subsets: {sorted(subsets)}")

    done = 0
    skipped = 0
    missing = 0
    errors = 0

    def process_one(key, src):
        subset, filename = key
        dst = os.path.join(output_dir, "videos", subset, filename)
        if os.path.exists(dst) or os.path.islink(dst):
            return "skipped"
        if not os.path.exists(src):
            return "missing"
        try:
            if mode == "symlink":
                os.symlink(os.path.abspath(src), dst)
            elif mode == "hardlink":
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
            return "done"
        except Exception as e:
            return f"error: {e}"

    if mode == "copy" and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_one, key, src): key
                for key, src in tasks.items()
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
                total_processed = done + skipped + missing + errors
                if total_processed % 500 == 0:
                    print(f"    progress: {total_processed}/{len(tasks)}")
    else:
        for key, src in tasks.items():
            result = process_one(key, src)
            if result == "done":
                done += 1
            elif result == "skipped":
                skipped += 1
            elif result == "missing":
                missing += 1
            else:
                errors += 1

    print(f"  Videos ({mode}): done={done}, skipped={skipped}, missing={missing}, errors={errors}")


def main():
    parser = argparse.ArgumentParser(description="Process Video-opd-Dataset for HuggingFace")
    parser.add_argument("--video-mode", choices=["symlink", "hardlink", "copy"], default="symlink",
                        help="How to handle video files (default: symlink)")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="Number of threads for copy mode (default: 32)")
    parser.add_argument("--skip-videos", action="store_true",
                        help="Skip video file processing, only generate metadata")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load JSON data
    print(f"[Step 1] Loading JSON: {JSON_FILE}")
    with open(JSON_FILE, "r") as f:
        data = json.load(f)
    print(f"  Total samples: {len(data)}")

    # Step 2: Extract records
    print(f"\n[Step 2] Extracting metadata...")
    records = extract_records(data)
    print(f"  Extracted: {len(records)} records")

    # Print subset stats
    from collections import Counter
    subset_counts = Counter(r["subset"] for r in records)
    for subset, count in sorted(subset_counts.items()):
        print(f"    {subset}: {count} samples")

    # Step 3: Organize videos
    if not args.skip_videos:
        print(f"\n[Step 3] Organizing video files (mode={args.video_mode})...")
        copy_videos(records, OUTPUT_DIR, mode=args.video_mode, num_workers=args.num_workers)
    else:
        print(f"\n[Step 3] Skipped video processing (--skip-videos)")

    # Step 4: Create HuggingFace Dataset
    print(f"\n[Step 4] Creating HuggingFace dataset...")
    # Remove internal fields
    clean_records = []
    for r in records:
        clean = {k: v for k, v in r.items() if k != "video_path_original"}
        clean_records.append(clean)

    dataset = Dataset.from_list(clean_records)
    print(f"  Dataset: {dataset}")

    # Save as parquet
    data_dir = os.path.join(OUTPUT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    parquet_path = os.path.join(data_dir, "train.parquet")
    dataset.to_parquet(parquet_path)
    print(f"  Saved: {parquet_path}")

    # Step 5: Create README
    readme_content = f"""---
dataset_info:
  features:
    - name: video
      dtype: string
    - name: video_filename
      dtype: string
    - name: subset
      dtype: string
    - name: system_prompt
      dtype: string
    - name: user_query
      dtype: string
    - name: assistant_response
      dtype: string
    - name: messages
      dtype: string
  splits:
    - name: train
      num_examples: {len(records)}
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.parquet
license: apache-2.0
task_categories:
  - video-text-to-text
  - visual-question-answering
language:
  - en
tags:
  - temporal-grounding
  - video-understanding
  - video-qa
size_categories:
  - 1K<n<10K
---

# Video-opd-Dataset

A video temporal grounding dataset with 2,500 samples sourced from TimeLens-100K.

## Dataset Description

This dataset contains video temporal grounding QA pairs where the model needs to identify precise time intervals for described events in videos.

### Data Statistics

| Subset | Samples | Description |
|--------|---------|-------------|
| cosmo_cap | {subset_counts.get('cosmo_cap', 0)} | Cosmo caption videos |
| queryd | {subset_counts.get('queryd', 0)} | QueryD videos |
| hirest | {subset_counts.get('hirest', 0)} | HiREST videos |
| internvid_vtime | {subset_counts.get('internvid_vtime', 0)} | InternVid VTime videos |
| didemo | {subset_counts.get('didemo', 0)} | DiDeMo videos |

**Total unique videos:** 2,268

### Data Format

Each sample contains:
- `video`: relative path to the video file
- `subset`: source subset name
- `system_prompt`: system prompt for the model
- `user_query`: user's temporal grounding query
- `assistant_response`: model's response with temporal boundaries
- `messages`: full conversation in JSON format

### Directory Structure

```
Video-opd-Dataset/
├── data/
│   └── train.parquet          (2,500 samples)
├── videos/
│   ├── cosmo_cap/             (video files)
│   ├── queryd/                (video files)
│   ├── hirest/                (video files)
│   ├── internvid_vtime/       (video files)
│   └── didemo/                (video files)
└── README.md
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("williamljz/Video-opd-Dataset")
print(dataset["train"][0])
```

## License

Apache 2.0
"""

    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"  README saved: {readme_path}")

    print(f"""
{'='*60}
DONE! Dataset prepared at: {OUTPUT_DIR}
{'='*60}

Directory structure:
  Video-opd-Dataset/
  ├── data/train.parquet         ({len(records)} samples)
  ├── videos/
  │   ├── cosmo_cap/
  │   ├── queryd/
  │   ├── hirest/
  │   ├── internvid_vtime/
  │   └── didemo/
  └── README.md

To upload to HuggingFace:
  1. huggingface-cli login
  2. huggingface-cli repo create Video-opd-Dataset --type dataset  (if not created yet)
  3. cd {OUTPUT_DIR}
  4. huggingface-cli upload {HF_REPO_ID} . . --repo-type dataset
""")


if __name__ == "__main__":
    main()
