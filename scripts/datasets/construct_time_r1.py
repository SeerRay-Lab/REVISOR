#%%
import decord
import json
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
import subprocess
from datasets import load_dataset
import os

def get_video_metadata_ffmpeg(video_path):
    """
    Get comprehensive video metadata using ffmpeg/ffprobe.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Dictionary containing duration, fps, resolution, frame count, and other metadata.
              Returns None if video cannot be processed.
    """
    video_path = str(video_path)
    
    # Command to get all video stream information in JSON format
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,codec_name,pix_fmt",
        "-show_entries", "format=duration,size,bit_rate,format_name",
        "-of", "json",
        video_path
    ]
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        
        if not data.get("streams") or len(data["streams"]) == 0:
            return None
            
        stream = data["streams"][0]
        format_info = data.get("format", {})
        
        # Calculate frame rate (try both r_frame_rate and avg_frame_rate)
        def parse_fraction(fraction_str):
            try:
                num, den = map(float, fraction_str.split('/'))
                return num / den if den != 0 else 0
            except:
                return 0
                
        r_frame_rate = parse_fraction(stream.get("r_frame_rate", "0/0"))
        avg_frame_rate = parse_fraction(stream.get("avg_frame_rate", "0/0"))
        fps = avg_frame_rate if avg_frame_rate > 0 else r_frame_rate
        
        # Get duration (prefer stream duration, fallback to format duration)
        duration = float(stream.get("duration", 0))
        if duration == 0 and "duration" in format_info:
            duration = float(format_info["duration"])
        
        # Get frame count (prefer nb_frames, fallback to duration*fps)
        frame_count = int(stream.get("nb_frames", 0))
        if frame_count == 0:
            frame_count = int(duration * fps)
        
        return {
            'resolution': [stream["width"], stream["height"]],
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': duration,
        }
        
    except Exception as e:
        print(f"Error processing video with ffprobe: {e}")
        return None

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--json_path',
        type=str,
        default='/workspace/images-ks3-starfs-hd/workspace/xuboshen/timer1_project/TimeR1_mi/dataset/trainval/train_2k5.json' 
    )
    args, _ = parser.parse_known_args()

    src_json_path = Path(args.json_path)

    nframes = 64


    src = json.load(src_json_path.open('r'))
    tgt = []
    # tgt_json_path = src_json_path.with_suffix(f'.rl_{version}_{nframes}.json')
    tgt_parquet_path = '/workspace/images-ks3-starfs-hd/dataset/omni/Video-R1-data/Time_r1_coc.parquet'
    # %%
    for item in tqdm(src):
        try:
            video_path = os.path.join('/workspace/images-ks3-starfs-hd/workspace/wangziheng/DATASET', item['video'].replace('./dataset/DATASET/', ''))
            info = get_video_metadata_ffmpeg(video_path=video_path)
            video_duration = item['duration']
            raw_query = item['sentence']
            gt_duration_range = item['timestamp']
            use_prompt_preffix = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.
Output your thought process within the <think> </think> tags. Then, provide the start and end times (in seconds) in the format "[start_time, end_time]" within the <time_interval> </time_interval> tags. For example: "<think>...</think><time_interval>[12.5, 19.8]</time_interval>"."""
            item = {
                'prompt': [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'content': "<video>\n"+use_prompt_preffix.replace("[EVENT]", raw_query) + f"The video durations is {video_duration:.1f} seconds.", 'role': 'user'}],
                'videos': [
                    {'fps': 1, 'type': 'video', 'video': video_path}
                ],
                'data_source': "temporal_grounding",
                'env_name': f'video_tool_temporal_grounding_v1',
                'reward_model': {'ground_truth': str(gt_duration_range), 'style': 'math'},
                'ability': "temporal grounding",
                'extra_info': {
                    'duration': str(video_duration),
                    'fps': str(info['fps']),
                    'id': os.path.basename(item['video']).replace('.mp4',''),
                    'query': raw_query,
                }
            }
            tgt.append(item)
        except Exception as e:
            continue

    tgt = pd.DataFrame(tgt)
    tgt.to_parquet(str(tgt_parquet_path))

    # Verify it can be loaded by datasets
    dataset = load_dataset('parquet', data_files=str(tgt_parquet_path))
