#%%
import decord
import json
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
from datasets import load_dataset
from verl.workers.agent.envs.mm_process_engine.prompt import PROMPT
import copy
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--json_path',
        type=str,
        default='/workspace/images-ks3-starfs-hd/dataset/omni/Video-R1-data/Video-R1-video_sample_ratio_0p2.json' 
    )
    args, _ = parser.parse_known_args()

    src_json_path = Path(args.json_path)
    version = 'v1_gqa'
    nframes = 64

    user_prompt_template = getattr(PROMPT, f'USER_PROMPT_NEW_{version}')
    system_prompt_template = getattr(PROMPT, f'SYSTEM_PROMPT_NEW_{version}')
    # src = json.load(src_json_path.open('r'))
    src = json.load(open('/workspace/images-ks3-starfs-hd/dataset/omni/CG-Bench/cgbench_filtered_shorter_10min.json', 'r'))
    src.extend(json.load(open('/workspace/images-ks3-starfs-hd/dataset/omni/CG-Bench/cgbench_filtered_bigger_10min.json', 'r')))
    
    
    tgt = []
    # tgt_json_path = src_json_path.with_suffix(f'.rl_{version}_{nframes}.json')
    tgt_parquet_path = '/workspace/images-ks3-starfs-hd/dataset/omni/CG-Bench/CG-Bench-timeintervl-bigger-4seconds.parquet'
    print(f'processing {src_json_path} with prompt version {version}')
    for item in tqdm(src):
        try:
            video_path = item['video_path']
            duration = item['duration']
            qa_list = item['items']
            
            for qa in qa_list:
                q = qa['question']
                choices = qa['choices']
                options = ''
                internvals = item['items'][0]['clue_intervals'][0]
                if len(choices) > 0:
                    for ii, op in enumerate(choices):
                        options += chr(65 + ii) + '. ' + op + '\n'
                use_prompt = copy.deepcopy(user_prompt_template)
                system_prompt = copy.deepcopy(system_prompt_template)
                system_prompt = system_prompt
                use_prompt_suffix = """Format strictly as:
                <think>Your reasoning steps</think><time_interval>[start_time, end_time]</time_interval>"""
                item = {
                    'prompt': [
                        {'role': 'system', 'content': system_prompt},
                        {'content': f"<video>\n{q.strip()}\n{options} The video durations is {duration:.1f} seconds."+use_prompt_suffix , 'role': 'user'}],
                    'videos': [
                        {'fps': 1, 'type': 'video', 'video': video_path}
                    ],
                    'data_source': "temporal_grounding",
                    'env_name': f'video_tool_temporal_grounding_v1',
                    'reward_model': {'ground_truth': str(internvals), 'style': 'math'},
                    'ability': "temporal grounding",
                    'extra_info': {
                        'duration': str(duration),
                        'fps': str(item['fps']),
                        'id': str(qa['qid']),
                        'query': q,
                    }
                }
                tgt.append(item)
        except Exception as e:
            continue
        
    with open(tgt_parquet_path.replace('.parquet', '.json'), 'w') as f:
        json.dump(tgt, f, indent=4)

    tgt = pd.DataFrame(tgt)
    tgt.to_parquet(str(tgt_parquet_path))

    # Verify it can be loaded by datasets
    dataset = load_dataset('parquet', data_files=str(tgt_parquet_path))
