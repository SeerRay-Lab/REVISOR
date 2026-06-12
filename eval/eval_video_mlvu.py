import argparse, json, os, sys
import cv2
import random
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import torch
from pathlib import Path 
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image
import base64
import decord
import io
from openai import OpenAI
import requests
from torch.utils.data import Dataset
from pandas import read_parquet
import re
import time

from verl.workers.agent.envs.mm_process_engine.prompt import PROMPT
from verl.workers.agent.envs.mm_process_engine.video_tool_v1 import VideoToolV1
from verl.workers.agent.envs.mm_process_engine.video_tool_v2 import VideoToolV2
from verl.workers.agent.envs.mm_process_engine.video_tool_v2_reflection import VideoToolV2Reflection
from verl.workers.agent.envs.mm_process_engine.video_tool_v1_reflection_single import VideoToolV1ReflectionSingle
from verl.workers.agent.envs.mm_process_engine.video_tool_new_v1 import VideoToolNEWV1
from verl.utils.dataset.vision_utils import process_video
from verl.utils.watermark import add_watermark_to_pil_image
from eval.generate_vmme_acc_table import generate_long_table

DATASET_INFO = {
    'LongVideoBench': 'LongVideoBench-path',
    'VideoMME': 'Video-MME-path',
    'MLVU': 'MLVU-path',
    'LVBench': 'LVBench-path'
}

def check_ans(pred, gt): 
    # 去掉前缀
    pred = pred.replace('Answer: ', '').strip()
    gt = gt.strip()

    # 提取首个选项字母（忽略大小写、去掉符号）
    def extract_option_and_content(text):
        text = text.lower().strip()
        # 匹配类似 a. / a, / a ajkll 或者 (a) / (a) ajkll
        match = re.match(r'^([a-zA-Z])[\.\,\)]?\s*(.*)$', text)
        if match:
            option = match.group(1)
            content = match.group(2).strip()
            return option, content
        # 如果没有匹配到上述格式，尝试提取括号中的选项 (a), (B), (c) 等
        match_in_brackets = re.match(r'^\(([a-zA-Z])\)\s*(.*)$', text)
        if match_in_brackets:
            option = match_in_brackets.group(1).lower()  # 统一转小写
            content = match_in_brackets.group(2).strip()
            return option, content
        return None, text  # 如果没匹配到，返回 None
    
    pred_option, pred_content = extract_option_and_content(pred)
    gt_option, gt_content = extract_option_and_content(gt)

    # 去掉 ground truth 内容末尾的句点
    if gt_content.endswith('.'):
        gt_content = gt_content[:-1].strip()

    # 只比较选项
    if pred_option and gt_option and pred_option == gt_option:
        return True
    
    return False


class MLVU_dataset(Dataset):
    def __init__(self):
        super().__init__()
        multi_options_tasks = ['1_plotQA', '2_needle', '3_ego', '4_count', '5_order', '6_anomaly_reco', '7_topic_reasoning']
        self.data_list = []
        data_root_dir = DATASET_INFO['MLVU']
        # files = json.load(os.path.join(data_root_dir, 'lvb_val.json'))
        json_files = {}
        for name in multi_options_tasks:
            with open(os.path.join(data_root_dir, 'json', name + '.json'), 'r') as f:
                files = json.load(f)
            json_files[name] = files
            
        for key, files in json_files.items():
            for meta_data in files:
                video_name = meta_data['video']
                video_path = os.path.join(data_root_dir, 'video', key, video_name)
                if not os.path.exists(video_path):
                    print(f'WARNING: {video_path} not found, skipping')
                    continue
                question = meta_data['question']
                options = meta_data['candidates']
                question_id = f'{video_name}__{question.replace(" ", "_")[:10]}__{"_".join([o.replace(" ", "_") for o in options])[:10]}'
                task_type = meta_data['question_type']
                answer = self.get_answer(meta_data)
                question = self.qa_template(meta_data)
                self.data_list.append({
                    'video_path': video_path,
                    'question_id': question_id,
                    'data': meta_data,
                    'video_frames': [],
                    'question': question,
                    'answer': answer,
                    'task_type': task_type
                })
        
        random.shuffle(self.data_list)
    
    def get_answer(self, meta_data):
        options = meta_data['candidates']
        text_answer = meta_data['answer']
        text_answer = text_answer.lower().strip('\n .')
        for ans_idx, o in enumerate(options):
            if o.lower().strip('\n .') == text_answer:
                return chr(ord('A') + ans_idx)
        raise ValueError(f'No correct answer found in:\nOptions: {options}\nAnswer: {text_answer}')
    
    def __len__(self):
        return len(self.data_list)
      
    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        for idx, c in enumerate([re.sub(r'^[A-Z]\.\s*', '', item) for item in data['candidates']]):
            question += f"({chr(ord('A') + idx)}) {c}\n"
        question = question.rstrip()
        return question
    
    def __getitem__(self, idx):
        
        return self.data_list[idx]


parser = argparse.ArgumentParser()
parser.add_argument("--video_duration_types", type=str, default='long', help='Evaluate with specific video duration types, e.g. short,long or ["short","long"]')
parser.add_argument('--eval_model_name', type=str, default="coc", help='Model name for evaluation')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--data_dir', default='') #不用传

parser.add_argument('--nframe', type=int, default=256)
parser.add_argument('--fps_max_tokens', type=int, default=64)
parser.add_argument('--start_port', type=int, default=10086, help='local api start port')
parser.add_argument('--n_devices', type=int, default=1, help='use how many devices as servers')
parser.add_argument('--save_path', type=str, default="outputs/temp", help='Path to save the results')
parser.add_argument('--version', type=str, default='NEW_v1')
parser.add_argument('--agent_clip_frames', type=int, default=64)
args = parser.parse_args()

if not args.save_path.endswith('mlvu_results'):
    args.save_path = args.save_path + '/mlvu_results'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)

dataset = MLVU_dataset()

clients = [
    OpenAI(
        api_key='EMPTY',
        base_url=f'http://localhost:{args.start_port + i}/v1',
    ) for i in range(args.n_devices)
]

eval_model_name = args.eval_model_name


tool_version = args.version
sys_prompt = getattr(PROMPT, f'SYSTEM_PROMPT_{tool_version}')
user_prompt = getattr(PROMPT, f'USER_PROMPT_{tool_version}')

if tool_version == 'v2_reflection':
    tool_cls = VideoToolV2Reflection
    tool_name = 'video_tool_v2_reflection'
elif tool_version == 'v1_reflection_single':
    tool_cls = VideoToolV1ReflectionSingle
    tool_name = 'video_tool_v1_reflection_single'
elif tool_version == "NEW_v1":
    tool_cls = VideoToolNEWV1
    tool_name = "video_tool_new_v1"
elif 'v1' in tool_version:
    tool_cls = VideoToolV1
    tool_name = 'video_tool_v1'
else:
    tool_cls = VideoToolV2
    tool_name = 'video_tool_v2'


start_token = "<time_interval>"
end_token = "</time_interval>"

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="jpeg")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def get_print_msg(input_msg):
    print_msg = []
    for msg in input_msg:
        role = msg['role']
        content = msg['content']

        new_content = []
        if isinstance(content, str):
            print_msg.append(msg)
            continue

        for content_item in content:
            t = content_item['type']
            if t == 'video_url':
                new_content.append({
                    'type': t, 'video_url': 'video_frames'
                })
            else:
                new_content.append(content_item)

        print_msg.append({
            'role': role,
            'content': new_content
        })

    return print_msg


def process(video_args):
    metadata = video_args['data']
    metadata.pop('candidates')
    # import pdb; pdb.set_trace()
    if clients is not None:
        client = random.choice(clients)
    question_id=video_args['question_id']
    save_path = os.path.join(args.save_path,f"{question_id}.json")
    # if os.path.exists(save_path):
    #     return json.load(open(save_path, 'r'))

    question = video_args['question'].replace('Question:', "").strip()
    question = question.replace('Options:\n', '').replace('Options:', '')
    video_path = video_args['video_path']
    vr = decord.VideoReader(video_path)
    n_video_frames = len(vr)
    video_duration = n_video_frames / vr.get_avg_fps()
    
    video_start = 0.0
    video_end = video_duration

    
    del vr
    video_tensor, sample_fps = process_video(
            {"video": video_path},
            fps=1,
            fps_min_frames=4,
            fps_max_frames=512,
            fps_min_pixels=6272,
            fps_max_pixels=100352,
            total_pixels=6422528,
        )
    timestampes = torch.linspace(video_start, video_end, video_tensor.shape[0]).tolist()
    
    tool = tool_cls(tool_name, "Tool for video zooming", {})
    origin_multi_modal_data={"video":[{"video":video_path}]}
    multi_modal_data={"video":[video_tensor]}
    tool.reset(raw_prompt="",multi_modal_data=multi_modal_data, origin_multi_modal_data=origin_multi_modal_data, agent_clip_frames=args.agent_clip_frames)
    base64_image=[]
    for i in range(len(video_tensor)):
        image = Image.fromarray(video_tensor[i].permute(1, 2, 0).byte().numpy())
        if tool_version != 'only_qa':
            image = add_watermark_to_pil_image(pil_image=image, text=f'{timestampes[i]:.1f}')
        base64_image.append(encode_pil_image_to_base64(image))
    use_prompt_suffix = """Format strictly as:
            <think>Your reasoning steps</think><time_interval>[start_time, end_time]</time_interval>"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_image)}"}},
            {"type": "text", "text": f"\n{question.strip()}\n The video durations is {video_duration:.1f} seconds."+use_prompt_suffix}
        ]}
    ]
    print_messages = get_print_msg(messages)

    chat_message = messages
    response_message = ""
    video_kwargs={'fps': [sample_fps]}
    
    status = 'success'
    chat_turn = 0
    chat_turn_Flag = False
    try:
        while '</answer>' not in response_message:
            # import pdb; pdb.set_trace()
            if '</answer>' in response_message and '<answer>' in response_message:
                break

            if chat_turn > 2:
                chat_turn_Flag = True
                break

            params = {
                "model": eval_model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_completion_tokens": 16 if tool_version == 'only_qa' else 2048,
                "stop": ["<|im_end|>\n".strip()],
                "extra_body": {"mm_processor_kwargs": video_kwargs}
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content
            
            if start_token in response_message:
                obs, reward, done, info = tool.execute(
                    response_message,
                    eval_mode=False,
                    fps_max_pixels=args.fps_max_tokens * 28 * 28,
                    fps_min_pixels=args.fps_max_tokens // 2 * 28 * 28
                )
                video_data=obs['multi_modal_data']['video'][0]
                video_kwargs['fps'].extend(obs['multi_modal_data']["fps"].tolist())
                new_base64_image=[]
                for j in range(len(video_data)):
                    # obs returned from tool is already watermarked
                    image = Image.fromarray(video_data[j].permute(1, 2, 0).byte().numpy())
                    new_base64_image.append(encode_pil_image_to_base64(image))

                new_image_message={"type": "video_url","video_url": {"url": f"data:video/jpeg;base64,{','.join(new_base64_image)}"}}
                    
                content_f = []
                content_f.append(new_image_message)
                content_f.append({"type": "text", "text": user_prompt})
            
                _message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": content_f,
                    }
                ]

                chat_message.extend(_message)
                p_message = get_print_msg(_message)
                print_messages.extend(p_message)
            else:
                p_message = [
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                ]
                print_messages.extend(p_message)                
            chat_turn += 1
            if 'only_qa' == args.version:
                break

    except Exception as e:
        print(f"Error!!!!", e)
        chat_turn_Flag = True
        status = f'error: {e}'
    if chat_turn_Flag:
        messages = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_image)}"}},
            {"type": "text", "text": "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"+ question +"\nOnly give the best option."}
        ]}
    ]
        params = {
                "model": eval_model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_completion_tokens": 16 if tool_version == 'only_qa' else 2048,
                "stop": ["<|im_end|>\n".strip()],
                "extra_body": {"mm_processor_kwargs": {'fps': [sample_fps]}}
            }
        response = client.chat.completions.create(**params)
        response_message = response.choices[0].message.content
        status = 'second_success'

    if '</answer>' in response_message and '<answer>' in response_message:
        output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
    else:
        output_text = response_message

    save_info = {}
    save_info['question_id'] = question_id
    save_info['video_path'] = video_args['video_path']
    save_info['question'] = question
    save_info['answer'] = video_args["answer"]
    save_info['pred_ans'] = output_text
    save_info['acc'] = check_ans(save_info['pred_ans'], save_info['answer'] )
    save_info['pred_output'] = print_messages
    save_info['status'] = status
    save_info['metadata'] = metadata
    # print(save_info)
    json.dump(save_info, open(save_path, 'w'), indent=2)
    return save_info


if __name__ == "__main__":
    save_name = f"result_mlvu_{args.video_duration_types}_{args.eval_model_name}.jsonl"
    save_json = []
    save_path = args.save_path
    
    print("multi-processing......")
    pool = multiprocessing.Pool(processes=args.n_devices * 1)
    # pool = multiprocessing.Pool(processes=1)
    with tqdm(total=len(dataset), desc="Processing mlvu "+str(args.video_duration_types)) as pbar:
        for result in pool.imap(process, dataset):
            if result is not None:
                save_json.append(result)
                pbar.update(1)
    pool.close()
    pool.join()

    generate_long_table(json_files=save_json, data_dir=Path(save_path))
