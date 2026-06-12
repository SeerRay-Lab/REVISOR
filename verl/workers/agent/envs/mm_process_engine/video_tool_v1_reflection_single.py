import numpy as np
import copy
from verl.workers.agent.tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
from verl.workers.agent.envs.mm_process_engine.prompt import PROMPT
from verl.utils.watermark import add_watermark_to_pil_image
from math import ceil, floor
# 临时修复
# ToolBase.registry = {}
from qwen_vl_utils import fetch_video, smart_resize
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import cv2
import numpy as np
from PIL import Image
from typing import Tuple
import torch
import decord
import time
import logging
import math
logger = logging.getLogger(__name__)

FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 256
FPS_MAX_FRAMES = 64  # qwen2p5vl original setting

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
        float: the sample fps of the video.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
    max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)        
        nframes = total_frames / video_fps * fps
    
    # ensure min_frames < nframes < total_frames and disvible by FRAME_FACTOR
    if nframes > total_frames:
        print(f"Warning:smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    
    if ele.get('fix_frames_fps', False):
        return nframes, 2.0

    # ensure 2/sample_fps is an integer
    new_fps = nframes / total_frames * video_fps
    second_per_grid_t = 2 / new_fps
    if second_per_grid_t.is_integer() == False:
        closest_int = round(second_per_grid_t)
        if closest_int < 1:
            closest_int = 1
        new_fps = 2 / closest_int
        nframes = total_frames / video_fps * new_fps
        nframes = floor_by_factor(max(nframes, min_frames), FRAME_FACTOR)

    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes, new_fps   

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {200}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def _read_video_decord_w_timestamp(
    ele: dict,
) -> Tuple[torch.Tensor, float]:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()

    # TODO: support start_pts and end_pts
    video_start = ele.get("video_start", 0.0)
    video_end = ele.get("video_end", total_frames / video_fps)

    start_frame = max(0, int(video_start * video_fps))
    end_frame = min(total_frames, int(video_end * video_fps))
    if end_frame <= start_frame:
        end_frame = start_frame + 1
        if end_frame > total_frames:
            end_frame = total_frames
            start_frame = max(0, end_frame - 1)
    effective_frames = end_frame - start_frame
    logger.info(
        f"decord: {video_path=}, {effective_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    )
    # nframes, sample_fps = smart_nframes(ele, total_frames=effective_frames, video_fps=video_fps)
    nframes = ele.get('nframes', 64)
    # sample_fps = nframes / max(effective_frames, 1e-6) * video_fps
    # if sample_fps > 2.0:
    #     sample_fps = 2.0
    #     param_dict = {'fps': sample_fps}
    #     nframes, _ = smart_nframes(param_dict, total_frames=effective_frames, video_fps=video_fps)
        
    idx = (
        torch.linspace(start_frame, end_frame - 1, nframes).round().long().tolist()
    )
    timestamps = torch.linspace(video_start, video_end, nframes).tolist()
    video = vr.get_batch(idx).asnumpy()
    images = []
    for i, timestamp in zip(range(len(video)), timestamps):
        raw_image = Image.fromarray(video[i])
        watermarked_image = add_watermark_to_pil_image(raw_image, text=f'{timestamp:.1f}')
        images.append(np.asarray(watermarked_image))
    video = np.stack(images)

    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(effective_frames, 1e-6) * video_fps
    resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
            )
    video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
    return video, sample_fps




class VideoToolV1ReflectionSingle(ToolBase):
    name = "video_tool_v1_reflection_single"

    user_prompt = PROMPT.USER_PROMPT_video_v1
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed


    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None
        
    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            A dictionary with the tool name and arguments.
            
        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.P
            info: Additional info.
        """
        # action_string = """<tool_call>{"name":"video_zoom_in_tool", "arguments":{"interval":"0.1 to 2.1"}}</tool_call>"""
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}
        action = self.extract_action(action_string)
        if not action:
            return "", 0.0, True, {}
        try:
            tool_call = json.loads(action.strip())  # 或使用 literal_eval
        except Exception as e:
            error_msg = f"Invalid tool call format: {action.strip()}. Error: {e}"
            obs = "\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n"
            info = {"error": str(e), "status": "failed"}
            return obs, 0.0, False, {}
        try:

            tool_name = tool_call["name"]
            args = tool_call["arguments"]
        
            if tool_name == "temporal_zoom_in_tool" or tool_name == 'video_zoom_in_tool':
                interval = args["interval"]
                start_time, end_time = (float(x.strip()) for x in interval.split("to"))
                interval = self.maybe_resize_interval(start_time, end_time)
                if interval is None or not interval or end_time - start_time <= 3:
                    raise ValueError(f"ZOOM IN ARGUMENTS ARE INVALID")
                print(f'[DEBUG] Execute video zoom in tool {interval=}')
                start_time, end_time = interval
                # video_path = self.multi_modal_data["video"][0]["video"] # {'max_frames': 32, 'min_frames': 4, 'n_frames': 16, 'type': 'video', 'video': 'data/Interaction_T1_28.mp4'}
                ele = {
                    "video": self.video_path,
                    "video_start": start_time,
                    "video_end": end_time,
                    "nframes": self.reading_frame_cnt,  # Number of frames to sample
                    "resized_height": self.resized_height,
                    "resized_width": self.resized_width,
                }
                video_data, fps = _read_video_decord_w_timestamp(ele)
                
                # video_data, fps = fetch_video({"video": video_path}, return_video_sample_fps=True) # TODO: sampling according to the argument [cbzhang]
            # Prepare the observation
            obs = {
                "prompt": "\n<|im_start|>user\n" +"<tool_response><video></tool_response>" + "<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"video": [video_data], "fps": torch.tensor([fps])}
            }
            reward = 0.1  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name}
            print(f'[DEBUG] SUCCESS ACTION {action_string=}')
            return obs, reward, done, info
        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[DEBUG] Execute WRONG - {str(e)} {action_string=}')
            obs = "\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        if 'agent_clip_frames' in kwargs:
            self.reading_frame_cnt = kwargs['agent_clip_frames']
        else:
            self.reading_frame_cnt = multi_modal_data['video'][0].shape[0]
        self.resized_height = multi_modal_data['video'][0].shape[-2]
        self.resized_width = multi_modal_data['video'][0].shape[-1]
        assert 'video' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['video']) > 0, f'[ERROR] {self.multi_modal_data["video"]=}'
        self.video_path = self.multi_modal_data["video"][0]["video"]
        vr = decord.VideoReader(self.video_path)
        self.video_length = len(vr) / vr.get_avg_fps()

    def validate_interval(self, start_time: float, end_time: float) -> bool:
        try:
            assert start_time < end_time, f"start_time should be less than end_time: {start_time=} {end_time=}"

            time_interval = end_time - start_time
            assert time_interval > 3, f"time interval should be greater than 1 second: {start_time=} {end_time=}"
            return True
        except Exception as err:
            print(f' [ERROR vl_agent #2] {err=}')
            return False


    def maybe_resize_interval(self, start_time: float, end_time: float):
        start_time = max(0, start_time)
        end_time = min(self.video_length, end_time)

        if not self.validate_interval(start_time, end_time):
            return None

        return [start_time, end_time]
        


    


if __name__ == "__main__":
    # Example usage (for testing)
    tool = VideoToolV1Reflection("video_tool_v1", "Tool for video zooming", {})
    tool.video_length = 100
    tool.video_path = "/workspace/images-ks3-starfs-hd/workspace/lijiaze/projects/DeepEyes-video-xiaomi/coc_data/tmp__0oxli9cTCA_v.1280x720.mp4"
    tool.reading_frame_cnt = 16
    # Test zoom in tool (should return reward=0.1)
    zoom_in_action = """
    <tool_call>
    {"name": "image_zoom_in_tool", "arguments": {"image_path": "test.jpg", "bbox": [10, 10, 100, 100]}}
    </tool_call>
    """
    zoom_in_action = """<tool_call>{"name":"video_zoom_in_tool", "arguments":{"interval":"0.1 to 1000"}}</tool_call>"""
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}")
    exit(0)
    # Test rotate tool (should return reward=0.1)
    rotate_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"image_path": "test.jpg", "angle": 90}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(rotate_action)
    print(f"Rotate result - Reward: {reward}, Info: {info}")
    
    # Test invalid JSON (should return reward=0.0)
    invalid_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"image_path": "test.jpg", "angle": 90}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(invalid_action)
    print(f"Invalid JSON result - Reward: {reward}, Info: {info}")
    
    # Test unknown tool (should return reward=0.0)
    unknown_tool_action = """
    <tool_call>
    {"name": "unknown_tool", "arguments": {"param": "value"}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(unknown_tool_action)
    print(f"Unknown tool result - Reward: {reward}, Info: {info}")
