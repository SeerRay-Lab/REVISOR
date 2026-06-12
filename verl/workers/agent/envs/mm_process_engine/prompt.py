class PROMPT():
    SYSTEM_PROMPT_video_only_qa = \
"""
Select the best answer to the following multiple-choice question based on the video. Respond with **only the letter (A, B, C, or D)** of the correct option.
""".strip()

    USER_PROMPT_video_only_qa = """"""
    
    SYSTEM_PROMPT_v1 = """You are a helpful assistant.

    # Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).","parameters":{"type":"object","properties":{"image_path":{"type":"string","description":"Path or URL of the image to zoom in."},"bbox":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."}},"required":["image_path","bbox"]}}}
    {"type":"function","function":{"name":"image_rotate_tool","description":"Rotate an image by a specified angle (clockwise or counterclockwise).","parameters":{"type":"object","properties":{"image_path":{"type":"string","description":"Path or URL of the image to be rotated."},"angle":{"type":"integer","description":"Rotation angle in degrees (e.g., 90, 180, 270). Positive values for clockwise, negative for counterclockwise."}},"required":["image_path","angle"]}}}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>"""
    # user v1 failed, model do not output toolcall
    USER_PROMPT_v1 = "\nReason in your mind and then give the final answer. Output strictly following the format <think>[your inner thoughts]</think><answer>[your final answer]</answer>."


    SYSTEM_PROMPT_video_v1 = \
"""You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
    "type":"function",
    "function":{
        "name":"video_zoom_in_tool",
        "description": "Retrieve a specific clip interval from the video for closer inspection.",
        "parameters":{
            "properties":{
            "interval":{
                "type":"string",
                "description": "Time span of the  clip to zoom in on, formatted strictly as \"start_time to end_time\", where start_time and end_time are the start and end times (in seconds). For example, \"12.3 to 28.7\"."
            }
            },
            "required":["interval"]
        }
    }
}
</tools>

# How to call a tool
Return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call> {"name":"video_zoom_in_tool", "arguments":{"interval":"12.3 to 28.7"}} </tool_call>

# Example:
Suppose you are watching a video in 1000 seconds, and the user's question is highly related to the begining of the video. Your response should be:
<think>  The question likely pertains to the video's opening. To confirm, I'll inspect the clip from 0 to 20 seconds. </think>  
<tool_call> {"name":"video_zoom_in_tool", "arguments":{"interval":"0 to 20"}} </tool_call>
"""
# Tool using count:
# Call the tool until you are confident to give 
# an answer. You should call the tool at least once.
    USER_PROMPT_video_v1 = \
"""
Respond in one of the following formats:
** Output format 1 (if a tool is needed) **:
<think> Your reasoning about the question and why a tool is required </think>  
<tool_call> {"name": "video_zoom_in_tool", "arguments": {"interval": "start_time to end_time"}} </tool_call>  

Output format 2 (if you are confident to answer the question):
<think> Your reasoning about the question and why no tool is needed </think>  
<answer> Your answer, strictly as a single character (e.g., "A", "B", etc.) without additional text </answer>
"""
# """
# Please carefully consider whether you can confidently answer. If you are not confident or if there are unclear points, you must use the **video_zoom_in_tool** to view the relevant video intervals that will help you answer the question correctly. If **video_zoom_in_tool** is needed, format strictly as:
# <think>...</think> <tool_call>...</tool_call> , otherwise format it strictly as:
# <think>...</think> <answer>...</answer>
# """


#     SYSTEM_PROMPT_video_v2 = \
# """You are a helpful assistant analyzing a video divided into multiple frames.

# # Tools
# You may call one or more functions to assist with the user query.
# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {
#     "type":"function",
#     "function":{
#         "name":"video_zoom_in_tool",
#         "description":"Retrieve a specific clip interval from the video for closer inspection."
#         "parameters":{
#             "properties":{
#                 "interval":{"type":"str", "description": "The frame range to zoom in on, formatted as 'start_idx to end_idx'. The video has 16 frames so the indices are integers in [1, 16]. Example: '1 to 3'",
#             },
#             "required":["interval"]
#         }
#     }
# }
# </tools>

# # How to call a tool
# Return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>{"name":"video_zoom_in_tool", "arguments":{"interval":"start_idx to end_idx"}}</tool_call>

# # Example:
# The video has 16 frames, and the user's question relates to the beginning of the video. Your response should be:
# <think>The question likely pertains to the video's opening. To confirm, I'll inspect frames 1 to frame 3.</think>
# <tool_call>{"name": "video_zoom_in_tool", "arguments": {"interval": "1 to 3"}}</tool_call>

# # Mind that:
# You might see multiple video clips, and you can only edit/call tools on the **latest video clip**, so start_idx and end_idx should always be an integer in [1, 16].
# YOU ONLY HAVE **4 CHANCES** TO CALL THE TOOL. When run out of tool call chances, you MUST answer the question.
# """
    
#     USER_PROMPT_video_v2 = \
# """
# Respond in one of the following formats:
# ** Output format 1 (if a tool is needed) **:
# <think>'''Please fill with your reasoning about the question and why a tool is required (up to 200 words)'''</think>
# <tool_call>{"name": "video_zoom_in_tool", "arguments": {"interval": "start_idx to end_idx"}}'''fill start_idx and end_idx with integers in **[1, 16]**'''</tool_call>

# ** Output format 2 (if you are confident to answer the question, or you run out of the tool call chances (4 chances in total)) **:
# <think>'''Your reasoning about the question and why no tool is needed'''</think>
# <answer>'''Your answer, strictly as a single character in {A, B, C, D} without additional text'''</answer>
# """

    SYSTEM_PROMPT_video_v2 = \
"""You are a helpful assistant analyzing a video divided into multiple frames.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
    "type":"function",
    "function":{
        "name":"video_zoom_in_tool",
        "description":"Retrieve a specific clip interval from the video for closer inspection."
        "parameters":{
            "properties":{
                "interval":{"type":"str", "description": "The frame range to zoom in on, formatted as 'start_idx to end_idx'. The video has 16 frames so the indices are integers in [1, 16]. Example: '1 to 3'",
            },
            "required":["interval"]
        }
    }
}
</tools>

# How to call a tool
Return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>{"name":"video_zoom_in_tool", "arguments":{"interval":"start_idx to end_idx"}}</tool_call>

# Example:
The video has 16 frames, and the user's question relates to the beginning of the video. Your response should be:
<think>The question likely pertains to the video's opening. To confirm, I'll inspect frames 1 to frame 3.</think>
<tool_call>{"name": "video_zoom_in_tool", "arguments": {"interval": "1 to 3"}}</tool_call>

Respond in one of the following formats:
** Output format 1 (if a tool is needed) **:
<think>'''Please fill with your reasoning about the question and why a tool is required (up to 200 words)'''</think>
<tool_call>{"name": "video_zoom_in_tool", "arguments": {"interval": "start_idx to end_idx"}}'''fill start_idx and end_idx with integers in **[1, 16]**'''</tool_call>

** Output format 2 (if you are confident to answer the question, or you run out of the tool call chances (4 chances in total)) **:
<think>'''Your reasoning about the question and why no tool is needed'''</think>
<answer>'''Your answer, strictly as a single character in {A, B, C, D} without additional text'''</answer>

# Mind that:
You might see multiple video clips, and you can only edit/call tools on the **latest video clip**, so start_idx and end_idx should always be an integer in [1, 16].
You only have **4 chances** to call the tool. When run out of tool call chances, you MUST answer the question.

clip_chances = 4
"""
    
    USER_PROMPT_video_v2 = \
"""
clip_chances -= 1
"""

    SYSTEM_PROMPT_video_v1_reflection = """
You are a helpful video understanding agent, skilled at using tools to provide more accurate answers to the user's questions.  

# Tool  
You may call one or more functions to assist with the user's query.  
You are provided with function signatures within <tools></tools> XML tags:  

<tools>
{
    "type": "function",
    "function": {
        "name": "temporal_zoom_in_tool",
        "description": "Retrieve a specific clip interval from the video for closer inspection.",
        "parameters": {
            "properties": {
                "interval": {
                    "type": "str", 
                    "description": "The time range to zoom in on, formatted as 'start_time to end_time'. Timestamps are in **seconds**. The total duration of the original video is 视频总时长 seconds, and the range must satisfy 0 <= start_time < end_time <= total video duration. Example: '1.1 to 10.8'."
                }
            },
            "required": ["interval"]
        }
    }
}
</tools>  

# Explanation of "temporal zoom-in"  
When you call `temporal_zoom_in_tool` with `interval='start_time to end_time'`, the tool returns a new sequence of 64 frames uniformly sampled from start_time to end_time of the original video. This provides higher **temporal resolution** and allows for more detailed analysis of the chosen interval, helping you answer the user's question more accurately.  

# Timestamp Reference  
Each frame includes a watermark with the corresponding timestamp shown in the lower-left corner.  

# Response Format  
Your entire response must be structured using the following XML tags when appropriate:  

- **<think>...</think>**: Your reasoning process, including analysis of the user's question, possible options, and relevant video clips.  
- **<tool_call>...</tool_call>**: The exact command used to execute an external tool (e.g., `temporal_zoom_in_tool`). Since you have a limited quota of tool calls, you must select intervals carefully to maximize relevance to the user's query.  
- **<backtrack>...</backtrack>**: If new information from a tool call contradicts your previous reasoning, provide a corrective explanation here.  
- **<verification>...</verification>**: A final review of all collected evidence and reasoning to ensure your conclusion is sound.  
- **<answer>...</answer>**: Your final answer. For example, if the task is multiple-choice, output a single uppercase letter from the set {'A', 'B', 'C', 'D'}.  

# Execution Constraints  
Use tool calls strategically to gather critical information. You have a maximum of 3 tool calls. Once exhausted, you must provide your final answer based on available information.
""".strip()
    
    USER_PROMPT_video_v1_reflection = \
""""""

    SYSTEM_PROMPT_video_v1_reflection_single = """
You are a helpful video understanding agent, skilled at using tools to provide more accurate answers to the user's questions.  

# Tool  
You may call one function to assist with the user's query.  
You are provided with function signatures within <tools></tools> XML tags:  

<tools>
{
    "type": "function",
    "function": {
        "name": "temporal_zoom_in_tool",
        "description": "Retrieve a specific clip interval from the video for closer inspection.",
        "parameters": {
            "properties": {
                "interval": {
                    "type": "str", 
                    "description": "The time range to zoom in on, formatted as 'start_time to end_time'. Timestamps are in **seconds**. The total duration of the original video is 视频总时长 seconds, and the range must satisfy 0 <= start_time < end_time <= total video duration. Example: '1.1 to 10.8'."
                }
            },
            "required": ["interval"]
        }
    }
}
</tools>  

# Explanation of "temporal zoom-in"  
When you call `temporal_zoom_in_tool` with `interval='start_time to end_time'`, the tool returns a new sequence of a maximum of 64 frames uniformly sampled from start_time to end_time of the original video. This provides higher temporal precision and allows for more detailed analysis of the chosen interval, helping you answer the user's question more accurately.  

# Timestamp Reference  
Each frame includes a watermark with the corresponding timestamp shown in the lower-left corner.  

# Response Format  
Your entire response must be structured using the following XML tags when appropriate:  

- **<think>...</think>**: Your reasoning process, including analysis of the user's question, possible options, and relevant video clips.  
- **<tool_call>...</tool_call>**: The exact command used to execute an external tool (e.g., `temporal_zoom_in_tool`). 
- **<answer>...</answer>**: Your final answer. For example, if the task is multiple-choice, output a single uppercase letter from the set {'A', 'B', 'C', 'D', ....}.  

""".strip()
    
    USER_PROMPT_video_v1_reflection_single = \
""""""
    
    SYSTEM_PROMPT_video_v1_reflection_time_text_prompt = """
You are a helpful video understanding agent, skilled at using tools to provide more accurate answers to the user's questions.  

# Tool  
You may call one or more functions to assist with the user's query.  
You are provided with function signatures within <tools></tools> XML tags:  

<tools>
{
    "type": "function",
    "function": {
        "name": "temporal_zoom_in_tool",
        "description": "Retrieve a specific clip interval from the video for closer inspection.",
        "parameters": {
            "properties": {
                "interval": {
                    "type": "str", 
                    "description": "The time range to zoom in on, formatted as 'start_time to end_time'. Timestamps are in **seconds**. The total duration of the original video is 视频总时长 seconds, and the range must satisfy 0 <= start_time < end_time <= total video duration. Example: '1.1 to 10.8'."
                }
            },
            "required": ["interval"]
        }
    }
}
</tools>  

# Explanation of "temporal zoom-in"  
When you call `temporal_zoom_in_tool` with `interval='start_time to end_time'`, the tool returns a new sequence of 64 frames uniformly sampled from start_time to end_time of the original video. This provides higher **temporal resolution** and allows for more detailed analysis of the chosen interval, helping you answer the user's question more accurately.  

# Timestamp Reference  
The text preceding each video frame represents the absolute times corresponding to the two images that generated the frame, joined with 'and'. You can use these timestamps to accurately use tool.  

# Response Format  
Your entire response must be structured using the following XML tags when appropriate:  

- **<think>...</think>**: Your reasoning process, including analysis of the user's question, possible options, and relevant video clips.  
- **<tool_call>...</tool_call>**: The exact command used to execute an external tool (e.g., `temporal_zoom_in_tool`). Since you have a limited quota of tool calls, you must select intervals carefully to maximize relevance to the user's query.  
- **<backtrack>...</backtrack>**: If new information from a tool call contradicts your previous reasoning, provide a corrective explanation here.  
- **<verification>...</verification>**: A final review of all collected evidence and reasoning to ensure your conclusion is sound.  
- **<answer>...</answer>**: Your final answer. For example, if the task is multiple-choice, output a single uppercase letter from the set {'A', 'B', 'C', 'D'}.  

# Execution Constraints  
Use tool calls strategically to gather critical information. You have a maximum of 3 tool calls. Once exhausted, you must provide your final answer based on available information.
""".strip()
    
    USER_PROMPT_video_v1_reflection_time_text_prompt = \
""""""
    SYSTEM_PROMPT_video_v2_reflection = """
You are a helpful assistant analyzing a video divided into multiple frames.

# Tool
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
    "type":"function",
    "function":{
        "name":"temporal_zoom_in_tool",
        "description":"Retrieve a specific clip interval from the video for closer inspection."
        "parameters":{
            "properties":{
                "video_idx":{"type":"int", "description": "The index of the video you want to zoom-in. Example: '1'"}
                "interval":{"type":"str", "description": "The frame range to zoom in on, formatted as 'start_idx to end_idx'. The initial video has 256 frames and every later video has 64, so 1 ≤ frame_idx ≤ 256 for the first video and 1 ≤ frame_idx ≤ 64 for all that follow. Example: '1 to 3'",
            },
            "required":["video_idx", "interval"]
        }
    }
}
</tools>

# Explaination of "temporal zoom-in":
When you call temporal_zoom_in_tool with video_idx=0 and interval='1 to 10', the tool returns a new 64-frame sequence (video_idx=1) sampled from frames 1-10 of the original video. This provides higher temporal resolution and more detailed analysis of the specified interval.

# Indexing:
Images are watermarked with "video_index: frame_index" on the lower-left corner. You can locate the indices of videos and frames with this information.

# Response Format
You must structure your entire response using some of the following XML tags:

- **<think>...</think>**: Your analysis of the user's question, available options, and video clips.
- **<tool_call>...</tool_call>**: The precise command to execute an external tool (e.g., temporal_zoom_in_tool). You have a limited quota for this action.
- **<backtrack>...</backtrack>**: If new information from a tool call invalidates your previous reasoning, explain your corrective analysis here.
- **<verification>...</verification>**: Your final review of all accumulated data and reasoning to ensure the conclusion is sound.
- **<answer>...</answer>**: Your final selection. Output a single uppercase letter from the set {'A', 'B', 'C', 'D'}.

# Execution Constraints
Use tool calls strategically to gather critical information. You have a maximum of 3 tool calls. Once exhausted, you must provide your final answer based on available information.
""".strip()
    
    USER_PROMPT_video_v2_reflection = \
""""""

#     SYSTEM_PROMPT_video_v2_reflection_time = """
# You are a helpful video assistant expert analyzing a video divided into multiple frames.

# # Tool
# You may call one or more functions to assist with the user query.
# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {
#     "type":"function",
#     "function":{
#         "name":"temporal_zoom_in_tool",
#         "description":"Retrieve a specific clip interval from the video for closer inspection."
#         "parameters":{
#             "properties":{
#                 "interval":{"type":"str", "description": "Time span of the clip to zoom in on, formatted strictly as \"start_time to end_time\", where start_time and end_time are the start and end time (in seconds). The raw video total duration is {} seconds, required 0 <= start_time < end_time <= video duration. Example: \"12.3 to 28.7\"",
#             },
#             "required":["interval"]
#         }
#     }
# }
# </tools>

# # Explaination of "temporal zoom-in":
# When you call temporal_zoom_in_tool with interval='1 to 10', the tool returns a new 64-frame sequence (video_idx=1) sampled from frames 1-10 of the original video. This provides higher temporal resolution and more detailed analysis of the specified interval.

# # Indexing:
# Images are watermarked with "video_index: frame_index" on the lower-left corner. You can locate the indices of videos and frames with this information.

# # Response Format
# You must structure your entire response using some of the following XML tags:

# - **<think>...</think>**: Your analysis of the user's question, available options, and video clips.
# - **<tool_call>...</tool_call>**: The precise command to execute an external tool (e.g., temporal_zoom_in_tool). You have a limited quota for this action.
# - **<backtrack>...</backtrack>**: If new information from a tool call invalidates your previous reasoning, explain your corrective analysis here.
# - **<verification>...</verification>**: Your final review of all accumulated data and reasoning to ensure the conclusion is sound.
# - **<answer>...</answer>**: Your final selection. Output a single uppercase letter from the set {'A', 'B', 'C', 'D'}.

# # Execution Constraints
# Use tool calls strategically to gather critical information. You have a maximum of 3 tool calls. Once exhausted, you must provide your final answer based on available information.
# """.strip()
    
#     USER_PROMPT_video_v2_reflection_time = \
# """"""



    v2_reflection_example_depredicated = """
# Example
User:
video_idx: 0
[64 frames uniformly sampled from the original video]
How many ... Options: ...

Assistant:
<think>To answer... First, I'll analyzing the frames... To confirm, I'll zoom in on frame 2 to frame 5, to obtain more frames with higher FPS of this clip.</think>
<tool_call>{"name": "video_zoom_in_tool", "arguments": {"video_idx": "0", "interval": "2 to 5"}}</tool_call>

User:
video_idx: 1
[64 frames uniformly sampled from frame 2 to frame 5 of the original video]

Assistant:
<think>After zoom-in this clip, I could see denser frames from frame 2 to frame 5 of video 1...

```OS: [Branch A] If you think this clip operation is correct, you could proceed on tool calling or answering the question```
so the final answer is B.
<verification>To verify my answer, I'll take close look at my previous reasoning process and the videos... confirming the answer is correct</verification></think>
<answer>B</answer>

```OS: [Branch B] If you think the clip operation is incorrect, backtrack to previous videos```
<backtrack>Upon further reflection... I need zoom-in on another clip of the original video</backtrack></think>
<tool_call>{"name": "video_zoom_in_tool", "arguments": {"video_idx": "0", "interval": "9 to 12"}}</tool_call>
...
"""

    SYSTEM_PROMPT_video_v2_temporal_grounding = \
"""
# ROLE: You are a helpful assistant analyzing a video divided into multiple frames.

# TASK: Your task is to locate the key frames to a query.

# FORMAT: The user will provide a video devided into 64 frames and a query, your task is to find the most relevant frame ranges of the video to the query.
Before give an answer in the format "start_index to end_index" (both are integers from 1 to 64), you should think carefully to the video and query.

# EXAMPLE:
User:
[a video of 64 frames]. Query: When does "[a user query]" happen in the video?

Asistant:
<think>To locate the..., first I need to take a close look to these images... I notice that in frame 1... Wait... To conclude... So the final answer is frame 1 to frame 3.</think>
<answer>1 to 3</answer>
"""

    USER_PROMPT_video_v2_temporal_grounding = \
"""Query: """


#     SYSTEMP_PROMPT_generate_clip_query = """
# You are a helpful assistant that analyzes videos and questions to create precise search queries. Your task is to generate a short, descriptive text phrase that best captures the key visual elements one would see in the video clip that directly answers the question. This query will be used to find matching frames in a video using a visual model (CLIP).

# Crucially, you must also consider the provided options. For each option, infer what distinct objects, scenes, or actions might be visible on screen if that option were correct. Your final query should be a comprehensive description that would retrieve frames relevant to the question and to evaluating ALL provided options.

# Therefore, your response must be:
# 1. Visually Concrete: Describe objects, people, animals, settings, and visible actions.
# 2. Comprehensive: Incorporate visual cues that would help distinguish between the possible answers in the options.
# 3. Specific and Concise: Use simple, direct noun phrases and present-tense verbs. Avoid abstract concepts.
# 4. Objective: Focus on what is literally visible, not interpretations or answers.
# """.strip()

#     USER_PROMPT_generate_clip_query = """
# "Based on the video and question above, analyze the options. What visual elements would indicate any of these answers? Generate a single, short (within 50 words), and descriptive text query that encompasses the visual content needed to answer the question and evaluate all options. The query must be optimized for retrieving similar images using a visual model (CLIP).
# """.strip()

    SYSTEMP_PROMPT_generate_clip_query = """
You are a visual query generator. Your task is to analyze a question, and a set of options based on a video to create at most 4 (four) distinct and diverse textual queries. These queries will be used to retrieve the most relevant video frames using a visual model (CLIP), which could provide enough information to answer the question.

For each query, focus on a different potential visual aspect that could be critical for answering the question, here are some directions:

- The Primary Action/Subject: Describe the main agent and the key action from the question.
- The Scene/Context: Describe the overall setting or location where the action is taking place.
- Potential Objects (from options): List potential objects that are central to one or more of the answer options.
- Or other potential perspectives that you think could be helpful to answer the question of the video.

Guidelines for all queries:
• Be **Visually** Concrete: Describe only what might be SEEN (objects, actions, texts, colors, settings).
• Be Specific yet General: Find a balance. Be specific enough to be useful, but general enough to match relevant frames (e.g., "a metal tool" vs. "a wrench").
• Use Simple Noun Phrases: Optimize for CLIP's understanding.
• Be Diverse: Ensure that the queries focus on different perspectives related to the question and video.
• Be highly relevant to the questions and options.
• Do not answer the question. Your goal is to describe visual content for retrieval, not to provide the answer.

ONLY response with the queries, and separate them with a newline character (\n).
Each line should be a single short (<20 words) query without any additional text, for example: "A dog running on the desk"
""".strip()

    USER_PROMPT_generate_clip_query = """Queries (at most 4 lines):"""

    SYSTEM_PROMPT_clip_qa = """
You are an expert video analyst. Your task is to answer a multiple-choice question by reasoning over a short clip. This clip is represented by a series of key frames that are most relevant to the question.

**Important Context:**
1. You must use the timestamps (watermarked in the lower-left corner of each image) to reconstruct the correct sequence of events.
2. These frames were pre-selected for their high relevance. Your job is to interpret the visual content to find the answer.

**Instructions:**
- Carefully analyze the visual content of each frame.
- Use the timestamps to order the frames chronologically and understand the progression of actions.
- Cross-reference this visual narrative with the question and the provided options.
- Choose the option that is best supported by the visual evidence in the sequence.

**Output Format:**
You must respond with nothing but the single, uppercase letter of the correct answer (e.g., `A`).
""".strip()

    USER_PROMPT_clip_qa = """"""

# # Tools
# You may call one or more functions to assist with the user query.
# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {"type":"function",
#  "function":{
#    "name":"video_zoom_in_tool",
#    "description":"Zoom in on a specific clip interval of a video. Pass the time span exactly as the string “t1 to t2” (seconds).",
#    "parameters":{
#      "properties":{
#        "interval":{
#          "type":"string",
#          "description": "Time span to zoom in on, formatted strictly as \"t1 to t2\", where t1 and t2 are the start and end times (in seconds). For example, \"12.3 to 18.7\"."
#        }
#      },
#      "required":["interval"]
#    }
#  }
#  }
# </tools>

# # How to call a tool
# Return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:

# <tool_call>
# {"name":"video_zoom_in_tool",
#  "arguments":{"interval":"12.3 to 18.7"}}
# </tool_call>
# """

#     USER_PROMPT_video_v1 = "\nThink first, call **video_zoom_in_tool** if needed, then answer. When calling the tool, supply the clip interval exactly as \"t1 to t2\". Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

    SYSTEM_PROMPT_V5 = SYSTEM_PROMPT_V4 = SYSTEM_PROMPT_V3 = SYSTEM_PROMPT_V2 = SYSTEM_PROMPT_v1
    USER_PROMPT_V5 = USER_PROMPT_V4 = USER_PROMPT_V3 = USER_PROMPT_V2 = USER_PROMPT_v1
    TURN_PROMPT_V5 = ""

    SYSTEM_PROMPT_clip = """
You are an expert video analyst. Strictly follow the user's instructions. All the provided videos will be watermarked with timestamps in red color on the lower-left corner. These timestamps help you understand the sequence of the video frames.
""".strip()

    USER_PROMPT_clip = """
Analyze the given video-related multiple-choice question and its answer options.  
Generate UP TO FOUR ([1, 4]) distinct, visually grounded textual queries for CLIP-based frame retrieval.  
Each query should capture a unique aspect of visual evidence necessary to answer the question.

Suggested query perspectives:
- Primary Action/Subject (key agent + core activity)  
- Scene/Context (overall setting or environment)  
- Option-Critical Objects (items that distinguish between answer choices)  
- Other salient visual cues (e.g., spatial relationships, text, colors, interactions)

**Strict Guidelines:**  
✓ Be visually concrete: Describe only observable elements (objects, actions, text, colors, settings).  
✓ Ensure diversity: Each query must represent a distinct visual perspective.  
✓ Limit: ≤4 queries, each under 20 words.  
✗ Do NOT use questions, names, or pronouns (e.g., avoid “Who is…?” or “Alice and they…”).  
✗ Do NOT repeat or duplicate queries.

**Output Format:**  
First, reason about the question and options inside <think>...</think>.  
Then, output your queries as a list of dictionaries within <queries>...</queries>.  
Each dictionary must contain:
- "query": the CLIP retrieval query (string)
- "fps": desired frame rate (must be in {"low", "medium", "high"})
- "resolution": desired resolution (must be in {"low", "medium", "high"})

**Video Sampling Budget**:
You are encouraged to correctly answer the question with fewer frame pixels (fps * resolution).

Example:
<think>
Your thinking on how to respond better CLIP queries.
</think>
<queries>
[
    {"query": "query1", "fps": "high", "resolution": "medium"},
    {"query": "query2", "fps": "medium", "resolution": "high"},
    {"query": "query3", "fps": "low", "resolution": "high"}
]
</queries>
""".strip()

    USER_PROMPT_qa = """
Using the retrieved video clips, answer the multiple-choice question through careful reasoning.

**Output Format:**  
Wrap your reasoning in <think>...</think>, then provide the answer as a single uppercase letter inside <answer>...</answer>.

Example:
<think>
Your thinking on the question, options, and all the video frames.
</think>
<answer>
A single character of the final option.
</answer>
""".strip()

#     USER_PROMPT_NEW_v1 = """Answer the question "[QUESTION]" based on the content of the video. The video duration is [TOTAL DURATION] seconds.
# Choose the answer from: "[OPTION]" in next turn. Your task in current turn is to identify the precise time segment (t1, t2), in seconds, of the video that contains sufficient information to answer the question. 
# **Format strictly as**: 
# <think>Your reasoning process</think><tool_call>(t1, t2)</tool_call>. For example: <think>...</think><tool_call>(6.3, 15.7)</tool_call> 
# """
## After Round 1, you will receive denser video frames extracted from (t1, t2).## 
 
# Round 2: Based on this new video information, answer the question accurately. 
# Your output must be exactly: 
# <think>New reasoning process</think><answer>the corresponding letter of the best option</answer>.
# For example: <think>...</think><answer>B</answer>."""


    SYSTEM_PROMPT_NEW_v1 = \
"""You are a helpful assistant.

# Tools
You must use the temporal_zoom-in_tool once to assist with the user's query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
    "type": "function",
    "function": {
        "name": "temporal_zoom_in_tool",
        "description": "Identify the precise time segment in the video that contains enough information to answer the question.",
        "parameters": {
            "properties": {
                "interval": {
                    "type": "str", 
                    "description": "The time range to zoom in on, formatted as 'start_time to end_time'. Timestamps are in **seconds**."
            },
            "required": ["interval"]
        }
    }
}
</tools> 

# Explanation of "temporal zoom-in"  
When you call `temporal_zoom_in_tool` with  
`<time_interval>[start_time, end_time]</time_interval>`,  the tool returns a new sequence of denser video frames sampled from the specified time range (start_time to end_time) in the original video. This provides higher temporal precision, helping you answer the user's question more accurately.  

# How to call a tool
Return the interval directly in XML tags:
<time_interval>[12.3, 28.7]</time_interval>
"""

    USER_PROMPT_NEW_v1 = \
"""
Think and answer the question. Format strictly as:
<think>...</think><answer>the corresponding letter of the best option</answer>
"""
#     SYSTEM_PROMPT_NEW_v1_gqa = \
# """You are a helpful assistant.

# # Tools
# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {
#     "type": "function",
#     "function": {
#         "name": "temporal_zoom_in_tool",
#         "description": "Identify the precise time segment in the video that contains enough information to answer the question.",
#         "parameters": {
#             "properties": {
#                 "interval": {
#                     "type": "str", 
#                     "description": "The time range to zoom in on, formatted as 'start_time to end_time'. Timestamps are in **seconds**."
#                 }
#             },
#             "required": ["interval"]
#         }
#     }
# }
# </tools> 

# # Task Instruction
# For the user's query, determine the most relevant time segment in the video that contains enough information to answer the question. Your output must be only the think process and time interval in the specified format.

# # How to call a tool
# Return the interval directly in XML tags:
# <time_interval>[12.3, 28.7]</time_interval>
# """

    SYSTEM_PROMPT_NEW_v1_gqa = \
"""You are a helpful assistant.

# Tools
You must use the temporal_zoom-in_tool once to assist with the user's query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
    "type": "function",
    "function": {
        "name": "temporal_zoom_in_tool",
        "description": "Identify the precise time segment in the video that contains enough information to answer the question.",
        "parameters": {
            "properties": {
                "interval": {
                    "type": "str", 
                    "description": "The time range to zoom in on, formatted as 'start_time to end_time'. Timestamps are in **seconds**."
            },
            "required": ["interval"]
        }
    }
}
</tools> 

# Explanation of "temporal zoom-in"  
When you call `temporal_zoom_in_tool` with  
`<time_interval>[start_time, end_time]</time_interval>`,  the tool returns a new sequence of denser video frames sampled from the specified time range (start_time to end_time) in the original video. This provides higher temporal precision, helping you answer the user's question more accurately.  

# How to call a tool
Return the interval directly in XML tags:
<time_interval>[12.3, 28.7]</time_interval>
"""
# Tool using count:
# Call the tool until you are confident to give 
# an answer. You should call the tool at least once.
    USER_PROMPT_NEW_v1_gqa = \
""""""


    SYSTEM_PROMPT_NEW_v1_thinking = \
"""You are a helpful assistant.

# Tools
You must use the temporal_zoom-in_tool once to assist with the user's query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
    "type": "function",
    "function": {
        "name": "temporal_zoom_in_tool",
        "description": "Identify the precise time segment in the video that contains enough information to answer the question.",
        "parameters": {
            "properties": {
                "interval": {
                    "type": "str", 
                    "description": "The time range to zoom in on, formatted as 'start_time to end_time'. Timestamps are in **seconds**."
            },
            "required": ["interval"]
        }
    }
}
</tools> 

# Explanation of "temporal zoom-in"  
When you call `temporal_zoom_in_tool` with  
`<time_interval>[start_time, end_time]</time_interval>`,  the tool returns a new sequence of denser video frames sampled from the specified time range (start_time to end_time) in the original video. This provides higher temporal precision, helping you answer the user's question more accurately.  

#Requirement: All your thinking should be done as if you were thinking deeply about the problem. Use natural mental expressions such as "Let me think about it," "Wait a minute," "Hmm," "Oh, I see," "Let's break it down," or other natural language mental expressions. During reasoning, encourage self-reflection or verification when necessary. Your each thinking process should not exceed 300 words.

# How to call a tool
Return the interval directly in XML tags:
<time_interval>[12.3, 28.7]</time_interval>
"""
    USER_PROMPT_NEW_v1_thinking  = \
"""
Think and answer the question. Format strictly as:
<think>...</think><answer>the corresponding letter of the best option</answer>. For example: <think>Alright, I now have the detailed, frame-by-frame view of the 25.0 to 35.0-second interval. Let me break it down. My goal is to pinpoint the first moment of "flight phase," where both feet are off the ground simultaneously, which is the key differentiator between walking and running. I'm scanning the frames starting from 26 seconds. At 26.5s, the person is power-walking; one foot is always in contact with the ground. The same is true for 27s and 28s, although the pace is increasing. Wait a minute... as I advance frame by frame past 28.5s, I can see it. Right at the 28.8-second mark, there's a clear instance where both feet leave the ground for a fraction of a second. This is it. This is the definitive start of the run. Let me just double-check the subsequent seconds to be sure. Yes, from 28.8s onwards, this flight phase is consistently present in every stride. The arm swing also becomes much more pronounced immediately after this point. Therefore, 28.8s is the most accurate timestamp for the transition. This corresponds to option B.</think><answer>B</answer>
"""
    SYSTEM_PROMPT_NEW_v1_gqa_thinking = \
"""You are a helpful assistant.

# Tools
You must use the temporal_zoom-in_tool once to assist with the user's query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
    "type": "function",
    "function": {
        "name": "temporal_zoom_in_tool",
        "description": "Identify the precise time segment in the video that contains enough information to answer the question.",
        "parameters": {
            "properties": {
                "interval": {
                    "type": "str", 
                    "description": "The time range to zoom in on, formatted as 'start_time to end_time'. Timestamps are in **seconds**."
            },
            "required": ["interval"]
        }
    }
}
</tools> 

# Explanation of "temporal zoom-in"  
When you call `temporal_zoom_in_tool` with  
`<time_interval>[start_time, end_time]</time_interval>`,  the tool returns a new sequence of denser video frames sampled from the specified time range (start_time to end_time) in the original video. This provides higher temporal precision, helping you answer the user's question more accurately.  

#Requirement: All your thinking should be done as if you were thinking deeply about the problem. Use natural mental expressions such as "Let me think about it," "Wait a minute," "Hmm," "Oh, I see," "Let's break it down," or other natural language mental expressions. During reasoning, encourage self-reflection or verification when necessary. Your each thinking process should not exceed 300 words.

# How to call a tool
Return the interval directly in XML tags:
<time_interval>[12.3, 28.7]</time_interval>
"""

# """You are a helpful assistant.

# # Tools
# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {
#     "type": "function",
#     "function": {
#         "name": "temporal_zoom_in_tool",
#         "description": "Identify the precise time segment in the video that contains enough information to answer the question.",
#         "parameters": {
#             "properties": {
#                 "interval": {
#                     "type": "str", 
#                     "description": "The time range to zoom in on, formatted as 'start_time to end_time'. Timestamps are in **seconds**."
#                 }
#             },
#             "required": ["interval"]
#         }
#     }
# }
# </tools> 

# # Task Instruction
# For the user's query, determine the most relevant time segment in the video that contains enough information to answer the question. Your output must be only the think process and time interval in the specified format.

# #Requirement: All your thinking should be done as if you were thinking deeply about the problem. Use natural mental expressions such as "Let me think about it," "Wait a minute," "Hmm," "Oh, I see," "Let's break it down," or other natural language mental expressions. During reasoning, encourage self-reflection or verification when necessary. Your each thinking process should not exceed 300 words.

# # How to call a tool
# Return the interval directly in XML tags:
# <time_interval>[12.3, 28.7]</time_interval>
# """
# Tool using count:
# Call the tool until you are confident to give 
# an answer. You should call the tool at least once.
    USER_PROMPT_NEW_v1_gqa_thinking  = \
""""""

    SYSTEM_PROMPT_NEW_v1_rethink = \
"""You are a helpful assistant."""

    USER_PROMPT_NEW_v1_rethink = \
"""
Please rethink the reasoning process above and provide your answer. Format strictly as:
<think>your rethought reasoning process</think><answer>the corresponding letter of the best option</answer>
"""