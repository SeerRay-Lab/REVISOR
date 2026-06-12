from openai import OpenAI
import requests
import random
import re
import os
import json

# from math_verify import parse, verify

openai_api_key = "EMPTY"
# openai_api_base_list = [
#     # "http://172.30.52.123:8000/v1",
#     # "http://10.39.3.123:18901/v1",
#     os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.44.241.49:18901/v1"),
# ]

# client_list = []
# for api_base in openai_api_base_list:
#     client = OpenAI(
#         api_key=openai_api_key,
#         base_url=api_base,
#     )
#     client_list.append(client)
# model_name_list = []
# for client in client_list:
#     response = requests.get(f"{api_base}/models")
#     models = response.json()
#     model_name_list.append(models['data'][0]['id'])

CONSISTENCY_PROMPT_temporal_grounding="""You are an expert about question answering. I will provide you a solution process and a final answer to the same question. Please evaluate the Consistency between the solution process and the final anwer: If the solution process draws the same conclusion with the final answer, rate the Consistency as 1, else rate 0.
Here is the solution process and the final answer for you to evaluate:\n

#### Solution Process: 
{gen_solution}

#### Final Answer: 
{gen_answer}

### Output Format (strictly follow)

Please provide an integer score to indicate the Consistency. Output the score in a JSON dictionary with nothing else for easy processing, in this form: {"Consistency": score}.

Your Evaluation Result:"""

CONSISTENCY_PROMPT="""You are an expert in evaluating multiple-choice question answers. I will provide a solution process and a final answer option (such as "A", "B", etc.) for the same multiple-choice question. Your task is to assess consistency: If the reasoning in the solution process logically supports and arrives at the exact final answer option provided, assign a score of 1; otherwise, assign 0. Ignore any extraneous details and focus only on whether the process justifies the specific option.
Here is the content to evaluate:

#### Solution Process: 
{gen_solution}

#### Final Answer: 
{gen_answer}

### Output Format (strictly follow)

Please provide an integer score to indicate the Consistency. Output the score in a JSON dictionary with nothing else for easy processing, in this form: {"Consistency": score}.

Your Evaluation Result:"""


def ours_consistency_reward(think_process: str, draft_answer: str) -> float:
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]
    while True:
        # think_process = predict_str.split("</think>")[0].split("<think>")[-1].strip()
        # draft_answer = predict_str.split("</draft>")[0].split("<draft>")[-1].strip()
        full_prompt = CONSISTENCY_PROMPT.replace("gen_solution", think_process).replace("gen_answer", draft_answer)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user", 
                    "content": full_prompt,
                }
            ],
            # extra_body={"chat_template_kwargs": {"enable_thinking": False,}},
            # seed=random.randint(0, 65535),
            # temperature=0.3,
        )
        # try:
        completion = completion.choices[0].message.content
        if "```" in completion:
            pattern = re.compile(r'```(.*?)```', re.DOTALL)
            matches = pattern.findall(completion)
            if len(matches) == 1:
                completion = matches[0]

        json_str = completion[completion.find('{'): completion.find('}', completion.find('{') + 1) + 1]
        json_str = json_str.replace('\\', '\\\\')
        completion = json.loads(json_str)
        outcome_r = int(completion['Consistency'])
        
        return outcome_r

def ours_consistency_reward_temporal_grounding(think_process: str, draft_answer: str) -> float:
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]
    while True:
        # think_process = predict_str.split("</think>")[0].split("<think>")[-1].strip()
        # draft_answer = predict_str.split("</draft>")[0].split("<draft>")[-1].strip()
        full_prompt = CONSISTENCY_PROMPT_temporal_grounding.replace("gen_solution", think_process).replace("gen_answer", draft_answer)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user", 
                    "content": full_prompt,
                }
            ],
            # extra_body={"chat_template_kwargs": {"enable_thinking": False,}},
            # seed=random.randint(0, 65535),
            # temperature=0.3,
        )
        # try:
        completion = completion.choices[0].message.content
        if "```" in completion:
            pattern = re.compile(r'```(.*?)```', re.DOTALL)
            matches = pattern.findall(completion)
            if len(matches) == 1:
                completion = matches[0]

        json_str = completion[completion.find('{'): completion.find('}', completion.find('{') + 1) + 1]
        json_str = json_str.replace('\\', '\\\\')
        completion = json.loads(json_str)
        outcome_r = int(completion['Consistency'])
        
        return outcome_r

def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #F
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt


def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。
    
    参数:
        text (str): 包含<answer>标签的文本
        
    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_video_score_ours_past(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # breakpoint()
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|video_pad|>")
    count_vision_2 = predict_str.count("<|video_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1 == 0:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    if check_multi_choice_ans(answer_text, ground_truth): # TODO: check HERE [cbzhang]
        acc_reward = 1.0
    else:
        acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    if count_vision_1 > 0 and acc_reward > 0.5:
        tool_count = min(count_vision_1, count_vision_2)
        tool_reward = 1.2 - 0.2 * (tool_count - 1)
    else:
        tool_reward = 0.0
    # tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    # reward 1
    # return 0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base
    # reward 2
    return 0.8 * acc_reward + 0.2 * format_reward + 1. * tool_reward

def extract_action(action_string: str):
    """
    Extracts the start and end time values from the action string.

    Args:
        action_string: The string containing the <time_interval> XML tag.
        
    Returns:
        A tuple of two floats: (start_time, end_time), or None if not found/invalid.
    """
    try:
        match = re.search(
            r"<time_interval>\s*\[([\d\.]+)\s*,\s*([\d\.]+)\]\s*</time_interval>",
            action_string
        )
        if not match:
            return None

        start_time = float(match.group(1))
        end_time = float(match.group(2))

        return end_time - start_time
    except Exception:
        return None

def compute_video_score_ours(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
  
    time_interval = None
    time_interval = extract_action(predict_str)
    duration = extra_info.get('duration', None)
    reward_weight = 0.9999
    if time_interval is not None and duration is not None:
        duration = float(duration)
        reward_weight = time_interval / duration
    
    reward_weight = 1 - reward_weight
    

    extra_actions = extra_info.get('extra_actions', None)
    if extra_actions is not None:
        extra_actions = extra_actions.replace('<|im_end|>', '')
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|video_pad|>")
    count_vision_2 = predict_str.count("<|video_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1 == 0:
        is_format_error = True
        
    count_time_interval_1 = predict_str.count("<time_interval>[")
    count_time_interval_2 = predict_str.count("]</time_interval>")
    if count_time_interval_1 != count_time_interval_2 or count_time_interval_1 == 0:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    if check_multi_choice_ans(answer_text, ground_truth): 
        acc_reward = 1.0
    else:
        acc_reward = 0.0
    
    clip_reward = 0.
    if extra_actions is not None:
        if check_multi_choice_ans(extra_actions, ground_truth):
            clip_reward = 1.0


    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0

  
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    
    return 0.6 * acc_reward + 0.1 * format_reward + 0.3 * clip_reward







def parse_time(action_string: str):
    """
    Extracts the start and end time values from the action string.

    Args:
        action_string: The string containing the <time_interval> XML tag.
        
    Returns:
        A tuple of two floats: (start_time, end_time), or None if not found/invalid.
    """
    try:
        match = re.search(
            r"\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]",
            action_string
        )
        if not match:
            return None

        start_time = float(match.group(1))
        end_time = float(match.group(2))

        return start_time, end_time
    except Exception:
        return None

def compute_temporal_grounding(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # breakpoint()

    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|video_pad|>")
    count_vision_2 = predict_str.count("<|video_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()       
    count_time_interval_1 = predict_no_think.count("<time_interval>[")
    count_time_interval_2 = predict_no_think.count("]</time_interval>")
    if count_time_interval_1 != count_time_interval_2 or count_time_interval_1 == 0:
        is_format_error = True
    
    answer_text = predict_str.split("<time_interval>")[-1].split("</time_interval>")[0].strip()

    acc_reward = 0.
    
    pred_times = parse_time(answer_text)
    
    gt_times = parse_time(str(ground_truth))
    
    if pred_times is not None and gt_times is not None:
        from_number, to_number = pred_times
        s, e = gt_times
        intersection = max(0, min(to_number, e) - max(from_number, s))
        union = max(to_number, e) - min(from_number, s)
        if union > 0:
            iou = intersection / union   # 0.1 0.3
        else:
            iou = 1. 
        acc_reward = iou


    format_reward = -1.0 if is_format_error else 0.0

    return 1.0 * acc_reward + 0.1 * format_reward 

def compute_temporal_grounding_thinking(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # breakpoint()

    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|video_pad|>")
    count_vision_2 = predict_str.count("<|video_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()       
    count_time_interval_1 = predict_no_think.count("<time_interval>[")
    count_time_interval_2 = predict_no_think.count("]</time_interval>")
    if count_time_interval_1 != count_time_interval_2 or count_time_interval_1 == 0:
        is_format_error = True
    
    answer_text = predict_str.split("<time_interval>")[-1].split("</time_interval>")[0].strip()

    acc_reward = 0.
    
    pred_times = parse_time(answer_text)
    
    gt_times = parse_time(str(ground_truth))
    
    if pred_times is not None and gt_times is not None:
        from_number, to_number = pred_times
        s, e = gt_times
        intersection = max(0, min(to_number, e) - max(from_number, s))
        union = max(to_number, e) - min(from_number, s)
        if union > 0:
            iou = intersection / union   # 0.1 0.3
        else:
            iou = 1. 
        acc_reward = iou


    format_reward = -1.0 if is_format_error else 0.0

    answer_thinking = re.findall(r'<think>(.*?)</think>', predict_str, re.DOTALL)
    
    thinking_reward = 0.0
    if answer_thinking:
        # 返回最后一个匹配项
        answer_thinking = answer_thinking[-1]
        thinking_reward = ours_consistency_reward_temporal_grounding(answer_thinking, answer_text)

    return (1.0 + 0.2 * thinking_reward) * acc_reward + 0.1 * format_reward 

def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
 
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()



    question_text = extra_info['question']
    full_prompt = get_prompt(answer_text, ground_truth, question_text)

    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        seed = random.randint(0, 1000000),
        temperature=0.3,
    )
    response = chat_response.choices[0].message.content.strip()
    # print(response)
    if 'Judgement:' in response:
        response = response.split('Judgement:')[-1].strip()
        if '1' in response:
            acc_reward = 1.0
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        if response == '1':
            acc_reward = 1.0
        elif response == '0':
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0

    
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0

    return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward





def compute_common_reasoning(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = extract_answer(predict_no_think) # predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
    if not answer_text:
        acc_reward = 0.0
        is_format_error = True
    elif len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True
    else:
        question_text = extra_info['question']
        client_idx = random.randint(0, len(client_list) - 1)
        client = client_list[client_idx]
        model_name = model_name_list[client_idx]
        full_prompt = COMMON_VERIFY_PROMPT.format(
            query=question_text,
            gold_ans=ground_truth,
            pred_ans=answer_text,
        )

        acc_reward = 0.0
        for ix in range(8):
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.5,
            )
            response = chat_response.choices[0].message.content.strip()
            judgement = response.split('## Equivalence Judgement')[-1].lower()
            if 'true' in judgement and 'false' not in judgement:
                acc_reward = 1.0
                break
            elif 'false' in judgement and 'true' not in judgement:
                acc_reward = 0.0
                break
            else:
                print(f' [ERROR] judgement format invalid: {judgement}')
                continue

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {answer_text=}, {acc_reward=}, {format_reward=}')
    return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward


def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)



def check_multi_choice_ans(pred, gt): 
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


def generative_verify(query, ground_truth, model_answer):
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    for it in range(8):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR math] generative_verify error: {e}')
            continue
    
    judgement = response.split('## Equivalence Judgement')[-1].lower()
    if 'true' in judgement and 'false' not in judgement:
        return True
    elif 'false' in judgement and 'true' not in judgement:
        return False
    else:
        print(f' [ERROR math] verify bug output: ')


def compute_score_math(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    model_answer = ""
    predict_no_think = predict_str.split('</think>')[-1].strip()
    answer_pattern = r'\\boxed{([^}]+)}'
    answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
    if len(answer_list) == 0:
        acc_reward = 0.0
        is_format_error = True
    else:
        if len(answer_list) > 1:
            is_format_error = True

        model_answer = answer_list[-1]
        if rule_math_verify(ground_truth, model_answer):
            acc_reward = 1.0
        else:
            acc_reward = 1.0 if generative_verify(extra_info['question'], ground_truth, model_answer) else 0.0
    
    format_reward = -1.0 if is_format_error else 0.0
    print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {model_answer=}, {acc_reward=}, {format_reward=}')
    return 1.2 * acc_reward + 0.4 * format_reward



if __name__ == '__main__':
    predict_str = "The answer is <think> 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    ground_truth = "left"
    extra_info = {'answer': 'The woman is to the left of the man who is holding the camera.', 'id': 0, 'image': '/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg', 'pred_ans': 'The woman is to the right of the man who is holding the camera.', 'question': 'Is the woman to the left or to the right of the man who is holding the camera?'}

    score = compute_score(predict_str, ground_truth, extra_info)
    print(f"Score: {score}")