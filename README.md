<div align="center">
  <h1> REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding </h1>

  <h3>🏆 CVPR 2026</h3>

  <br>

  <a href="https://arxiv.org/abs/2511.13026">
    <img src="https://img.shields.io/badge/arXiv-2511.13026-b31b1b.svg?style=flat" alt="Paper">
  </a>
  <a href="https://huggingface.co/williamljz/REVISOR">
    <img src="https://img.shields.io/badge/🤗 Hugging Face-Model-FFD21E.svg?style=flat" alt="Model">
  </a>
  <a href="https://huggingface.co/datasets/williamljz/REVISOR-25k">
    <img src="https://img.shields.io/badge/🤗 Hugging Face-Dataset-FFD21E.svg?style=flat" alt="Dataset">
  </a>
</div>

<br>

<p align="center">
Jiaze Li<sup>1*</sup>, Hao Yin<sup>1*</sup>, Wenhui Tan<sup>3*</sup>, Jingyang Chen<sup>1*</sup>, Boshen Xu<sup>3</sup>, Yuxun Qu<sup>1</sup>, Yijing Chen<sup>3</sup>, Jianzhong Ju<sup>1†</sup>, Zhenbo Luo<sup>1</sup>, Jian Luan<sup>1</sup>
<br>
<sup>1</sup> MiLM Plus, Xiaomi Inc. &nbsp; <sup>3</sup> Renmin University of China
<br>
<sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Corresponding author
</p>

---

## Abstract

Self-reflection mechanisms that rely on purely text-based rethinking processes perform well in most multimodal tasks. However, when directly applied to long-form video understanding scenarios, they exhibit clear limitations: (1) long-form video understanding involves richer and more dynamic visual input, meaning rethinking only the text information is insufficient; (2) purely text-based reflection mechanisms lack cross-modal interaction capabilities.

We propose **REVISOR** (REflective VIsual Segment Oriented Reasoning), a novel framework for tool-augmented multimodal reflection. REVISOR enables MLLMs to collaboratively construct introspective reflection processes across textual and visual modalities, significantly enhancing their reasoning capability for long-form video understanding. We further design the **Dual Attribution Decoupled Reward (DADR)** mechanism integrated into the GRPO training strategy, enforcing causal alignment between the model's reasoning and the selected video evidence.

REVISOR achieves impressive results on **VideoMME**, **LongVideoBench**, **MLVU**, and **LVBench** without requiring supplementary supervised fine-tuning or external models.

## News

- **[2026/06]** Code, model weights, and training dataset are released!
- **[2026/02]** REVISOR is accepted to **CVPR 2026**!
- **[2025/11]** Paper is available on [arXiv](https://arxiv.org/abs/2511.13026).

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/SeerRay-Lab/REVISOR.git
cd REVISOR

# Install dependencies
pip install -e .
pip install vllm==0.7.3
pip install flash_attn==2.7.4.post1
```

### Training

#### 1. Download Training Data

Download the training dataset from [Hugging Face](https://huggingface.co/datasets/williamljz/REVISOR-25k):

```bash
huggingface-cli download williamljz/REVISOR-25k --repo-type dataset --local-dir ./data/REVISOR-25k
```


#### 2. Construct Training Parquet Files

Scripts for constructing each subset are provided under `scripts/datasets/`:

```bash
python scripts/datasets/construct_video_r1.py
python scripts/datasets/construct_time_r1.py
python scripts/datasets/construct_cgbench.py
python scripts/datasets/construct_rextime.py
```

#### 3. Launch Training

Training uses GRPO with 8 GPUs on a single node. We use [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) as the base model.

```bash
bash scripts/train/train.sh
```

#### 1. Run Benchmark Evaluation

| Benchmark | Command |
|-----------|---------|
| VideoMME | `bash eval/run_multiple_videomme.sh 256 <ckpt_path>` |
| LongVideoBench | `bash eval/run_multiple_longvideobench.sh 256 <ckpt_path>` |
| MLVU | `bash eval/run_multiple_mlvu.sh 256 <ckpt_path>` |
| LVBench | `bash eval/run_multiple_lvbench.sh 256 <ckpt_path>` |

**Example:**
```bash
bash eval/run_multiple_videomme.sh 256 /path/to/your/checkpoint
```

## Model Weights

Pre-trained model weights are available on Hugging Face:

| Model | Base | Link |
|-------|------|------|
| REVISOR | Qwen2.5-VL-7B-Instruct | [williamljz/REVISOR](https://huggingface.co/williamljz/REVISOR) |



## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{li2026revisor,
  title={Revisor: Beyond textual reflection, towards multimodal introspective reasoning in long-form video understanding},
  author={Li, Jiaze and Yin, Hao and Tan, Wenhui and Chen, Jingyang and Xu, Boshen and Qu, Yuxun and Chen, Yijing and Ju, Jianzhong and Luo, Zhenbo and Luan, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5059--5069},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [DeepEyes](https://github.com/Visual-Agent/DeepEyes) for their excellent agentic RL training framework, on which our implementation is built. We also thank the authors of [vLLM](https://github.com/vllm-project/vllm) and [VeRL](https://github.com/volcengine/verl) for their foundational infrastructure.

## License

This project is released under the [Apache 2.0 License](./LICENSE).
