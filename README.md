# G-OPD
[![arXiv](https://img.shields.io/badge/arXiv-2602.12125-red.svg)](https://arxiv.org/abs/2602.12125)


We propose **G-OPD**, a generalized on-policy distillation framework, in which we introduce a reward scaling factor and a flexible reference model. Building on G-OPD, we propose **ExOPD** (On-Policy Distillation with Reward Extrapolation), which outperforms standard OPD in both same-size and strong-to-weak distillation settings.

---------

## News
- [2026.2.16] We release the evaluation code. Training code is coming soon!
- [2026.2.13] We release our paper on [arxiv](https://arxiv.org/abs/2602.12125).


## Installation
Our code is mainly based on [verl](https://github.com/volcengine/verl) (v0.6.1). We will provide the instructions when we release the training code.

## Training
Training code is coming soon!


## Evaluation

### Math Reasoning Evaluation
Math evaluation data is in the ``data/`` folder. Math evaluation code and script are in the ``math_eval/`` folder. 

```bash
cd math_eval/
sh scripts/run_eval_math.sh
```

### Code Generation Evaluation
Our evaluation is mainly based on the code provided in [Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner). 

#### EvalPlus

```bash
CUDA_VISIBLE_DEVICES=0 bash code_eval/scripts/run_evalplus.sh humaneval <MODEL_PATH>  0 1.0 1.0 4
```

#### LiveCodeBench

Download data first

```bash
git clone https://hf-mirror.com/datasets/livecodebench/code_generation_lite code_eval/coding/LiveCodeBench/code_generation_lite
```

Evaluation
```bash
bash code_eval/scripts/run_lcb_gen.sh --model Qwen3-4B-NonThinking --local_model_path  <MODEL_PATH>
```


## Acknowledgments
Our training code is mainly based on [verl](https://github.com/volcengine/verl). Our evaluation code is mainly based on [Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner), which is built opon [EvalPlus](https://github.com/evalplus/evalplus) and [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).



## Citation
If you find our work helpful, please kindly cite as
```bibtex

```