<div align="center">
    <a href="https://arxiv.org/abs/2512.09271"><img src="https://img.shields.io/badge/Arxiv-preprint-red"></a>
    <a href="https://welldky.github.io/LongT2IBench-Homepage/"><img src="https://img.shields.io/badge/Homepage-green"></a>
    <!-- <a href="https://huggingface.co/spaces/orpheus0429/FGResQ"><img src="https://img.shields.io/badge/?¤?%20Hugging%20Face-Spaces-blue"></a> -->
    <a href='https://github.com/yzc-ippl/LongT2IBench/stargazers'><img src='https://img.shields.io/github/stars/yzc-ippl/LongT2IBench.svg?style=social'></a>

</div>

<h1 align="center">LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations</h1>

<div align="center">
    <a href="https://github.com/yzc-ippl/" target="_blank">Zhichao Yang</a><sup>1</sup>,
    <a href="https://github.com/welldky" target="_blank">Tianjiao Gu</a><sup>1</sup>,
    <a href="https://github.com/satan-7" target="_blank">Jianjie Wang</a><sup>1</sup>,
    <a href="https://github.com/Guapicat0" target="_blank">Feiyu Lin</a><sup>1</sup>,
    <a href="https://github.com/sxfly99" target="_blank">Xiangfei Sheng</a><sup>1</sup>,
    <a href="https://faculty.xidian.edu.cn/cpf/" target="_blank">Pengfei Chen</a><sup>1*</sup>,
    <a href="https://web.xidian.edu.cn/ldli/" target="_blank">Leida Li</a><sup>1,2*</sup>
</div>

<div align="center">
  <sup>1</sup>School of Artificial Intelligence, Xidian University
  <br>
  <sup>2</sup>State Key Laboratory of Electromechanical Integrated Manufacturing of High-Performance Electronic Equipments, Xidian University
</div>

<div align="center">
<sup>*</sup>Corresponding author
</div>

<div align="center">
  <img src="LongT2IBench.png" width="800"/>
</div>

<div style="font-family: sans-serif; margin-bottom: 2em;">
    <h2 style="border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-bottom: 1em;">News</h2>
    <ul style="list-style-type: none; padding-left: 0;">
        <li style="margin-bottom: 0.8em;">
            <strong>[2025-12-09]</strong> The data and pre-trained models have been released.
        </li>
        <li style="margin-bottom: 0.8em;">
            <strong>[2025-11-08]</strong> Our paper, "LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations", has been accepted for an oral presentation at AAAI 2026!
        </li>
    </ul>
</div>

## Quick Start

This guide will help you get started with the LongT2IBench inference code.

### 1. Installation

First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yzc-ippl/LongT2IBench.git
cd LongT2IBench
pip install -r requirements.txt
```

### 2. Download Pre-trained Weights and Dataset

##### Prepare Pre-trained Weights

You can download the pre-trained model weights of <strong>[LongT2IExpert]</strong> from the following link: [**(Baidu Netdisk)**](https://pan.baidu.com/s/1Ltj77l31hyBkn6nLtYctnQ?pwd=i8ug)

Place the downloaded files in the `weights` directory.

- ``./weights/LongT2IBench-checkpoints``: The main model for generation and scoring.

Create the `weights` directory if it doesn't exist and place the files inside.

##### Prepare Datasets

You can download the dataset of <strong>[LongPrompt-3K]</strong> and <strong>[LongT2IBench-14K]</strong> from the following link: [**(Baidu Netdisk)**](https://pan.baidu.com/s/1uNMzLd1HKEQCaTgJQm8lpg?pwd=igq2)

Place the downloaded files in the `data` directory.

Create the `data` directory if it doesn't exist and place the files inside.

```
LongT2IBench/
|-- weights/
|   |-- LongT2IBench-checkpoints
|   |   |-- config.json
|   |   |-- ...
|-- data/
|   |-- imgs
|   |-- split
|   |   |-- train.json
|   |   |-- test.json
|   |   |-- val.json
|-- config.py
|-- dataset.py
|-- model.py
|-- requirements.txt
|-- README.md
|-- test_generation.py
|-- test_score.py
```

### 3. Run Inference

The `LongT2IExpert` provides two main inference tasks: Long T2I Alignment Scoring and Long T2I Alignment Interpreting.

##### Long T2I Alignment Scoring

```
python test_score.py
```

##### Long T2I Alignment Interpreting

```
python test_generation.py
```

### 4. Run Training

You can run this code to train <strong>[LongT2IExpert]</strong> from start to finish. 

First, make sure the initially untrained weights are located at 

- ``./weights/Qwen2.5-VL-7B-Instruct`` ：You can download the untrained weights from the following link [**(Baidu Netdisk)**](https://pan.baidu.com/s/17PcO4CvgB6FDHh6JBgM_Lg?pwd=3h8m)

```bash
python train.py
```

## Citation

If you find this work is useful, pleaes cite our paper!

```bibtex
@misc{yang2025longt2ibenchbenchmarkevaluatinglong,
      title={LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations}, 
      author={Zhichao Yang and Tianjiao Gu and Jianjie Wang and Feiyu Lin and Xiangfei Sheng and Pengfei Chen and Leida Li},
      year={2025},
      eprint={2512.09271},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.09271}, 
}
```
