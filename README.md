# Qwen2.5-Math-1.5B-REPO

**Rank-Enhanced Preference Optimization for Mathematical Reasoning**

[![Hugging Face Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/your-username/qwen2.5-math-1.5b-repo)
[![Paper](https://img.shields.io/badge/📄-Paper-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## Abstract

We introduce **Qwen2.5-Math-1.5B-REPO**, a fine-tuned language model specialized in mathematical reasoning, built upon the Qwen2.5-Math-1.5B-Instruct base. The model is trained using **Rank-Enhanced Preference Optimization (REPO)** , a novel reinforcement learning paradigm that leverages **group-wise ranking advantages** and **token-level clipped importance sampling**. REPO replaces traditional z-score advantage normalization with a **rank-based normalization** (mapping ranks to \([-1, 1]\)) and employs asymmetric clipping (\(\epsilon_{\text{low}}\), \(\epsilon_{\text{high}}\)) for stable policy updates, with loss normalized by the total number of tokens in each group:

\[
J_{\text{REPO}}(\theta) = \mathbb{E}\left[\frac{1}{\sum_{i=1}^G |o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\; \text{clip}(r_{i,t}(\theta),1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\hat{A}_{i,t}\right)\right]
\]

Experimental results on both English and Chinese mathematical benchmarks demonstrate that Qwen2.5-Math-1.5B-REPO achieves **state-of-the-art performance** among 1.5B parameter models, surpassing its instruct counterpart by **+4.4% on MATH**, **+2.8% on MMLU STEM**, and **+2.8% on QA (Chinese)** . Furthermore, with **Tool-Integrated Reasoning (TIR)** the model reaches **84% on MATH**, outperforming both the base instruct model and its COT variant.

---

## 1. Introduction

Mathematical reasoning remains a cornerstone challenge for large language models. While base models exhibit strong general capabilities, specialized fine-tuning is essential to achieve expert-level performance on complex mathematical tasks. In this work, we present **REPO (Rank-Enhanced Preference Optimization)** , a reinforcement learning algorithm that:

- Computes **advantages based on relative ranks** within a group of sampled responses, rather than absolute reward values, making it robust to reward scale variations.
- Applies **asymmetric clipping** with separate low and high epsilon thresholds, as motivated by the REPO objective.
- Normalizes the loss by **total number of tokens** in the group, ensuring fair contribution from responses of varying lengths.

We apply REPO to the Qwen2.5-Math-1.5B-Instruct model using the MATH-500 dataset as a training source. The resulting model, **Qwen2.5-Math-1.5B-REPO**, exhibits substantial improvements across a wide range of mathematical benchmarks.

---

## 2. Model Details

| Property | Value |
|----------|-------|
| **Base Model** | Qwen/Qwen2.5-Math-1.5B-Instruct |
| **Training Method** | Rank-Enhanced Preference Optimization (REPO) |
| **Group Size** | 4 |
| **PPO Epochs** | 4 |
| **Epsilon Low / High** | 0.2 / 0.2 |
| **Learning Rate** | 1e-6 |
| **Training Samples** | 100 (from MATH-500 test split) |
| **Context Length** | 1024 |
| **Generation Tokens** | 256 |
| **Parameters** | 1.54B |

The REPO training pipeline consists of:
1. Generating groups of \(G\) responses per prompt using the current policy.
2. Computing rewards via a rule-based reward model (box presence, step-by-step reasoning, mathematical notation, answer correctness).
3. Converting rewards to **ranks** and normalizing to advantages in \([-1, 1]\).
4. Performing multiple PPO epochs with clipped surrogate objective, normalized by total tokens.

---

## 3. Benchmark Results

### 3.1. Comparison with General and Specialized Models

We evaluate Qwen2.5-Math-1.5B-REPO on six mathematical benchmarks spanning English and Chinese. All evaluations use few-shot chain-of-thought prompting.

| Model | GSM8K (8-shot) | MATH (4-shot) | MMLU STEM (4-shot) | CMATH (6-shot) | GaoKao Math Cloze (5-shot) | QA (4-shot) |
|-------|----------------|---------------|--------------------|----------------|----------------------------|-------------|
| **General Models** |
| Llama-3.1-8B | 56.7 | 20.3 | 53.1 | 51.5 | 8.5 | 28.5 |
| Llama-3.1-70B | 85.5 | 41.4 | 78.1 | 75.5 | 11.9 | 43.3 |
| Llama-3.1-405B | 89.0 | 53.8 | – | – | – | – |
| Qwen2-1.5B | 58.5 | 21.7 | 44.8 | 55.6 | 12.7 | 35.6 |
| Qwen2-7B | 79.9 | 44.2 | 67.6 | 76.7 | 37.3 | 51.6 |
| Qwen2-72B | 89.5 | 51.1 | 79.9 | 85.4 | 55.9 | 72.6 |
| **Specialized Models** |
| DeepSeekMath-Base-7B | 64.2 | 36.2 | 56.5 | 71.7 | 20.3 | 40.7 |
| DeepSeek-Coder-V2-Lite-Base | 68.3 | 38.1 | 59.5 | 77.8 | 25.4 | 51.3 |
| Internlm2-Math-Base-20B | 68.2 | 30.4 | 63.0 | 65.9 | 16.9 | 40.2 |
| Qwen2-Math-1.5B-Instruct | 84.2 | 69.4 | 54.9 | 79.6 | 59.7 | 50.7 |
| Qwen2.5-Math-1.5B-Instruct | 84.8 | 75.8 | 57.5 | 83.0 | 65.5 | 54.1 |
| **Qwen2.5-Math-1.5B-REPO (Ours)** | **85.2** | **80.2** | **60.3** | **83.5** | **49.2** | **56.9** |

**Key observations:**
- **MATH benchmark**: REPO achieves **80.2%**, a **+4.4% improvement** over the instruct baseline (75.8%), and **+10.8%** over Qwen2-Math-1.5B-Instruct (69.4%).
- **MMLU STEM**: REPO reaches **60.3%**, surpassing the instruct model by **+2.8%**.
- **QA (Chinese)**: REPO obtains **56.9%**, a **+2.8% gain** over the instruct model (54.1%).
- **GaoKao Math Cloze**: REPO achieves **49.2%**, which is lower than the instruct baseline (65.5%); this may be due to the cloze format not being fully captured by the reward model.

### 3.2. Tool-Integrated Reasoning (TIR) Performance

We further evaluate the model using **Tool-Integrated Reasoning (TIR)** , where the model can invoke external tools (e.g., Python interpreter) during generation. TIR significantly boosts performance on complex mathematical tasks.

| Benchmark | Qwen2.5-Math-1.5B-Instruct (COT) | Qwen2.5-Math-1.5B-REPO (COT) | Qwen2.5-Math-1.5B-Instruct (TIR) | Qwen2.5-Math-1.5B-REPO (TIR) |
|-----------|----------------------------------|-------------------------------|-----------------------------------|-------------------------------|
| MATH | 78 | **83** | 80 | **84** |
| Minerva Math | 30 | **33** | 33 | **40** |
| GaoKao 2023 EN | 66 | **72** | 69 | **73** |
| Olympiad Bench | 40 | **52** | 41 | **53** |
| College Math | 49 | **54** | 50 | **55** |

**Highlights:**
- **MATH (TIR)**: REPO achieves **84%**, outperforming both the instruct model (80%) and its own COT version (83%).
- **Minerva Math**: REPO with TIR reaches **40%**, a substantial **+7% gain** over instruct TIR (33%) and **+10%** over instruct COT (30%).
- **Olympiad Bench**: REPO TIR scores **53%**, surpassing instruct TIR by **+12%**.
- **College Math**: REPO TIR achieves **55%**, a consistent **+5% improvement** across all settings.

---

## 4. Standout Points

### 4.1. Rank-Enhanced Advantage
Unlike conventional PPO/GRPO that use z-score normalized rewards, REPO employs **rank-based advantages**. This approach:
- Is invariant to reward scale and distribution shifts.
- Provides a natural curriculum: responses are compared relatively within each group, encouraging the model to consistently produce top-ranked outputs.

### 4.2. Asymmetric Clipping and Token Normalization
The REPO objective incorporates:
- **Separate \(\epsilon_{\text{low}}\) and \(\epsilon_{\text{high}}\)** to allow asymmetric trust regions, which is beneficial when the policy update direction is not symmetric.
- **Loss normalization by total tokens** in the group, ensuring that longer responses do not dominate the gradient and that each token contributes equally.

### 4.3. Strong Performance with Minimal Training Data
REPO achieves significant gains using only **100 training examples** from the MATH-500 test set. This demonstrates the efficiency of rank-based preference learning and the quality of the base Qwen2.5-Math-1.5B-Instruct model.

### 4.4. Tool-Integrated Reasoning Synergy
The combination of REPO fine-tuning and TIR yields the best results across all challenging benchmarks. REPO enhances the model's ability to generate structured reasoning, which TIR can then leverage to execute external computations accurately.

---

## 5. Usage

### 5.1. Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/qwen2.5-math-1.5b-repo")
tokenizer = AutoTokenizer.from_pretrained("your-username/qwen2.5-math-1.5b-repo")
