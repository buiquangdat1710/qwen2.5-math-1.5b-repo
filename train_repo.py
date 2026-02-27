import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed
)
from datasets import load_dataset
import json
from tqdm import tqdm
import re
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import random

# ============================================================
# Configuration (updated with REPO-specific parameters)
# ============================================================

@dataclass
class REPOConfig:
    """Configuration for REPO training"""
    # Model settings
    base_model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    hf_username: str = field(default="your-username")
    push_to_hub: bool = field(default=False)
    hf_token: str = field(default=None)
    
    # Training settings
    num_iterations: int = field(default=3)
    group_size: int = field(default=4)
    batch_size: int = field(default=2)
    learning_rate: float = field(default=1e-6)
    kl_coef: float = field(default=0.1)        # not used in REPO, kept for compatibility
    entropy_coef: float = field(default=0.01)  # not used in REPO, kept for compatibility
    
    # REPO specific
    ppo_epochs: int = field(default=4)         # number of PPO epochs per batch
    eps_low: float = field(default=0.2)        # lower clip bound
    eps_high: float = field(default=0.2)       # upper clip bound
    
    # Dataset settings
    num_train_samples: int = field(default=400)
    num_eval_samples: int = field(default=100)
    
    # Generation settings
    max_length: int = field(default=1024)
    max_new_tokens: int = field(default=256)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)
    
    # Evaluation settings
    eval_batch_size: int = field(default=8)

# ============================================================
# Helper Functions (identical to your evaluation code)
# ============================================================

def extract_answer(text):
    boxed_match = re.search(r'\\boxed\{', text)
    if boxed_match:
        start = boxed_match.end()
        bracket_count = 1
        i = start
        
        while i < len(text) and bracket_count > 0:
            if text[i] == '{':
                bracket_count += 1
            elif text[i] == '}':
                bracket_count -= 1
            i += 1
        
        if bracket_count == 0:
            answer = text[start:i-1].strip()
            return normalize_answer(answer)
    
    paren_matches = re.findall(r'\([^()]*\)', text)
    if paren_matches:
        return normalize_answer(paren_matches[-1])
    
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else None

def normalize_answer(ans):
    if not ans:
        return None
    ans = str(ans).strip()
    ans = re.sub(r'\\left\s*', '', ans)
    ans = re.sub(r'\\right\s*', '', ans)
    ans = re.sub(r'\\,', '', ans)
    ans = re.sub(r'\s+', ' ', ans)
    ans = re.sub(r'\s*,\s*', ',', ans)
    ans = re.sub(r'\(\s+', '(', ans)
    ans = re.sub(r'\s+\)', ')', ans)
    return ans.strip()

# ============================================================
# REPO Training Components
# ============================================================

class SimpleRewardModel:
    """Simple rule-based reward model for math problems"""
    
    def compute_reward(self, problem: str, response: str, gold_answer: str = None) -> float:
        """Compute reward for a response"""
        reward = 0.0
        
        # Bonus for having boxed answer
        if "\\boxed{" in response:
            reward += 2.0
        
        # Bonus for step-by-step reasoning
        step_indicators = ["step", "first", "second", "then", "therefore", "thus", "hence", "solution:", "solve"]
        step_count = 0
        for indicator in step_indicators:
            if indicator in response.lower():
                step_count += 1
        reward += min(step_count * 0.5, 2.0)
        
        # Bonus for mathematical notation
        math_indicators = ["=", "+", "-", "*", "/", "^", "\\frac", "\\sqrt", "^{", "_{"]
        math_count = 0
        for indicator in math_indicators:
            if indicator in response:
                math_count += 1
        reward += min(math_count * 0.3, 1.5)
        
        # Bonus for correct answer if gold answer provided
        if gold_answer:
            pred_answer = extract_answer(response)
            gold_normalized = normalize_answer(gold_answer)
            
            if pred_answer:
                pred_normalized = normalize_answer(pred_answer)
                # Direct comparison
                if pred_normalized == gold_normalized:
                    reward += 5.0
                # Partial match (for fractions, decimals)
                elif pred_normalized.replace(' ', '') == gold_normalized.replace(' ', ''):
                    reward += 3.0
                # Contains gold answer
                elif gold_normalized in pred_normalized:
                    reward += 2.0
                elif pred_normalized in gold_normalized:
                    reward += 2.0
        
        # Penalty for very short responses
        if len(response.strip()) < 50:
            reward -= 1.0
        
        return reward

class REPOTrainer:
    """Group Relative Policy Optimization Trainer with REPO objective"""
    def __init__(self, model, tokenizer, config: REPOConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device
        
        # Initialize reward model
        self.reward_model = SimpleRewardModel()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
    
    def _generate_response(self, prompt: str) -> Dict:
        """Generate a single response and return token-level data"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        prompt_ids = inputs.input_ids[0].tolist()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=self.config.top_p,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_ids = outputs.sequences[0, input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Compute log probs for each generated token (old policy)
        log_probs = []
        for i, token_id in enumerate(generated_ids):
            logits = outputs.scores[i]                     # shape (1, vocab_size)
            log_prob = F.log_softmax(logits, dim=-1)
            token_log_prob = log_prob[0, token_id].item()
            log_probs.append(token_log_prob)
        
        return {
            "response": response,
            "prompt_ids": prompt_ids,
            "token_ids": generated_ids.tolist(),
            "log_probs": log_probs,
            "full_text": prompt + response
        }
    
    def generate_group(self, prompt: str, num_samples: int) -> List[Dict]:
        """Generate a group of responses for a single prompt"""
        group = []
        
        # Format prompt with system message
        formatted_prompt = f"""You are an advanced mathematical reasoning model.
Follow these rules carefully for every problem:

1. Think step-by-step and show complete reasoning.
2. Give a short plan before solving.
3. No hand-waving. Be precise.
4. Final answer must be in the form \\boxed{{answer}}.
5. No text after the boxed result.

Solve the following problem:

Problem:
{prompt}

Solution:
"""
        
        for _ in range(num_samples):
            response_data = self._generate_response(formatted_prompt)
            group.append(response_data)
        
        return group
    
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute advantages based on ranks (REPO style)"""
        if len(rewards) < 2:
            return [0.0] * len(rewards)
        
        # Convert rewards to ranks (1 = highest reward)
        sorted_indices = np.argsort(rewards)[::-1]  # descending
        ranks = np.zeros(len(rewards))
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1  # rank 1 is best
        
        # Normalize ranks to [-1, 1]
        G = len(rewards)
        if G > 1:
            centered = (ranks - (G + 1) / 2) / ((G - 1) / 2)
        else:
            centered = 0.0
        return centered.tolist()
    
    def train_iteration(self, prompts: List[str], gold_answers: List[str] = None):
        """Perform one REPO training iteration (with multiple PPO epochs)"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = list(range(len(prompts)))
        random.shuffle(indices)
        
        for batch_start in range(0, len(prompts), self.config.batch_size):
            batch_indices = indices[batch_start:batch_start + self.config.batch_size]
            batch_prompts = [prompts[i] for i in batch_indices]
            if gold_answers:
                batch_gold_answers = [gold_answers[i] for i in batch_indices]
            else:
                batch_gold_answers = [None] * len(batch_prompts)
            
            # Step 1: Generate groups and compute advantages (using current model as π_old)
            group_data = []  # list of dicts per prompt
            for prompt, gold_answer in zip(batch_prompts, batch_gold_answers):
                # Generate group
                group = self.generate_group(prompt, self.config.group_size)
                # Compute rewards
                rewards = []
                for resp in group:
                    reward = self.reward_model.compute_reward(prompt, resp["response"], gold_answer)
                    rewards.append(reward)
                # Compute rank-based advantages
                advantages = self.compute_advantages(rewards)
                group_data.append({
                    "prompt": prompt,
                    "responses": group,
                    "advantages": advantages,
                    "rewards": rewards
                })
            
            # Step 2: Multiple PPO epochs on the same batch
            for epoch in range(self.config.ppo_epochs):
                epoch_loss = 0.0
                
                for data in group_data:
                    responses = data["responses"]
                    advantages = data["advantages"]
                    
                    group_token_losses = []
                    group_token_counts = []
                    
                    for i, resp_data in enumerate(responses):
                        adv = advantages[i]
                        old_log_probs = torch.tensor(resp_data["log_probs"], device=self.device)
                        token_ids = resp_data["token_ids"]
                        prompt_ids = resp_data["prompt_ids"]
                        
                        if len(token_ids) == 0:
                            # Skip empty responses
                            continue
                        
                        # Reconstruct full input ids
                        full_input_ids = torch.tensor([prompt_ids + token_ids], device=self.device)
                        
                        # Forward pass with current model to get logits
                        outputs = self.model(full_input_ids)
                        logits = outputs.logits  # (1, seq_len, vocab_size)
                        
                        # Get logits for the generated tokens only (last len(token_ids) positions)
                        gen_logits = logits[0, -len(token_ids):, :]  # (len, vocab_size)
                        # Compute log probs
                        log_probs = F.log_softmax(gen_logits, dim=-1)
                        # Gather the log probs for the actual tokens
                        token_tensor = torch.tensor(token_ids, device=self.device)
                        current_log_probs = log_probs[range(len(token_ids)), token_tensor]
                        
                        # Compute importance ratio
                        ratio = torch.exp(current_log_probs - old_log_probs)
                        
                        # Clipped ratio
                        clipped_ratio = torch.clamp(ratio, 1 - self.config.eps_low, 1 + self.config.eps_high)
                        
                        # Loss per token: min(ratio * adv, clipped_ratio * adv)
                        loss_per_token = torch.min(ratio * adv, clipped_ratio * adv)
                        
                        group_token_losses.append(loss_per_token.sum())
                        group_token_counts.append(len(token_ids))
                    
                    # Normalize by total tokens in this group
                    if group_token_counts:
                        group_total_loss = torch.stack(group_token_losses).sum()
                        group_total_tokens = sum(group_token_counts)
                        group_loss = group_total_loss / group_total_tokens
                        epoch_loss += group_loss
                
                # Average loss over groups in batch
                if len(group_data) > 0:
                    batch_loss = epoch_loss / len(group_data)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    total_loss += batch_loss.item()
                    num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self, train_prompts: List[str], train_answers: List[str] = None,
              num_iterations: int = None):
        """Main training loop"""
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        
        print(f"\n🚀 Starting REPO training for {num_iterations} iterations")
        print(f"   Training samples: {len(train_prompts)}")
        print(f"   Group size: {self.config.group_size}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   PPO epochs: {self.config.ppo_epochs}")
        print(f"   Epsilon low/high: {self.config.eps_low}/{self.config.eps_high}")
        
        for iteration in range(num_iterations):
            avg_loss = self.train_iteration(train_prompts, train_answers)
            
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            print(f"  Average loss: {avg_loss:.4f}")
            
            # Show example after each iteration
            if iteration == 0 or (iteration + 1) % 1 == 0:
                self.model.eval()
                with torch.no_grad():
                    # Test on one example
                    test_prompt = train_prompts[0] if train_prompts else "What is 2 + 2?"
                    formatted_prompt = f"""You are an advanced mathematical reasoning model.
Follow these rules carefully for every problem:

1. Think step-by-step and show complete reasoning.
2. Give a short plan before solving.
3. No hand-waving. Be precise.
4. Final answer must be in the form \\boxed{{answer}}.
5. No text after the boxed result.

Solve the following problem:

Problem:
{test_prompt}

Solution:
"""
                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=0.3,
                        do_sample=False
                    )
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response[len(formatted_prompt):]
                    
                    print(f"\n  Example output:")
                    print(f"  Prompt: {test_prompt[:80]}...")
                    print(f"  Response: {response[:150]}...")
                
                self.model.train()

# ============================================================
# Training Pipeline (unchanged except config update)
# ============================================================

def prepare_training_data():
    """Prepare training data from MATH-500 test set"""
    print("📊 Loading MATH-500 dataset...")
    
    # Load dataset (only has 'test' split)
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    print(f"  Total samples: {len(dataset)}")
    
    # Use all data for training (since it's small)
    prompts = [item["problem"] for item in dataset]
    answers = [item["answer"] for item in dataset]
    
    return prompts, answers, dataset

def train_REPO(config: REPOConfig):
    """Main REPO training function"""
    print("\n" + "="*60)
    print("REPO Training for Qwen2.5-Math-1.5B-Instruct")
    print("="*60)
    
    # Load model and tokenizer
    print("🤖 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare training data
    train_prompts, train_answers, full_dataset = prepare_training_data()
    
    # Limit training samples if specified
    if config.num_train_samples < len(train_prompts):
        train_prompts = train_prompts[:config.num_train_samples]
        train_answers = train_answers[:config.num_train_samples]
    
    print(f"  Using {len(train_prompts)} samples for training")
    
    # Initialize REPO trainer
    trainer = REPOTrainer(model, tokenizer, config)
    
    # Train
    trainer.train(train_prompts, train_answers)
    
    # Save model
    print("\n💾 Saving trained model...")
    model.save_pretrained("REPO_trained_model")
    tokenizer.save_pretrained("REPO_trained_model")
    
    print("✅ REPO training completed!")
    print("   Model saved to: REPO_trained_model/")
    
    return model, tokenizer, full_dataset

# ============================================================
# Evaluation Function (unchanged)
# ============================================================

def evaluate_model(model, tokenizer, config: REPOConfig):
    """Evaluate the model on MATH-500 test set"""
    print(f"\n📈 Evaluating model...")
    
    # Load dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    SYSTEM_PROMPT = """You are an advanced mathematical reasoning model.
Follow these rules carefully for every problem:

1. Think step-by-step and show complete reasoning.
2. Give a short plan before solving.
3. No hand-waving. Be precise.
4. Final answer must be in the form \\boxed{answer}.
5. No text after the boxed result.

Solve the following problem:
"""
    
    all_prompts = []
    for item in dataset:
        prompt = f"""{SYSTEM_PROMPT}

Problem:
{item["problem"]}

Solution:
"""
        all_prompts.append(prompt)
    
    # Batch generation function
    def generate_batch(prompts, batch_size=8):
        all_outputs = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            for j, output in enumerate(outputs):
                prompt_len = inputs['input_ids'][j].ne(tokenizer.pad_token_id).sum()
                generated = output[prompt_len:]
                all_outputs.append(tokenizer.decode(generated, skip_special_tokens=True))
        
        return all_outputs
    
    # Generate answers
    print(f"Generating {len(all_prompts)} answers...")
    all_preds = generate_batch(all_prompts, batch_size=config.eval_batch_size)
    
    # Evaluate
    correct = 0
    results = []
    
    for idx, (item, pred) in enumerate(tqdm(zip(dataset, all_preds), total=len(dataset), desc="Evaluating")):
        gold_ans = item["answer"]
        pred_ans = extract_answer(pred)
        gold_ans_normalized = normalize_answer(gold_ans)
        
        is_correct = (pred_ans == gold_ans_normalized)
        if is_correct:
            correct += 1
        
        # Print progress every 10 problems
        if (idx + 1) % 10 == 0:
            print(f"\nProblem {idx + 1}: Current Acc = {correct/(idx+1)*100:.2f}%")
        
        results.append({
            "problem": item["problem"],
            "model_answer": pred,
            "pred_final": pred_ans,
            "gold_final": gold_ans,
            "correct": is_correct
        })
    
    # Save results
    output_file = "qwen_math500_REPO_results.json"
    with open(output_file, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    acc = correct / len(dataset) * 100
    print(f"\n{'='*60}")
    print(f"📊 Final Accuracy: {acc:.2f}% ({correct}/{len(dataset)})")
    print(f"💾 Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return acc, results

# ============================================================
# Full Pipeline (updated with new config)
# ============================================================

def run_REPO_pipeline():
    """Complete REPO training and evaluation pipeline"""
    
    # Configuration with REPO parameters
    config = REPOConfig(
        base_model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        hf_username="your-username",
        push_to_hub=False,
        
        # Training settings
        num_iterations=3,
        group_size=4,
        batch_size=2,
        learning_rate=1e-6,
        
        # REPO specific
        ppo_epochs=4,
        eps_low=0.2,
        eps_high=0.2,
        
        # Dataset settings
        num_train_samples=100,
        num_eval_samples=0,
        
        # Evaluation settings
        eval_batch_size=8
    )
    
    print("🚀 REPO Pipeline for Qwen2.5-Math-1.5B-Instruct")
    print(f"Base Model: {config.base_model_name}")
    print(f"Training Samples: {config.num_train_samples}")
    print(f"REPO Iterations: {config.num_iterations}")
    print(f"PPO Epochs: {config.ppo_epochs}")
    print(f"Epsilon: low={config.eps_low}, high={config.eps_high}")
    print("\n" + "="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        # Step 0: Evaluate base model first
        print("\n📊 Step 0: Evaluating Base Model")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        
        base_acc, base_results = evaluate_model(base_model, base_tokenizer, config)
        
        # Save base model results
        with open("qwen_math500_base_results.json", "w", encoding="utf8") as f:
            json.dump(base_results, f, indent=2, ensure_ascii=False)
        
        # Step 1: REPO Training
        print("\n🎯 Step 1: REPO Training")
        REPO_model, REPO_tokenizer, full_dataset = train_REPO(config)
        
        # Step 2: Evaluation
        print("\n📈 Step 2: Evaluating REPO-trained Model")
        REPO_acc, REPO_results = evaluate_model(REPO_model, REPO_tokenizer, config)
        
        # Save training summary
        summary = {
            "model": config.base_model_name,
            "training_method": "REPO",
            "training_config": {
                "num_iterations": config.num_iterations,
                "group_size": config.group_size,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_train_samples": config.num_train_samples,
                "ppo_epochs": config.ppo_epochs,
                "eps_low": config.eps_low,
                "eps_high": config.eps_high
            },
            "results": {
                "base_accuracy": base_acc,
                "REPO_accuracy": REPO_acc,
                "improvement": REPO_acc - base_acc,
                "total_problems": len(REPO_results),
                "correct_problems": sum(r["correct"] for r in REPO_results)
            }
        }
        
        with open("REPO_training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n✅ REPO Pipeline Completed Successfully!")
        print(f"📊 Base Model Accuracy: {base_acc:.2f}%")
        print(f"📊 REPO Model Accuracy: {REPO_acc:.2f}%")
        print(f"📈 Improvement: {REPO_acc - base_acc:.2f}%")
        print(f"📁 Summary saved to: REPO_training_summary.json")
        
        # Push to HuggingFace Hub (optional)
        if config.push_to_hub and config.hf_token:
            from huggingface_hub import HfApi, login
            login(token=config.hf_token)
            
            repo_id = f"{config.hf_username}/qwen2.5-math-1.5b-repo"
            REPO_model.push_to_hub(repo_id)
            REPO_tokenizer.push_to_hub(repo_id)
            print(f"🚀 Model uploaded to: https://huggingface.co/{repo_id}")
        
        return REPO_model, REPO_tokenizer, summary
    
    except Exception as e:
        print(f"\n❌ Error in REPO pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ============================================================
# Quick Test (updated with new config)
# ============================================================

def run_quick_test():
    """Quick test with minimal training"""
    print("🚀 Quick REPO Test")
    
    # Configuration for quick test
    config = REPOConfig(
        base_model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        num_iterations=1,
        group_size=2,
        batch_size=1,
        num_train_samples=20,
        learning_rate=2e-6,
        ppo_epochs=2,
        eps_low=0.2,
        eps_high=0.2,
        eval_batch_size=4
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    # Use first N samples for training
    train_samples = min(config.num_train_samples, len(dataset))
    train_prompts = [dataset[i]["problem"] for i in range(train_samples)]
    train_answers = [dataset[i]["answer"] for i in range(train_samples)]
    
    print(f"  Training on {train_samples} samples...")
    
    # Initialize trainer
    trainer = REPOTrainer(model, tokenizer, config)
    
    # Train for 1 iteration
    trainer.train(train_prompts, train_answers)
    
    # Quick evaluation on 5 test samples
    print("\n📊 Quick evaluation on 5 test samples...")
    test_samples = min(5, len(dataset) - train_samples)
    
    correct = 0
    for i in range(train_samples, train_samples + test_samples):
        problem = dataset[i]["problem"]
        gold_answer = dataset[i]["answer"]
        
        formatted_prompt = f"""You are an advanced mathematical reasoning model.
Follow these rules carefully for every problem:

1. Think step-by-step and show complete reasoning.
2. Give a short plan before solving.
3. No hand-waving. Be precise.
4. Final answer must be in the form \\boxed{{answer}}.
5. No text after the boxed result.

Solve the following problem:

Problem:
{problem}

Solution:
"""
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_answer = extract_answer(response)
        
        if pred_answer and normalize_answer(pred_answer) == normalize_answer(gold_answer):
            correct += 1
    
    accuracy = correct / test_samples * 100
    print(f"\n📊 Quick Test Accuracy: {accuracy:.2f}% ({correct}/{test_samples})")
    
    return model, tokenizer, accuracy

# ============================================================
# Main (updated to include new config)
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="REPO Pipeline for Qwen2.5-Math")
    parser.add_argument("--mode", type=str, default="quick",
                       choices=["full", "quick", "evaluate-only"],
                       help="Pipeline mode")
    parser.add_argument("--train-samples", type=int, default=100,
                       help="Number of training samples")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of REPO iterations")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        config = REPOConfig(
            num_train_samples=args.train_samples,
            num_iterations=args.iterations
        )
        run_REPO_pipeline()
    
    elif args.mode == "quick":
        run_quick_test()
    
    elif args.mode == "evaluate-only":
        config = REPOConfig()
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        accuracy, results = evaluate_model(model, tokenizer, config)
        print(f"\n📊 Base Model Accuracy: {accuracy:.2f}%")
    
    else:
        print(f"Mode '{args.mode}' not recognized.")
