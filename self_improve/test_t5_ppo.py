import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import LlamaTokenizer
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


model_name = "/root/autodl-tmp/flan-t5-large"
dataset_name = "/root/autodl-tmp/BIG-Bench-Hard/bbh/navigate.json"
log_step = 50
save_step = 2
output_dir = "/root/autodl-tmp/output_model/nevigate_01"
frozen_layer_num = 22
epoch_num = 10
seed_num = 1001
set_seed(seed_num)
# input_min_text_length = 6
# input_max_text_length = 12


def create_and_prepare_dataset(tokenizer, dataset_path):
    # dataset = load_dataset(dataset_name, split="train[:1%]")
    dataset = load_dataset("json",data_files=dataset_path,field="examples")['train']
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(example):
        text_size = input_size()
        example["input_ids"] = tokenizer.encode(example["question"])[:text_size]
        example["query"] = tokenizer.decode(example["input_ids"])
        return example

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format("torch")
    return dataset

def build_dataset_json(tokenizer,dataset_path):

    # load datasets
    ds_train = load_dataset("json",data_files=dataset_path,field="examples")['train']
    # ds=ds.train_test_split(test_size=0.2)
    # ds_train=ds['train']
    ds_train = ds_train.rename_columns({"input": "question"})
    original_columns = ds_train.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = question + "\nLet’ s think step by step."
            tokenized_question = tokenizer(query, padding=True, truncation=True)
            new_examples["query"].append(question)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = ds_train.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    ds.set_format(type="torch")
    return ds

def frezee_layer_t5(model,frozen_layer_num):
    layers_to_frozen_name = [f"decoder.block.{i}." for i in range(frozen_layer_num)]+["encoder.block"]+ ["shared"]+ ["encoder.final_layer_norm"]+ ["decoder.final_layer_norm"]
    for name, param in model.named_parameters():
        if param.requires_grad:
            for layer_name in layers_to_frozen_name:
                if layer_name in name:
                    param.requires_grad = False
                    break

# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    model_name,
    device_map={"": 0}
)

if frozen_layer_num != 0:
    frezee_layer_t5(model,frozen_layer_num)
    
tokenizer = T5Tokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

dataset = build_dataset_json(tokenizer,"/root/autodl-tmp/BIG-Bench-Hard/bbh/navigate.json")
# dataset = create_and_prepare_dataset(tokenizer)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


config = PPOConfig(
    model_name=model_name,
    steps=1500,
    learning_rate=1e-5,
    batch_size=32,
    ppo_epochs=4,
    mini_batch_size=2,
    # gradient_accumulation_steps=2,
    init_kl_coef=0.1,
    target=6,
    horizon=10000,
    gamma=0.99,
    lam=0.95,
    optimize_cuda_cache=True,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=1,
    seed=seed_num,
    log_with='wandb'
)

opt_kwargs={
    "lr": 1.0e-4,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "weight_decay": 1.0e-6,
}

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),**opt_kwargs)

opt_kwargs={
    "T_max": 10000,
    "eta_min": 1.0e-6,
}
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,**opt_kwargs)

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)

generation_kwargs = {
    "max_new_tokens": 200,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
}

# Generate the input for the self-evaluation step with prompting
def single_se_input(q, r):
    question_text = q
    response_text = r
    prompt_text = f"The question is: {question_text}. The answer is:{response_text}. Is the answer to the question correct?"
    # prompt_text = tokenizer.encode(prompt_text, return_tensors="pt").to(model.pretrained_model.device)
    return prompt_text

# def se_reward_text(input_text):
#     se_input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.pretrained_model.device)
#     se_output_ids = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
#     se_output_text = tokenizer.batch_decode(se_output_ids)
#     return

def se_reward_number_t5(text_reward):
    text_reward = text_reward.lower().strip()
    text_reward = text_reward.replace('<pad>', '').replace('</s>', '').strip()
    reward_number = 0.
    if "yes" in text_reward:
        reward_number = 1.
    elif "no" in text_reward:
        reward_number = -1.
    return reward_number

# def se_reward_number(text_reward):
#     text_reward = text_reward.lower().strip()
#     reward_number = 0.
#     if "yes" in text_reward:
#         reward_number = 1.
#     elif "no" in text_reward:
#         reward_number = -1.
#     return reward_number

# Create the pipeline
model_se = T5ForConditionalGeneration.from_pretrained("/root/autodl-tmp/flan-t5-large")
tokenizer_se = T5Tokenizer.from_pretrained("/root/autodl-tmp/flan-t5-large")
se_generator = pipeline("text2text-generation", model=model_se, tokenizer=tokenizer_se,
                    do_sample= False,
                    max_length=64,
                    eos_token_id= tokenizer_se.eos_token_id,
    device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable_params = " + str(trainable_params))


for epoch in range(epoch_num):
    
    for batch_num, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    #     # Generate self-evaluation input
    #     texts = [single_se_input(q, r) for q, r in zip(batch["query"], batch["response"])]
    #     print(texts)
    #     se_inputs_list = tokenizer(texts, padding=True, truncation=True)
    #     se_inputs = se_inputs_list['input_ids']
    #     # se_inputs = tokenizer.encode(texts, return_tensors="pt").to(model.pretrained_model.device)
    #     se_inputs = [torch.tensor(se_input,device = model.pretrained_model.device) for se_input in se_inputs]

    #     # Get text reward for self-evaluation
    #     se_output_ids = ppo_trainer.generate(
    #         se_inputs,
    #         return_prompt=False,
    #         **generation_kwargs,
    #     )
    #     se_texts = tokenizer.batch_decode(se_output_ids, skip_special_tokens=True)
    #     print(se_texts)

    #     # Compute self-evaluation score
    #     se_rewards = [torch.tensor(se_reward_number_t5(se_text)) for se_text in se_texts]
    #     print(se_rewards)

        # Use pipeline to generate reward
        texts = [single_se_input(q, r) for q, r in zip(batch["query"], batch["response"])]
        pipeline_output = se_generator(texts)
        print(pipeline_output)
        se_rewards = [torch.tensor(se_reward_number_t5(se_text['generated_text'])) for se_text in pipeline_output]
        print(se_rewards)

        # # Compute reward score
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(ppo_trainer.accelerator.device)
        # raw_rewards = ppo_trainer.model.compute_reward_score(**inputs)
        # rewards = [raw_rewards[i, -1, 1] for i in range(len(raw_rewards))]  # take last token

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, se_rewards)
        ppo_trainer.log_stats(stats, batch, se_rewards)

        # if epoch and epoch % log_step == 0:
        #     ppo_trainer.log_stats(stats, batch, se_rewards)

    if epoch and (epoch+1) % save_step == 0:
        ppo_trainer.save_pretrained(output_dir + f"epoch_{epoch}")