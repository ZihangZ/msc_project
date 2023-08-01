from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import LlamaTokenizer
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


model_name = "/root/autodl-tmp/llama-7b"
dataset_name = "/root/autodl-tmp/BIG-Bench-Hard/bbh/logical_deduction_three_objects.json"

input_min_text_length = 6
input_max_text_length = 12


def create_and_prepare_dataset(tokenizer, dataset_path):
    # dataset = load_dataset(dataset_name, split="train[:1%]")
    dataset = load_dataset("json",data_files=dataset_path,field="examples")['train']

    # input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(example):
        # text_size = input_size()
        example["input_ids"] = tokenizer.encode(example["input"])
        example["query"] = tokenizer.decode(example["input_ids"])
        return example

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format("torch")
    return dataset

def build_dataset_json(
    tokenizer,
    dataset_path,
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load datasets
    ds_train = load_dataset("json",data_files=dataset_path,field="examples")['train']
    # ds=ds.train_test_split(test_size=0.2)
    # ds_train=ds['train']
    original_columns = ds_train.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
            "query_origin": [],
        }
        for question in examples["input"]:
            query = "Question: " + question + ". Answer: "
            tokenized_question = tokenizer(query, padding=True, truncation=True)
            new_examples["query"].append(query)
            new_examples["query_origin"].append(question)
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


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map={"": 0},
    peft_config=lora_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

dataset = build_dataset_json(tokenizer,"/root/autodl-tmp/BIG-Bench-Hard/bbh/logical_deduction_three_objects.json")
# dataset = create_and_prepare_dataset(tokenizer)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
)

opt_kwargs={
    "lr": 1.0e-4,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "weight_decay": 1.0e-6,
},
optimizer = torch.optim.AdamW(**opt_kwargs, lr=1)

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
}

# Generate the input for the self-evaluation step with prompting
def single_se_input(q, r):
    question_text = q
    response_text = r
    prompt_text = f"Is the answer to the question correct? If you think the answer is correct, reply yes. If you think the answer is wrong, reply no. Otherwise, reply none. {question_text}{response_text}"
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
    if text_reward == "yes":
        reward_number = 1.
    elif text_reward == "no":
        reward_number = -1.
    return reward_number

def se_reward_number(text_reward):
    text_reward = text_reward.lower().strip()
    reward_number = 0.
    if text_reward == "yes":
        reward_number = 1.
    elif text_reward == "no":
        reward_number = -1.
    return reward_number


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        max_length = 256,
        return_prompt=False,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    # Generate self-evaluation input
    texts = [single_se_input(q, r) for q, r in zip(batch["query"], batch["response"])]
    print(texts)
    se_inputs_list = tokenizer(texts, padding=True, truncation=True)
    se_inputs = se_inputs_list['input_ids']
    # se_inputs = tokenizer.encode(texts, return_tensors="pt").to(model.pretrained_model.device)
    se_inputs = [torch.tensor(se_input,device = model.pretrained_model.device) for se_input in se_inputs]
    
    # Get text reward for self-evaluation
    se_output_ids = ppo_trainer.generate(
        se_inputs,
        max_length = 512,
        return_prompt=False,
        **generation_kwargs,
    )
    se_texts = tokenizer.batch_decode(se_output_ids, skip_special_tokens=True)
    print(se_texts)
    
    # Compute self-evaluation score
    se_rewards = [torch.tensor(se_reward_number_t5(se_text)) for se_text in se_texts]
    print(se_rewards)
    

    # # Compute reward score
    # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    # inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(ppo_trainer.accelerator.device)
    # raw_rewards = ppo_trainer.model.compute_reward_score(**inputs)
    # rewards = [raw_rewards[i, -1, 1] for i in range(len(raw_rewards))]  # take last token

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, se_rewards)
    ppo_trainer.log_stats(stats, batch, se_rewards)
