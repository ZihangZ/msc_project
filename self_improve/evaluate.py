import json
import os
import sys
from typing import Dict, List
import torch
import random

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed

seed_val = 1000
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

ds = load_dataset("json", data_files="/root/autodl-tmp/BIG-Bench-Hard/bbh/navigate.json",field="examples")['train']
ds_split=ds.train_test_split(test_size=0.2)
prompt_all=ds['input']
prompt_all_new= [prompt.replace('\n', ' ') for prompt in prompt_all]
prompt_all_new=['[{}] Let’ s think step by step.'.format(prompt.replace('\n', ' ')) for prompt in prompt_all_new]
answer_all=ds['target']
prompt_train=ds_split['train']['input']
prompt_test=ds_split['test']['input']
answer_test=ds_split['test']['target']
answer_train=ds_split['train']['target']


prompt_test_new= ['[{}] Let’ s think step by step.'.format(prompt.replace('\n', ' ')) for prompt in prompt_test]
prompt_train_new= ['[{}] Let’ s think step by step.'.format(prompt.replace('\n', ' ')) for prompt in prompt_train]
    

def accuracy(prompts: List[str], outputs: List[str]) -> Dict[str, List[float]]:
    match=[]
        
    for i,prompt in enumerate(prompts):

        index = prompt_all_new.index(prompt)
        if answer_all[index].lower().strip() in outputs[i].lower().strip():
            is_correct=1.0
        else:
            is_correct=0.0
                
        match.append(is_correct)

    return sum(match)/len(match)
    
    # Load the model
print("load base model")
model_se_0 = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("/root/autodl-tmp/flan-t5-large",device_map={"": 0})

    # Load the tokenizer
tokenizer_se = T5Tokenizer.from_pretrained("/root/autodl-tmp/flan-t5-large")

    # Create the pipeline
se_generator= pipeline("text2text-generation", model=model_se_0, tokenizer=tokenizer_se,
                        do_sample= True,
                        top_k= 50,
                        top_p= 0.95,
                        max_length=100,
                        eos_token_id= -1,
                        temperature= 1.0,
                          device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,)

    # Load the model
print("load checkpoint model")
model_se_1 = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("/root/autodl-tmp/output_model/nevigate_01epoch_7",device_map={"": 0})

    # Create the pipeline
se_generator_1 = pipeline("text2text-generation", model=model_se_1, tokenizer=tokenizer_se,
                        do_sample= True,
                        top_k= 50,
                        top_p= 0.95,
                        max_length=100,
                        eos_token_id= -1,
                        temperature= 1.0,
                        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,)

print("generate output from base model")
outputs_0=[]
for i in range(len(prompt_train)):
    
    question=prompt_train[i]
    answer=answer_train[i]
    #print('££££££££££££££££££££££££££',question)
    generation=se_generator(question)[0]['generated_text']
    outputs_0.append(generation)
    print('base model : ',generation)
    #print('============',answer)
    #print(reward_fn(prompts=[question], outputs=[generation]))
    
outputs_1=[]
for i in range(len(prompt_train)):
    
    question=prompt_train[i]
    answer=answer_train[i]
    #print('££££££££££££££££££££££££££',question)
    generation=se_generator_1(question)[0]['generated_text']
    outputs_1.append(generation)
    print('checkpoint output : ',generation)
    #print('============',answer)
    #print(reward_fn(prompts=[question], outputs=[generation]))
    
accuracy_0=accuracy(prompt_train_new,outputs_0)
print(accuracy_0)

accuracy_1=accuracy(prompt_train_new,outputs_1)
print(accuracy_1)