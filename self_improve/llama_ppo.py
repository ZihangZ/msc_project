from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="/root/autodl-tmp/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="/root/autodl-tmp/Llama-2-7b-chat-hf", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="/root/autodl-tmp/Llama-2-7b-chat-hf", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    epoch_num: Optional[int] = field(default=10, metadata={"help": "the number of train epochs"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adama: Optional[bool] = field(default=True, metadata={"help": "whether to use the adama optimizer"})
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=1000, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
dataset_name = "/root/autodl-tmp/BIG-Bench-Hard/bbh/logical_deduction_three_objects.json"
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
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


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset_json(tokenizer,dataset_name)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def frezee_layer_llama(model,frozen_layer_num):
    layers_to_frozen_name = [f"model.layers.{i}." for i in range(frozen_layer_num)]+["lm_head.weight"]+ ["model.embed_tokens.weight"]
    for name, param in model.named_parameters():
        if param.requires_grad:
            for layer_name in layers_to_frozen_name:
                if layer_name in name:
                    param.requires_grad = False
                    break

# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
    
if script_args.adama:
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

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
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

# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)

# Create the pipeline
model_se = AutoModelForCausalLMWithValueHead.from_pretrained(config.reward_model_name)
tokenizer_se = AutoTokenizer.from_pretrained(config.reward_model_name)
se_generator = pipeline("text2text-generation",
                        model=reward_model_name,
                        device_map={"": current_device},
                        model_kwargs={"load_in_8bit": True},
                        tokenizer=tokenizer,
                        return_token_type_ids=False,
                        do_sample= False,
                        max_length=64,
                        eos_token_id= tokenizer_se.eos_token_id
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable_params = " + str(trainable_params))

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "max_new_tokens": 200,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer_se.eos_token_id,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

def single_se_input(q, r):
    question_text = q
    response_text = r
    prompt_text = f"The question is: {question_text}. The answer is:{response_text}. Is the answer to the question correct?"
    # prompt_text = tokenizer.encode(prompt_text, return_tensors="pt").to(model.pretrained_model.device)
    return prompt_text
def se_reward_number_t5(text_reward):
    text_reward = text_reward.lower().strip()
    text_reward = text_reward.replace('<pad>', '').replace('</s>', '').strip()
    reward_number = 0.
    if "yes" in text_reward:
        reward_number = 1.
    elif "no" in text_reward:
        reward_number = -1.
    return reward_number
for epoch in range(config.epoch_num):
    for batch_num, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if batch_num >= config.steps:
            break

        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute reward score (using the sentiment analysis pipeline)
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        # rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

        texts = [single_se_input(q, r) for q, r in zip(batch["query"], batch["response"])]
        pipeline_output = se_generator(texts)
        print(pipeline_output)
        se_rewards = [torch.tensor(se_reward_number_t5(se_text['generated_text'])) for se_text in pipeline_output]
        print(se_rewards)

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
