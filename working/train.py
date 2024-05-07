import os
import glob
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich import pretty
import logging
import warnings
import subprocess

QUANTIZATION = '4bit'
MODEL_ID = 'mistralai/Mistral-7B-v0.1'
OUTPUT_DIR = os.path.abspath('./output')
DATA_DIR = os.path.abspath('./data')
MAX_STEPS=50
CHECKPOINT_STEPS=10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def removeFiles(dir):
  for root, dirs, files in os.walk(dir, topdown=False):
      for file in files:
          console.print(f"deleted {file}")
          os.remove(os.path.join(root, file))

      # Add this block to remove folders
      for dir in dirs:
          console.print(f"deleted {dir}")
          os.rmdir(os.path.join(root, dir))

if DEVICE == "cuda":
  torch.cuda.empty_cache()

# Configure logging
logging.disable(logging.WARNING) # disable warnings in the logger
warnings.filterwarnings("ignore") # disable warnings in warnings

logging.basicConfig(level=logging.INFO, format="%(message)s",handlers=[RichHandler()])
console = Console()

console.rule(f"{MODEL_ID}")
console.print(f"Using device: {DEVICE}")


# ------ Clear output -------
console.print("")
console.print("Clearing output folder",style="bold magenta")
removeFiles(OUTPUT_DIR)


# ------ Load the model -------
console.print("")
console.print("Loading the base model", style="bold magenta")

# Pre-define quantization configs
# Quantiziation allows us to load a large model when we have limited
# GPU memory

# 4bit
bb_config_4b = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
##########################################

# 8bit
bb_config_8b = BitsAndBytesConfig(
    load_in_8bit=True,
)

def quantization_config(quantization):
    if quantization == "8bit":
        return bb_config_8b
    else:
        return bb_config_4b

if QUANTIZATION == "none":
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
else: 
    model = AutoModelForCausalLM.from_pretrained(
      MODEL_ID, 
      quantization_config=quantization_config(QUANTIZATION),
      device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=False,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)

tokenizer.pad_token = tokenizer.eos_token

# ------ Load the datasets -------
console.print("")
console.print(f"Loading data from [white]{DATA_DIR}[/white]", style="bold magenta")

common_columns = ['Title', 'Description', 'Owner', 'Owner Email', 'Speakers', 'Status', 'Date Submitted']

# read the data files in the data directory
dfs = []
for f in glob.glob(f'{DATA_DIR}/*.xlsx'):
    df = pd.read_excel(f)[common_columns]
    dfs.append(df)

# merge the files
data = pd.concat(dfs)

# set the column names
data = data.set_axis(['title', 'description', 'owner', 'email', 'speakers', 'status','submitted'], axis=1)

# remove null descriptions
data = data.dropna()

# remove non ascii characters
data['title'] = data['title'].str.encode('ascii','ignore').str.decode('ascii')
data['description'] = data['description'].str.encode('ascii','ignore').str.decode('ascii')

dataset = Dataset.from_pandas(data)

dataset = dataset.train_test_split(test_size=0.1)
val_test_dataset = dataset['test'].train_test_split(test_size=0.5)

train_dataset = dataset["train"]
eval_dataset = val_test_dataset["train"]
test_dataset = val_test_dataset["test"]


# ------ Format data for training -------
console.print("")
console.print(f"Format the data from training", style="bold magenta")

def process_prompt(data):
    messages = [
        {
            "role": "user", 
            "content": f"Create me a conference abstract about {data['title']}",
        },
        {
            "role": "assistant", 
            "content": data['description'],
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    result = tokenizer(prompt, return_token_type_ids=True, padding='max_length', max_length=2048, truncation=True)

    # set labels and input ids to the same for self supervised training
    # https://neptune.ai/blog/self-supervised-learning
    result["labels"] = result["input_ids"].copy()
    return result

tokenizer_columns = ["input_ids","token_type_ids","attention_mask","labels"]

tokenized_train_ds = train_dataset.map(process_prompt,num_proc=os.cpu_count())
tokenized_val_ds = eval_dataset.map(process_prompt,num_proc=os.cpu_count())

# Remove unecessary columns
tokenized_train_ds = tokenized_train_ds.remove_columns(train_dataset.column_names)
tokenized_val_ds = tokenized_val_ds.remove_columns(train_dataset.column_names)

data.info()


# ------ Create Lora configuration -------
console.print("")
console.print(f"Prepare model for Lora training", style="bold magenta")

peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    modules_to_save = ["lm_head", "embed_tokens"],
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# ------ Train the model -------
console.print("")
console.print(f"Train the model, writing output to {OUTPUT_DIR}/lora", style="bold magenta")

# Parallelization is possible if system is multi-GPU
if torch.cuda.device_count() > 1: 
    model.is_parallelizable = True
    model.model_parallel = True

# Training configs
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_val_ds,
    args=transformers.TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/lora",
        warmup_steps=5,
        per_device_train_batch_size=2,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=MAX_STEPS,
        learning_rate=2.5e-5,
        logging_steps=CHECKPOINT_STEPS,
        bf16=True if (QUANTIZATION != "8bit") else False,
        fp16=True if (QUANTIZATION == "8bit") else False,
        optim="paged_adamw_8bit",
        logging_dir="/workshop/logs",
        save_strategy="steps",
        save_steps=CHECKPOINT_STEPS,
        evaluation_strategy="steps", 
        eval_steps=CHECKPOINT_STEPS,
        report_to="none",
        do_eval=True,
        gradient_checkpointing_kwargs={"use_reentrant":True},
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Silencing warnings. If using for inference, consider re-enabling.
model.config.use_cache = False 

# Train! 
trainer.train()

console.print("")
console.print("Merge trained parameters with base model", style="bold magenta")

base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        return_dict=True,
        torch_dtype=torch.float16,
        trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load LoRA adapter and merge
ft_model = PeftModel.from_pretrained(base_model, f"{OUTPUT_DIR}/lora/checkpoint-50", torch_dtype=torch.float16, device=DEVICE)
ft_model = ft_model.merge_and_unload()

ft_model.save_pretrained(f"{OUTPUT_DIR}/finetuned", safe_serialization=False) # safe_serialization allows the creation of bin files
tokenizer.save_pretrained(f"{OUTPUT_DIR}/finetuned")

# clear the GPU cache
if DEVICE == "cuda":
  torch.cuda.empty_cache()

console.print("")
console.print(f"Convert model to gguf {OUTPUT_DIR}/final/finetuned-f16.gguf", style="bold magenta")

subprocess.run([
  "mkdir",
  f"{OUTPUT_DIR}/final",
])

subprocess.run([
  "./.venv/bin/python", 
  "/llama_cpp/convert.py", 
  f"{OUTPUT_DIR}/finetuned",
  "--outfile", f"{OUTPUT_DIR}/final/finetuned-f16.gguf",
  "--outtype","f16",
])

console.print("")
console.print("Quantizie model", style="bold magenta")

subprocess.run([
  "/llama_cpp/quantize",
  f"{OUTPUT_DIR}/final/finetuned-f16.gguf",
  f"{OUTPUT_DIR}/final/finetuned-Q4_K_M.gguf",
  "Q4_K_M",
])


console.print("")
console.print(f"Cleanup {OUTPUT_DIR}", style="bold magenta")

removeFiles(f"{OUTPUT_DIR}/finetuned")
removeFiles(f"{OUTPUT_DIR}/lora")
os.remove(f"{OUTPUT_DIR}/final/finetuned-f16.gguf")