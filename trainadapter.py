import os
# set environment variables
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer
)
from adapters import AutoAdapterModel, AdapterTrainer
import logging
from dataHelper import get_dataset
import evaluate
import wandb
import argparse

@dataclass
class RunArguments:
    model_name: str = field(default="roberta-base", metadata={"help": "the name of the pretrained model"})
    dataset_name: str = field(default="restaurant_sup", metadata={"help": "the name of the dataset"})
    seed: int = field(default=37, metadata={"help": "random seed"})

@dataclass
class TrainArguments(TrainingArguments):
    learning_rate: float = field(default=2e-5, metadata={"help": "learning rate"})
    num_train_epochs: int = field(default=3, metadata={"help": "training epoch"})
    weight_decay: float = field(default=0.01, metadata={"help": "weight decay"})

    output_dir: str = field(default="./results", metadata={"help": "path to the output"})
    eval_strategy: str = field(default="epoch", metadata={"help": "evaluation strategy, can be steps or epoch"})
    save_strategy: str = field(default="epoch", metadata={"help": "save strategy, can be steps or epoch"})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "training batch size"})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "evaluation batch size"})
    logging_dir: str = field(default="./logs", metadata={"help": "logging directory"})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "load the best model at the end of training"})

    logging_steps: int = field(default=10, metadata={"help": "wandb hyperparameter: logging steps"})
    save_steps: int = field(default=0, metadata={"help": "wandb hyperparameter: save steps"})
    report_to: str = field(default="wandb", metadata={"help": "report to wandb for logging"})

parser = argparse.ArgumentParser(description="Training script arguments")
parser.add_argument("--model_name", type=str, help="the name of the pretrained model")
parser.add_argument("--dataset_name", type=str, help="the name of the dataset")
parser.add_argument("--seed", type=int, help="random seed")
args = parser.parse_args()
# use HfArgumentParser to parse the arguments
train_args = TrainArguments()
run_args = RunArguments()
if args.model_name:
    run_args.model_name = args.model_name
if args.dataset_name:
    run_args.dataset_name = args.dataset_name
if args.seed:
    run_args.seed = args.seed

print(run_args)
print(train_args)

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="w"
)
set_seed(run_args.seed)

# setup wandb
wandb.init(project="NLP_assignment2", name="{}-{}-{}".format(run_args.model_name, run_args.dataset_name, run_args.seed))

logging.info("Starting the training script")

config = AutoConfig.from_pretrained(run_args.model_name, num_labels=6)
tokenizer = AutoTokenizer.from_pretrained(run_args.model_name, use_fast=True)
model = AutoAdapterModel.from_pretrained(run_args.model_name, config=config)
dataset = get_dataset(run_args.dataset_name, "<SEP>")

# add adapter
model.add_classification_head("sentiment", num_labels=6)
model.add_adapter("sentiment")
model.set_active_adapters("sentiment")
model.train_adapter("sentiment")

def map_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

dataset = dataset.map(map_function, batched=True)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    f1_micro = f1_metric.compute(predictions=predictions, references=labels, average="micro")
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "micro_f1": f1_micro["f1"],
        "macro_f1": f1_macro["f1"]
    }

data_collator = DataCollatorWithPadding(tokenizer)

trainer = AdapterTrainer(
    model=model,
    args=train_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()