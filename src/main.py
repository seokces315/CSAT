from argparser import parse_args
from utils import set_seed, supports_bf16
from data import load_data, split_data, CSATKorDataset
from model.model import load_llm
from model.trainer import create_collate_fn
from model.metrics import compute_regression_metrics, compute_classification_metrics

from pathlib import Path

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

import torch

import wandb

import warnings

warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")


# Return the parent directory path
def get_grandparent_dir():
    # Get the current file path
    current_path = Path(__file__)

    # Get the directory two levels above the current file
    grandparent_dir = current_path.parent.parent

    return grandparent_dir


# Main flow
def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Set the device to GPU (CUDA) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose dtype and attention based on BF16 support
    bf16_available = supports_bf16()
    dtype = torch.bfloat16 if bf16_available else torch.float16

    # Load data
    grandparent_path = get_grandparent_dir()
    data_path = f"{grandparent_path}/data/{args.dataset}"
    csat_kor_df = load_data(data_path=data_path)

    # Split data
    csat_kor_train_df, csat_kor_eval_df, csat_kor_test_df = split_data(
        total_df=csat_kor_df, test_size=args.test_size, seed=args.seed
    )

    print(len(csat_kor_train_df))
    print(len(csat_kor_eval_df))
    print(len(csat_kor_test_df))

    # Initialize train, val and test dataset
    csat_kor_train_dataset = CSATKorDataset(
        df=csat_kor_train_df, task_type=args.task_type
    )
    csat_kor_eval_dataset = CSATKorDataset(
        df=csat_kor_eval_df, task_type=args.task_type
    )
    csat_kor_test_dataset = CSATKorDataset(
        df=csat_kor_test_df, task_type=args.task_type
    )

    # Load tokenizer & model
    tokenizer, model = load_llm(
        dtype=dtype,
        task_type=args.task_type,
        model_id=args.model_id,
        gpu_id=args.gpu_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pooling_type=args.pooling_type,
        head_type=args.head_type,
        r=args.r,
        dropout=args.dropout,
        criterion_type=args.criterion_type,
        delta=args.delta,
        gamma=args.gamma,
    )
    model = model.to(device=device, dtype=dtype)

    # Intialize W&B only with project name
    model_id = args.model_id.split("/")[-1].split("-")[0]
    run_name = f"{args.task_type}_{model_id}_{args.pooling_type}_{args.head_type}_{args.criterion_type}"
    wandb.init(
        project=args.project_name,
        name=run_name,
    )

    # Branching by experiment type
    if args.project_name == "CSAT":
        # Define train configurations for the CSAT fine-tuning experiment
        metric_for_best_model = "rmse" if args.task_type == "REG" else "accuracy"
        greater_is_better = False if args.task_type == "REG" else True
        training_args = TrainingArguments(
            output_dir=f"../res/{run_name}",
            data_seed=args.seed,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accum_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.scheduler_type,
            optim=args.optimizer,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            disable_tqdm=False,
            full_determinism=True,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            run_name=run_name,
            label_names=["labels"],
            report_to="wandb",
        )

        # Initialize the Trainer with datasets, metrics, and an early stopping callback
        data_collator = create_collate_fn(tokenizer=tokenizer, task_type=args.task_type)
        compute_metrics = (
            compute_regression_metrics
            if args.task_type == "REG"
            else compute_classification_metrics
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=csat_kor_train_dataset,
            eval_dataset=csat_kor_eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_threshold=args.early_stopping_threshold,
                )
            ],
        )

        # Train the model with early stopping based on evaluation metrics
        trainer.train()

        # Run evaluation and automatically log metrics to W&B
        trainer.evaluate(eval_dataset=csat_kor_test_dataset)
    else:
        # Define train-free configurations for the CSAT fine-tuning experiment
        zero_shot_args = TrainingArguments(
            output_dir=f"../res/{run_name}",
            per_device_eval_batch_size=8,
            disable_tqdm=False,
            full_determinism=True,
            remove_unused_columns=False,
            run_name=run_name,
            label_names=["labels"],
            report_to="wandb",
        )

        # Initialize the Trainer for evaluation-only
        data_collator = create_collate_fn(tokenizer=tokenizer, task_type=args.task_type)
        compute_metrics = (
            compute_regression_metrics
            if args.task_type == "REG"
            else compute_classification_metrics
        )
        trainer = Trainer(
            model=model,
            args=zero_shot_args,
            data_collator=data_collator,
            eval_dataset=csat_kor_test_dataset,
            compute_metrics=compute_metrics,
        )

        # Run evaluation and automatically log metrics to W&B
        trainer.evaluate()

    # Safely end W&B session
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
