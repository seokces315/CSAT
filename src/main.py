from argparser import parse_args
from utils import set_seed
from data import load_data, split_data, CSATKorDataset
from model.model import load_llm

from pathlib import Path

from transformers import TrainingArguments

import torch


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

    # Load data
    grandparent_path = get_grandparent_dir()
    data_path = f"{grandparent_path}/data/{args.dataset}"
    csat_kor_df = load_data(data_path=data_path)

    # Split data
    csat_kor_train_df, csat_kor_eval_df, csat_kor_test_df = split_data(
        total_df=csat_kor_df, test_size=args.test_size, seed=args.seed
    )

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

    # Define train configurations for the CSAT fine-tuning experiment
    metric_for_best_model = "rmse" if args.task_type == "REG" else "accuarcy"
    greater_is_better = False if args.task_type == "REG" else True
    model_id = args.model_id.split("/")[-1]
    training_args = TrainingArguments(
        output_dir="./res",
        data_seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.scheduler_type,
        optim=args.optimizer,
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        disable_tqdm=False,
        full_determinism=True,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        run_name=f"CSAT_{args.task_type}_{model_id}_{args.pooling_type}_{args.head_type}_{args.criterion_type}",
        label_names=["labels"],
        report_to="wandb",
    )

    print("No Prob.!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
