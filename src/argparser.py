import argparse


# Function to parse args
def parse_args():
    # Define argparser
    arg_parser = argparse.ArgumentParser(description="CSAT Experiment settings")

    # Add argument
    arg_parser.add_argument("--seed", default=42, type=int)
    arg_parser.add_argument("--dataset", default="CSAT_Kor.csv", type=str)
    arg_parser.add_argument("--test_size", default=0.1, type=float)
    arg_parser.add_argument("--task_type", default="REG", type=str)
    arg_parser.add_argument(
        "--model_id", default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", type=str
    )
    arg_parser.add_argument("--lora_r", default=8, type=int, help="Size of LoRA")
    arg_parser.add_argument(
        "--lora_alpha", default=16, type=int, help="LoRA update strength"
    )
    arg_parser.add_argument(
        "--lora_dropout", default=0.05, type=float, help="Dropout to LoRA layers"
    )
    arg_parser.add_argument(
        "--pooling_type", default="CLS", type=str, help="Method for representation"
    )
    arg_parser.add_argument(
        "--head_type", default="LINEAR", type=str, help="Projection type"
    )
    arg_parser.add_argument("--r", default=8, type=int, help="Reduction scale")
    arg_parser.add_argument("--dropout", default=0.3, type=float, help="Dropout rate")
    arg_parser.add_argument(
        "--criterion_type", default="MSE", type=str, help="Loss function"
    )
    arg_parser.add_argument("--delta", default=1.0, type=float, help="Margin factor")
    arg_parser.add_argument("--gamma", default=2.0, type=float, help="Focusing param")
    arg_parser.add_argument(
        "--project_name", default="CSAT", type=str, help="W&B Project name"
    )
    arg_parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size per device"
    )
    arg_parser.add_argument(
        "--accum_steps", default=2, type=int, help="Number of accumulation steps"
    )
    arg_parser.add_argument(
        "--num_epochs", default=3, type=int, help="Total number of training epochs"
    )
    arg_parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="Initial learning rate"
    )
    arg_parser.add_argument(
        "--scheduler_type", default="linear", type=str, help="LR scheduler type"
    )
    arg_parser.add_argument(
        "--optimizer", default="adamw_torch", type=str, help="Optimizer type"
    )
    arg_parser.add_argument(
        "--early_stopping_patience", default=5, type=int, help="Number of steps to wait"
    )
    arg_parser.add_argument(
        "--early_stopping_threshold", default=5e-4, type=float, help="Min improvement"
    )

    # Return args
    args = arg_parser.parse_args()

    return args
