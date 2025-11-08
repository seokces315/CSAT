import os
import sys

# Get parent folder path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append to sys.path
sys.path.append(parent_dir)

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

import pandas as pd

import ast


# Custom dataset for CSAT Project
class CSATKorDataset(Dataset):
    def __init__(self, df, task_type):
        self.df = df
        self.task_type = task_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            choices = ast.literal_eval(row["choices"])

            prompt = (
                f"[시험명]\n{row['year']}년 {row['month']}월 {row['exam_type']}\n\n"
                f"[시험 내 문항 위치]\n{row['question_section']}\n\n"
                f"[지문]\n{row['paragraph']}\n\n"
                f"[질문]\n{row['question']}\n{row['passage']}\n\n"
                f"[선택지]\n1. {choices[0]}\n2. {choices[1]}\n3. {choices[2]}\n4. {choices[3]}\n5. {choices[4]}\n\n"
                f"[정답]\n{row['answer']}"
            )

            text = prompt.replace("\nnan", "")

            if self.task_type == "REG":
                label = row["answer_rate"]
            else:
                label = row["difficulty"]

            return {"text": text, "label": label}

        except Exception as e:
            return None


# Function to process month
def process_month(question_num):
    month = question_num[3]
    return int(month[1]) if month[0] == "0" else int(month)


# Function to classify exam's style
def classify_exam(month):
    if month in [3, 4, 5, 7, 10]:
        exam_type = "전국연합학력평가"
    elif month in [6, 9]:
        exam_type = "모의평가"
    else:
        exam_type = "대학수학능력시험"

    return exam_type


# Function to bucketize question number
def bucketize_num(question_num):
    question_num = int(question_num[4])
    return (
        "초반"
        if question_num <= 10
        else (
            "초중반"
            if question_num <= 20
            else (
                "중반"
                if question_num <= 30
                else "중후반" if question_num <= 40 else "후반"
            )
        )
    )


# Function to label answer rate
def rate2label_c(answer_rate):
    return 0 if answer_rate >= 0.7 else 1 if answer_rate >= 0.3 else 2


# Function to label answer rate
def rate2label(answer_rate):
    return 0 if answer_rate >= 0.8 else 1 if answer_rate >= 0.4 else 2


# Function to re-define given data
def load_data(data_path):
    # Load csv file
    csat_kor_df = pd.read_csv(data_path)

    # Data transformation
    csat_kor_df["question_num"] = csat_kor_df["question_num"].map(
        lambda x: x.split("_")[1:] if x.split("_")[0] == "Odd" else x.split("_")
    )
    csat_kor_df["year"] = csat_kor_df["question_num"].map(lambda x: int(x[2]))
    csat_kor_df["month"] = csat_kor_df["question_num"].map(process_month)
    csat_kor_df["exam_type"] = csat_kor_df["month"].map(classify_exam)
    csat_kor_df["paragraph"] = (
        csat_kor_df["paragraph_image_description"].fillna("").astype(str) + " "
    ).str.lstrip() + csat_kor_df["paragraph"]
    csat_kor_df["question_section"] = csat_kor_df["question_num"].map(bucketize_num)
    csat_kor_df["passage"] = (
        csat_kor_df["question_image_description"].fillna("").astype(str) + " "
    ).str.lstrip() + csat_kor_df["passage"]
    csat_kor_df["answer"] = csat_kor_df["answer"].astype(int)
    csat_kor_df["answer_rate"] = csat_kor_df["answer_rate"].map(
        lambda x: round(0.01 * x, 2)
    )
    csat_kor_df["difficulty"] = csat_kor_df["answer_rate"].map(rate2label_c)

    # Reorder DataFrame's columns
    new_columns = [
        "year",
        "month",
        "exam_type",
        "paragraph",
        "question_section",
        "question",
        "passage",
        "choices",
        "answer",
        "answer_rate",
        "difficulty",
    ]
    csat_kor_df = csat_kor_df[new_columns]

    return csat_kor_df


# Function to split the dataset into train, valid, test subsets
def split_data(total_df, test_size, seed):
    # 1. Split total dataset into train, else
    csat_kor_train_df, csat_kor_eval_test_df = train_test_split(
        total_df,
        test_size=2 * test_size,
        random_state=seed,
        stratify=total_df["difficulty"],
    )

    # 2. Split (eval + test) dataset into valid, test
    csat_kor_eval_df, csat_kor_test_df = train_test_split(
        csat_kor_eval_test_df,
        test_size=0.5,
        random_state=seed,
        stratify=csat_kor_eval_test_df["difficulty"],
    )

    return csat_kor_train_df, csat_kor_eval_df, csat_kor_test_df


if __name__ == "__main__":
    # Local vars
    dataset_name = "CSAT_Kor.csv"
    data_path = f"{parent_dir}/data/{dataset_name}"

    # Define CSAT Korean dataset
    csat_kor_df = load_data(data_path=data_path)
    # print(csat_kor_df.iloc[0])
    csat_kor_dataset = CSATKorDataset(df=csat_kor_df, task_type="REG")
    print(csat_kor_dataset[1]["text"])
