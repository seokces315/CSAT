import torch


def create_collate_fn(tokenizer, task_type):
    def collate_fn(batch):
        # Extract input texts and corresponding labels from the batch
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        # Tokenize input texts with dynamic padding
        encoded_texts = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        labels = (
            torch.tensor(labels, dtype=torch.float)
            if task_type == "REG"
            else torch.tensor(labels, dtype=torch.long)
        )

        return {
            "input_ids": encoded_texts["input_ids"],
            "attention_mask": encoded_texts["attention_mask"],
            "labels": labels,
        }

    return collate_fn
