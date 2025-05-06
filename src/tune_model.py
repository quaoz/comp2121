import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm

# Map string labels to integers
label_to_id = {"SUPPORT": 0, "CONTRADICT": 1, "NO_RELATION": 2}
id_to_label = {v: k for k, v in label_to_id.items()}


# Create dataset class
class ScifactDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Convert labels
        self.dataframe["label_id"] = self.dataframe["label"].map(label_to_id)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        claim = row["claim"]
        evidence = row["evidence"]
        label = row["label_id"]

        # Combine claim and evidence with special tokens
        input_text = claim + " [SEP] " + evidence

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Return as tensors
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Training function
def train_model(
    device, model, train_dataloader, eval_dataloader, epochs=3, learning_rate=2e-5
):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")

        # Evaluation
        print("Evaluating...")
        eval_model(model, eval_dataloader, device)

    return model


# Evaluation function
def eval_model(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")

    # Suppress warnings for undefined metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    return accuracy, f1, all_preds
