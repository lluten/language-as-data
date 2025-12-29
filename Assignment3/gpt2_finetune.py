import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    dataset = load_dataset("glue", "sst2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets.set_format("torch")

    # Model Initialization
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # Metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training Configuration
    training_args = TrainingArguments(
        output_dir="./Assignment3/gpt2-sst2-finetuned/checkpoints",        
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        save_total_limit=2 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Execution
    trainer.train()
    
    # Final Evaluation & Save
    results = trainer.evaluate()
    print(f"Final Validation Accuracy: {results['eval_accuracy']:.4f}")

    model.save_pretrained("./Assignment3/gpt2-sst2-finetuned/gpt2_sst2_model")
    tokenizer.save_pretrained("./Assignment3/gpt2-sst2-finetuned/./gpt2_sst2_model")

if __name__ == "__main__":
    main()