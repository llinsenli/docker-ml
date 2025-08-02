import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
OUTPUT_DIR = "/app/output/model"
LOG_DIR = "/app/output/logs"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load dataset (SQuAD v1.1)
logger.info("Loading SQuAD v1.1 dataset...")
dataset = load_dataset("squad")

# Load tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocess dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    answers = [a["text"][0] for a in examples["answers"]]  # Take first answer

    # Format input as "question: {question} context: {context}"
    inputs = [f"question: {q} context: {c}" for q, c in zip(questions, contexts)]
    targets = answers

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
logger.info("Preprocessing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir=LOG_DIR,
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=3e-5,
    fp16=True,  # Enable mixed precision for GPU
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Check for checkpoint
checkpoint = get_last_checkpoint(OUTPUT_DIR)
if checkpoint:
    logger.info(f"Resuming from checkpoint: {checkpoint}")
    trainer.train(resume_from_checkpoint=checkpoint)
else:
    logger.info("Starting training from scratch...")
    trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
logger.info(f"Model saved to {OUTPUT_DIR}")