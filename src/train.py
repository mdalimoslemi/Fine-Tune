import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from config import BASE_MODEL, OUTPUT_DIR, TRAINING_CONFIG, DATASET_CONFIG

def load_and_prepare_data():
    """Load and prepare the dataset for training"""
    # Load the datasets
    dataset = load_dataset(
        'json',
        data_files={
            'train': DATASET_CONFIG['train_file'],
            'validation': DATASET_CONFIG['validation_file'],
            'test': DATASET_CONFIG['test_file']
        }
    )
    
    # Convert the data format to what the model expects
    def format_data(example):
        text = f"### Instruction: {example['instruction']}\n\n### Response: {example['response']}"
        return {"text": text}
    
    # Apply the formatting to each split
    formatted_dataset = {}
    for split in dataset.keys():
        # Expand the 'data' field which contains our examples
        expanded = dataset[split].map(
            lambda x: {"instruction": x["data"][0]["instruction"], "response": x["data"][0]["response"]},
            remove_columns=dataset[split].column_names
        )
        # Format the data for training
        formatted_dataset[split] = expanded.map(format_data)
    
    return formatted_dataset

def prepare_model_and_tokenizer():
    """Initialize the model and tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the input texts"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

def main():
    # Load dataset
    dataset = load_and_prepare_data()
    
    # Initialize model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()
    
    # Tokenize datasets
    tokenized_datasets = {}
    for split in dataset:
        tokenized_datasets[split] = dataset[split].map(
            lambda x: tokenize_function(x, tokenizer, TRAINING_CONFIG["max_length"]),
            batched=True,
            remove_columns=dataset[split].column_names
        )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        eval_strategy=TRAINING_CONFIG["evaluation_strategy"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        save_steps=TRAINING_CONFIG["save_steps"],
        fp16=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

if __name__ == "__main__":
    main()