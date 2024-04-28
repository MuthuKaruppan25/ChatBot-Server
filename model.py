from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset

# Load WikiSQL dataset
dataset = load_dataset("wikisql")

# Separate dataset into train, validation, and test
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Load T5 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Preprocessing function
def preprocess_function(examples):
    inputs = [f"translate SQL to English: {query}" for query in examples["question"]]
    targets = [sql["human_readable"] for sql in examples["sql"]]
    return {"input_text": inputs, "target_text": targets}

# Tokenize datasets using AutoTokenizer
train_tokenized = train_dataset.map(preprocess_function, batched=True)
validation_tokenized = validation_dataset.map(preprocess_function, batched=True)

# Debugging: Print lengths of tokenized datasets
print("Length of Training Tokenized:", len(train_tokenized))
print("Length of Validation Tokenized:", len(validation_tokenized))

print(train_tokenized[:5])
# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer)

# Define training arguments
batch_size = 8
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    num_train_epochs=1,
    report_to="tensorboard",
    logging_dir='./logs',
    output_dir='./results',
    overwrite_output_dir=True
)

# Define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized,
    eval_dataset=validation_tokenized,
    tokenizer=tokenizer,
)

# Train model
trainer.train()

# Save trained model
model.save_pretrained("fine_tuned_t5_sql_model")
