from transformers import (
AutoModelWithLMHead,
AutoConfig,
Trainer,
AutoTokenizer,
TextDataset,
DataCollatorForLanguageModeling,
TrainingArguments)

def train_model(text_path, output_dir, epochs, model_name, batch_size, cache_dir = "cache"):

    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = TextDataset(
      tokenizer=tokenizer,
      file_path=text_path,
      block_size=256
    )
    
    training_args =TrainingArguments(
    output_dir="trained_model",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=500,
    save_steps=2000,
    logging_steps=10,
    prediction_loss_only=True
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()


