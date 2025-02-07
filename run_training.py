from unsloth import FastLanguageModel
from load_llm import setup_model, get_training_arguments
from llm_config import ModelConfig
from inference import prepare_for_inference, run_inference_with_stats
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer


def generate_instruction(dataset):
    """Get unique class names and combine them with the instruction.
    
    Args:
        dataset: Hugging Face dataset containing 'train' split with 'status' field
        
    Returns:
        str: instruction text to be used with every prompt
    """
    status_list = []
    for i in range(len(dataset['train'])):
        status_list.append(dataset['train'][i]['status'].lower())
    class_names = list(set(status_list))

    classes_str = ", ".join(class_names)
    instruction = f"""You are a thoughtful assistant that does sentiment classification and returns one of the following classes: {classes_str}
    Please only return the class name, not any other text."""
    return instruction


def load_mental_health_dataset(test_size=0.01):
    dataset = load_dataset("csv", data_files="hf://datasets/btwitssayan/sentiment-analysis-for-mental-health/data.csv")
    # Truncate the text in super long statements
    for i in range(len(dataset['train'])):
        if len(dataset['train'][i]['statement']) > 4096:
            dataset['train'][i]['statement'] = dataset['train'][i]['statement'][:4096]
    # Split the dataset into train and test using the split_dataset function
    # Everything is stored in the 'train' partition by default
    dataset = dataset['train'].train_test_split(test_size=test_size)

    return dataset


def format_chat_template(row, tokenizer, instruction) -> str:
    row_json = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": row["statement"]},
        {"role": "assistant", "content": row["status"]},
    ]

    # Add to the dataset inplace
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize = False, add_generation_prompt = False)
    return row


def prepare_train_test_datasets(tokenizer, config):
    dataset = load_mental_health_dataset(config.test_size)
    instruction = generate_instruction(dataset)

    # Create a new text column in the dataset for training
    train_dataset = dataset['train'].map(lambda x: format_chat_template(x, tokenizer, instruction))
    test_dataset = dataset['test']
    return train_dataset, test_dataset, instruction


def setup_trainer(model, tokenizer, train_dataset, config, print_setup_check=False):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = config.max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 4,
        packing = False, # Can make training 5x faster for short sequences.
        args = get_training_arguments(config),
    )

    # Set the trainer up to only train on the loss from the outputs/classifications
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # Check that training is correclty splitting out the instruction and response
    if print_setup_check:
        print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
        space = tokenizer(" ", add_special_tokens = False).input_ids[0]
        print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))

    return trainer



def main():
    config = ModelConfig()
    # Load model and tokenizer
    model, tokenizer = setup_model(config)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    train_dataset, test_dataset, instruction = prepare_train_test_datasets(tokenizer, config)

    # Run inference on test dataset before training
    test_inputs_extracted = prepare_for_inference(test_dataset, instruction)
    run_inference_with_stats(model, tokenizer, test_inputs_extracted, test_dataset)

    # Train the model
    # Set up SFTTrainer with config settings
    trainer = setup_trainer(model, tokenizer, train_dataset, config)

    # Train the model
    FastLanguageModel.for_training(model)
    trainer.train()

    # Test for improvement
    run_inference_with_stats(model, tokenizer, test_inputs_extracted, test_dataset)

    # Save the model
    model_name = config.model_name.split("/")[-1].replace("-bnb-4bit", "")
    save_path = f"sentiment_tuned_{model_name.lower()}"
    model.save_pretrained(save_path)  # Local saving
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()