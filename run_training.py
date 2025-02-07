from load_llm import setup_model, get_training_arguments
from llm_config import ModelConfig
from inference import count_correct, predict_from_messages, get_response
from datasets import load_dataset
from trl import SFTTrainer




def get_class_names(dataset):
    """Get unique class names from dataset.
    
    Args:
        dataset: Hugging Face dataset containing 'train' split with 'status' field
        
    Returns:
        list: Unique class names in lowercase
    """
    status_list = []
    for i in range(len(dataset['train'])):
        status_list.append(dataset['train'][i]['status'].lower())
    class_names = list(set(status_list))
    return class_names





def load_mental_health_dataset():
    dataset = load_dataset("csv", data_files="hf://datasets/btwitssayan/sentiment-analysis-for-mental-health/data.csv")
    # Truncate the text in super long statements
    for i in range(len(dataset['train'])):
        if len(dataset['train'][i]['statement']) > 4096:
            dataset['train'][i]['statement'] = dataset['train'][i]['statement'][:4096]
    # Split the dataset into train and test using the split_dataset function
    # Everything is stored in the 'train' partition by default
    dataset = dataset['train'].train_test_split(test_size=0.02)
    return dataset


def prepare_train_test_datasets(dataset):
    

def main():
    config = ModelConfig()
    # Load model and tokenizer
    model, tokenizer = setup_model(config)
    
    # Load dataset
    dataset = load_dataset("csv", data_files="mental_health_dataset.csv")["train"]
    
    # Prepare dataset
    dataset = prepare_dataset(dataset, tokenizer)
