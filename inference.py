import torch
from tqdm import tqdm
from unsloth import FastLanguageModel


def prepare_for_inference(dataset, instruction):
    inputs_extracted = []
    for i in range(len(dataset)):
        inputs_extracted.append(
            [
                {"role": "system", "content": instruction},
                {"role" : "user", "content": dataset[i]['statement']}
            ]
        )
    print('Number of test inputs extracted: ', len(inputs_extracted))
    return inputs_extracted

def get_response(text_output):
    response_start = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    try:
        split_output = text_output[0].split(response_start)
        if len(split_output) < 2 or not split_output[1].strip():
            return "unknown"
        response_string = split_output[1].replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
        if not response_string.strip():
            return "unknown"
        classification = response_string.split()[0].lower()
        return classification
    except (IndexError, AttributeError):
        return "unknown"


def predict_from_messages(messages, tokenizer, dataset_mapped, model, batch_size=32, max_length=1024):
    predicted_outputs = []
    correct_outputs = []
    
    # Process messages in batches
    for i in tqdm(range(0, len(messages), batch_size), desc="Predicting"):
        batch_messages = messages[i:i + batch_size]
        
        # Process all messages in batch at once with padding and truncation
        batch_inputs = tokenizer.apply_chat_template(
            batch_messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
            padding = True,           # Add padding
            truncation = True,        # Add truncation
            max_length = max_length,        # Set max length to match your model's config
        ).to("cuda")

        # Add attention mask to tell model which tokens to ignore
        attention_mask = (batch_inputs != tokenizer.pad_token_id).to("cuda")

        # Generate for entire batch at once with optimized parameters
        batch_outputs = model.generate(
            input_ids = batch_inputs,
            attention_mask = attention_mask,
            max_new_tokens = 64,
            use_cache = True,
            temperature = 0.7,  # Lower temperature for more deterministic outputs in classification
            min_p = 0.05,       # Lower min_p for classification tasks
            do_sample = False,  # Set to False for deterministic outputs in classification
            num_beams = 1,      # No need for beam search in classification
        )
        
        # Decode all outputs in batch
        batch_text_outputs = tokenizer.batch_decode(batch_outputs)

        # Clear memory for this batch
        del batch_inputs, batch_outputs
        torch.cuda.empty_cache()
        
        # Process each output in the batch
        for j, text_output in enumerate(batch_text_outputs):
            try: 
                predicted_outputs.append(get_response([text_output]))
            except Exception as e:
                print(f"Error in get_response for item {i+j}...\n")
                print(text_output)
                print(e)
                predicted_outputs.append("unknown")  # Add fallback value

            correct_output = dataset_mapped[i + j]['status']
            correct_outputs.append(correct_output.lower())

    torch.cuda.empty_cache()

    return predicted_outputs, correct_outputs

def count_correct(predicted_outputs, correct_outputs, original_dataset, print_examples=False):
    correct_count = 0
    for i in range(len(predicted_outputs)):
        if predicted_outputs[i] == correct_outputs[i]:
            correct_count += 1
        # Print every 50th statement
        if print_examples:
            if i % 50 == 0:
                print(original_dataset[i]['statement'], ":  ", predicted_outputs[i], "  ", correct_outputs[i])
    print("Accuracy: ", correct_count / len(predicted_outputs))


def run_inference_with_stats(model, tokenizer, inputs_extracted, dataset, batch_size=32, max_length=1024):
    FastLanguageModel.for_inference(model)
    predicted_outputs, correct_outputs = predict_from_messages(inputs_extracted, tokenizer, dataset, model, batch_size, max_length)
    print("Running inference with stats...\n")
    count_correct(predicted_outputs, correct_outputs, dataset)