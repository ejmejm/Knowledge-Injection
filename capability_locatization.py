import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt


def load_llama_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the Llama 3.2-1B model with flash attention and bf16 precision.
    
    Returns:
        Tuple containing the model and tokenizer.
    """
    model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )
    
    return model, tokenizer


def calculate_weight_importance(model: AutoModelForCausalLM, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Calculates the weight importance for a single sample.
    
    Args:
        model: The Llama model.
        input_ids: The input tensor.
    
    Returns:
        A dictionary of weight importances for each parameter.
    """
    model.zero_grad()
    output = model(input_ids)
    loss = output.logits.norm()
    loss.backward()
    
    importance = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            importance[name] = param.grad.abs().detach().cpu()
    
    return importance


def process_dataset(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: List[str]) -> Dict[str, torch.Tensor]:
    """
    Processes a dataset and calculates average weight importances.
    
    Args:
        model: The Llama model.
        tokenizer: The tokenizer.
        prompts: List of prompts to process.
    
    Returns:
        A dictionary of average weight importances for each parameter.
    """
    total_importance = {}
    count = 0
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        importance = calculate_weight_importance(model, input_ids)
        
        for name, imp in importance.items():
            if name not in total_importance:
                total_importance[name] = imp
            else:
                total_importance[name] += imp
        
        count += 1
    
    # Calculate average
    for name in total_importance:
        total_importance[name] /= count
    
    return total_importance


def save_results(filename: str, data: Dict[str, torch.Tensor]):
    """Saves the results to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_results(filename: str) -> Dict[str, torch.Tensor]:
    """Loads the results from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def calculate_difference(set1: Dict[str, torch.Tensor], set2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Calculates the difference between two sets of weight importances."""
    diff = {}
    for name in set1:
        diff[name] = set1[name] - set2[name]
    return diff


def visualize_feature_importances(importances: Dict[str, torch.Tensor], title: str):
    """
    Visualizes feature importances for the model.
    
    Args:
        importances: Dictionary of weight importances.
        title: Title for the plot.
    """
    plt.figure(figsize=(20, 30))
    plt.title(title)

    layer_order = [
        'model.layers.{}.input_layernorm',
        'model.layers.{}.self_attn',
        'model.layers.{}.mlp',
        'model.layers.{}.post_attention_layernorm'
    ]

    num_layers = max([int(k.split('.')[2]) for k in importances.keys() if k.startswith('model.layers')]) + 1
    current_y = 0

    # Visualize embed_tokens at the beginning
    weights = importances['model.embed_tokens.weight']
    feature_imp = weights.mean(dim=0)
    plt.imshow(feature_imp.unsqueeze(0), aspect='auto', extent=[0, feature_imp.size(0), current_y, current_y+1])
    plt.text(-0.5, current_y+0.5, 'embed', va='center', ha='right')
    current_y += 1

    for layer in range(num_layers):
        for layer_template in layer_order:
            layer_name = layer_template.format(layer)
            layer_weights = [v for k, v in importances.items() if k.startswith(layer_name)]
            
            for weights in layer_weights:
                feature_imp = weights.mean(dim=0)
                plt.imshow(feature_imp.unsqueeze(0), aspect='auto', extent=[0, feature_imp.size(0), current_y, current_y+1])
                plt.text(-0.5, current_y+0.5, f'L{layer}_{layer_name.split(".")[-1]}', va='center', ha='right')
                current_y += 1

    # Visualize embed_tokens at the end (shared with head)
    weights = importances['model.embed_tokens.weight']
    feature_imp = weights.mean(dim=0)
    plt.imshow(feature_imp.unsqueeze(0), aspect='auto', extent=[0, feature_imp.size(0), current_y, current_y+1])
    plt.text(-0.5, current_y+0.5, 'head', va='center', ha='right')

    plt.colorbar(label='Average Importance')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Example prompts
    questions = [
        "Who was the first president of the United States?",
        "What is the capital of France?",
        "When was the Declaration of Independence signed?",
    ]
    
    statements = [
        "George Washington played a crucial role in the founding of the United States.",
        "The Eiffel Tower is an iconic landmark in a major European city.",
        "The American Revolution led to the creation of a significant document in 1776.",
    ]
    
    # Load model and tokenizer
    model, tokenizer = load_llama_model()
    
    # Process datasets
    print("Processing question set...")
    question_importances = process_dataset(model, tokenizer, questions)
    save_results('question_importances.pkl', question_importances)
    
    print("Processing statement set...")
    statement_importances = process_dataset(model, tokenizer, statements)
    save_results('statement_importances.pkl', statement_importances)
    
    # Calculate and save differences
    print("Calculating differences...")
    differences = calculate_difference(question_importances, statement_importances)
    save_results('importance_differences.pkl', differences)
    
    print("Results saved. You can now analyze the weight importances and their differences.")

    # Load the saved results
    question_importances = load_results('question_importances.pkl')
    statement_importances = load_results('statement_importances.pkl')
    differences = load_results('importance_differences.pkl')

    # Visualize the results
    visualize_feature_importances(question_importances, "Question Importances")
    visualize_feature_importances(statement_importances, "Statement Importances")
    visualize_feature_importances(differences, "Importance Differences")

    print("Visualizations saved as PNG files.")


if __name__ == "__main__":
    main()
