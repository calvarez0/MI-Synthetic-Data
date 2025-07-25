import json

def convert_mi_to_alpaca(input_file, output_file, max_items=100):
    """
    Convert MI training data to Alpaca format for LLaMA fine-tuning.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        max_items (int): Maximum number of items to convert (default 100)
    """
    
    # Load the original dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take first max_items (they're already ranked by MITI score)
    selected_data = data[:max_items]
    
    alpaca_dataset = []
    
    for item in selected_data:
        # Extract fields
        system_prompt = item.get('system_prompt', '')
        context = item.get('question', '').split('Client: ')[0].strip()  # Everything before "Client:"
        client_statement = item.get('question', '').split('Client: ')[-1] if 'Client: ' in item.get('question', '') else item.get('question', '')
        response = item.get('response', '')
        
        # Clean up context (remove "Context: " prefix if present)
        if context.startswith('Context: '):
            context = context[9:]  # Remove "Context: " prefix
        
        # Create the input combining context and client statement
        if context.strip():
            input_text = f"Context: {context.strip()}\n\nClient: {client_statement.strip()}"
        else:
            input_text = f"Client: {client_statement.strip()}"
        
        # Create Alpaca format entry
        alpaca_entry = {
            "instruction": system_prompt,
            "input": input_text,
            "output": response
        }
        
        alpaca_dataset.append(alpaca_entry)
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alpaca_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(alpaca_dataset)} entries to Alpaca format")
    print(f"📁 Saved to: {output_file}")
    
    # Print a sample for verification
    if alpaca_dataset:
        print("\n📋 Sample entry:")
        sample = alpaca_dataset[0]
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Input: {sample['input'][:100]}...")
        print(f"Output: {sample['output'][:100]}...")

if __name__ == "__main__":
    # Usage
    input_file = "mi_training_data.json"  # Change this to your input file path
    output_file = "mi_alpaca_format.json"  # Change this to your desired output path
    
    convert_mi_to_alpaca(input_file, output_file, max_items=100)