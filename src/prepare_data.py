import json
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split

def load_electrical_engineering_data(data_dir: str) -> List[Dict]:
    """
    Load and process electrical engineering QA pairs.
    This is a placeholder - you would need to add your actual data loading logic.
    """
    # Example data structure
    data = [
        {
            "instruction": "Explain Ohm's Law",
            "response": "Ohm's Law states that the current flowing through a conductor is directly proportional to the voltage across the conductor and inversely proportional to its resistance. It is expressed as V = IR, where V is voltage in volts, I is current in amperes, and R is resistance in ohms.",
            "category": "basic_principles"
        },
        # Add more examples here
    ]
    return data

def format_data_for_training(data: List[Dict]) -> List[Dict]:
    """Convert data into format suitable for model training"""
    formatted_data = []
    for item in data:
        formatted_data.append({
            "text": f"### Instruction: {item['instruction']}\n\n### Response: {item['response']}"
        })
    return formatted_data

def split_and_save_data(data: List[Dict], output_dir: Path):
    """Split data into train/validation/test sets and save"""
    # Create splits (70% train, 15% validation, 15% test)
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    for split_name, split_data in [
        ("train", train_data),
        ("validation", val_data),
        ("test", test_data)
    ]:
        with open(output_dir / f"{split_name}.json", 'w') as f:
            json.dump({"data": split_data}, f, indent=2)

def main():
    # Load raw data
    data = load_electrical_engineering_data("raw_data")
    
    # Format data for training
    formatted_data = format_data_for_training(data)
    
    # Split and save data
    split_and_save_data(formatted_data, Path("../data"))

if __name__ == "__main__":
    main()