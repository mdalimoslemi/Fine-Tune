import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import OUTPUT_DIR

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        str(OUTPUT_DIR),
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
    return model, tokenizer

def generate_response(model, tokenizer, instruction: str, max_length: int = 512):
    """Generate a response for a given instruction"""
    # Format the input text
    input_text = f"### Instruction: {instruction}\n\n### Response:"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean up the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].strip()
    
    return response

def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    print("Electrical Engineering Assistant loaded! Type 'quit' to exit.")
    
    while True:
        question = input("\nWhat's your electrical engineering question? ")
        if question.lower() == 'quit':
            break
            
        try:
            response = generate_response(model, tokenizer, question)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()