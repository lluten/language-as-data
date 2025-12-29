import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load bloomz-3b model and tokenizer
    model_name = "bigscience/bloomz-3b" 
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16 
    )

    header = (
        "Definition: You are an expert translator. Translate the following English sentence into Slovak.\n\n"
    )

    examples = [
        ("Good morning.", "Dobré ráno."),              # Greeting
        ("Thank you very much.", "Ďakujem veľmi pekne."), # Politeness
        ("The dog runs.", "Pes beží."),                # Simple Subject-Verb
        ("I do not know.", "Neviem."),                 # Negation 
        ("Where is the station?", "Kde je stanica?"),  # Question
        ("She is my best friend.", "Ona je moja najlepšia kamarátka."), # Gender
        ("He read a book.", "Čítal knihu."),           # Past tense
        ("I will go home.", "Pôjdem domov.")           # Future tense
    ]

    # build the few-shot block
    few_shot_prompt = header
    for en, sk in examples:
        few_shot_prompt += f"Input: {en}\nOutput: {sk}\n\n"

    test_suite = []
    
    # present vs past tense
    tense_pairs = [
        ("The girl eats an apple.", "The girl ate an apple."),
        ("I walk to the park.", "I walked to the park."),
        ("He writes a letter.", "He wrote a letter."),
        ("The birds fly south.", "The birds flew south."),
        ("We drink water.", "We drank water.")
    ]
    for p, (pres, past) in enumerate(tense_pairs):
        test_suite.append({"ID": f"Tense_{p}", "Var": "Present", "Text": pres})
        test_suite.append({"ID": f"Tense_{p}", "Var": "Past",    "Text": past})

    # simple vs complex sentences
    complex_pairs = [
        ("The boy runs fast.", "The boy who wears a hat runs fast."),
        ("The dog barks.", "The dog that saw the cat barks."),
        ("The car is fast.", "The car that my father bought is fast."),
        ("The woman laughs.", "The woman standing by the door laughs.")
    ]
    for p, (sim, com) in enumerate(complex_pairs):
        test_suite.append({"ID": f"Complex_{p}", "Var": "Simple",  "Text": sim})
        test_suite.append({"ID": f"Complex_{p}", "Var": "Complex", "Text": com})

    print(f"Testing on {len(test_suite)} instances...")

    results = []
    
    for item in test_suite:
        text = item["Text"]
        
        final_prompt = few_shot_prompt + f"Input: {text}\nOutput:"
        inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=30,      # Keep generation short to reduce hallucination
                do_sample=False,        # Deterministic (Greedy)
                repetition_penalty=1.2, # penalize repeating the input
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # get only the generated translation part
        raw_translation = full_text[len(final_prompt):].strip()
        
        # stop if model generates newlines or "Input:"
        clean_translation = raw_translation.split('\n')[0]
        clean_translation = clean_translation.split('Input:')[0]
        
        print(f"[{item['Var']}] {text} -> {clean_translation}")
        
        results.append({
            "ID": item["ID"],
            "Variation": item["Var"],
            "English": text,
            "Slovak": clean_translation
        })

    # save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("slovak_translation_results_bloomz-3b.csv", index=False)
    print("\nDone.")

if __name__ == "__main__":
    main()