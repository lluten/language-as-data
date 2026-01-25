import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "ai-forever/mGPT" 
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        # device_map="auto", 
        torch_dtype=torch.float16 
    ).to(device)

    header = (
        "English to Slovak Translation:\n\n"
    )

    examples = [
        ("Good morning.", "Dobré ráno."),              # greeting
        ("Thank you very much.", "Ďakujem veľmi pekne."), # politeness
        ("The dog runs.", "Pes beží."),                # simple subject-verb
        ("I do not know.", "Neviem."),                 # negation 
        ("Where is the station?", "Kde je stanica?"),  # question
        ("She is my best friend.", "Ona je moja najlepšia kamarátka."), # gender
        ("He read a book.", "Čítal knihu."),           # past tense
        ("I will go home.", "Pôjdem domov.")           # future tense
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

    complex_pairs = [
        ("The boy runs fast.", "The boy who wears a hat runs fast."),
        ("The dog barks.", "The dog that saw the cat barks."),
        ("The car is fast.", "The car that my father bought is fast."),
        ("The man walks.", "The man wearing a blue coat walks."),
        ("The woman laughs.", "The woman standing by the door laughs.")
    ]

    for p, (sim, com) in enumerate(complex_pairs):
        test_suite.append({"ID": f"Complex_{p}", "Var": "Simple",  "Text": sim})
        test_suite.append({"ID": f"Complex_{p}", "Var": "Complex", "Text": com})

    # negation and modality
    negation_pairs = [
        ("I know the answer.", "I do not know the answer."),
        ("She can swim.", "She cannot swim."),
        ("They are coming.", "They are not coming."),
        ("He likes pizza.", "He does not like pizza."),
        ("We will succeed.", "We will not succeed.")
    ]

    for p, (affirm, neg) in enumerate(negation_pairs):
        test_suite.append({"ID": f"Negation_{p}", "Var": "Affirmative", "Text": affirm})
        test_suite.append({"ID": f"Negation_{p}", "Var": "Negative",  "Text": neg})

    # lexical semantics
    lexical_pairs = [
        ("The boy is happy.", "The boy is joyful."),
        ("He entered the room.", "He left the room."),
        ("The car is fast.", "The car is quick."),
        ("She is smart.", "She is intelligent."),
        ("The child is small.", "The child is tiny.")
    ]

    for p, (base, variant) in enumerate(lexical_pairs):
        test_suite.append({"ID": f"Lexical_{p}", "Var": "Base",    "Text": base})
        test_suite.append({"ID": f"Lexical_{p}", "Var": "Variant", "Text": variant})


    print(f"Testing on {len(test_suite)} instances...")

    results = []
    
    for item in test_suite:
        text = item["Text"]
        
        final_prompt = few_shot_prompt + f"Input: {text}\nOutput:"
        inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=30,      # keep generation short to reduce hallucination
                do_sample=False,        # deterministic (greedy)
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
    df.to_csv("slovak_translation_results_mgpt.csv", index=False)
    print("\nDone.")

if __name__ == "__main__":
    main()