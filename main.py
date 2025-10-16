#!/usr/bin/env python
#
# Anonymization via Finetuned FLAN-T5 Model
# Demo Usage Script
#

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from jupyter.libs.dataset import anonymize_text
    import torch
    from dotenv import dotenv_values
except ImportError as e:
    raise e


# main script
def main():
    # environment
    config_env: dict = dotenv_values("localenv")
    OUTPUT_DIR: str = config_env.get("OUTPUT_DIR", "jupyter/flan-finetuned-ita")

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {OUTPUT_DIR}")
    print(f"Using device: {device}\n")

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
        model.to(device)
        model.eval()
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(
            "\nMake sure you have trained the model first by running the Jupyter notebook."
        )
        return

    # Test examples
    test_examples = [
        "Mi chiamo Laura Bianchi e il mio numero è 339-1234567.",
        "Il paziente Marco Rossi, nato il 15/03/1985, abita in Via Roma 25.",
        "Contattare la dottoressa Anna Verdi all'email anna.verdi@ospedale.it",
        "Pagamento con carta 4532-1234-5678-9010 intestata a Paolo Conti.",
        "CF: RSSMRA85M01H501X, residente in Corso Italia 50, Milano.",
        "Inviare documenti a Giulia Ferrari, Via Garibaldi 88, Napoli. Tel: 081-3456789.",
    ]

    print("=" * 100)
    print("PII ANONYMIZATION EXAMPLES")
    print("=" * 100)

    for i, text in enumerate(test_examples, 1):
        anonymized = anonymize_text(text, model, tokenizer)
        print(f"\nExample {i}:")
        print(f"  Originale:    {text}")
        print(f"  Anonimizzato: {anonymized}")

    print("\n" + "=" * 100)

    # Interactive mode
    print("\n\nInteractive Mode - Enter Italian text to anonymize (or 'quit' to exit):")
    print("-" * 100)

    while True:
        user_input = input("\nTesto italiano: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break

        if not user_input:
            continue

        anonymized = anonymize_text(user_input, model, tokenizer)
        print(f"Anonimizzato:   {anonymized}")


# Main entrypoint
if __name__ == "__main__":
    main()
