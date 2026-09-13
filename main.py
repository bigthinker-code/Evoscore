import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_engine(model_name="facebook/esm2_t33_650M_UR50D"):
    print(f"Loading {model_name} onto GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to("cuda").eval()
    return tokenizer, model

def scan_position(sequence, position, tokenizer, model):
    original_aa = sequence[position - 1]
    inputs = tokenizer(sequence, return_tensors="pt").to("cuda")
    inputs["input_ids"][0, position] = tokenizer.mask_token_id

    with torch.no_grad():
        output = model(**inputs)

    raw_scores = output.logits[0, position]
    log_probs = F.log_softmax(raw_scores, dim=-1)
    wt_log_prob = log_probs[tokenizer.convert_tokens_to_ids(original_aa)].item()

    results = []
    for aa in list("ACDEFGHIKLMNPQRSTVWY"):
        aa_id = tokenizer.convert_tokens_to_ids(aa)
        llr = log_probs[aa_id].item() - wt_log_prob
        results.append({
            "Mutation": f"{original_aa}{position}{aa}",
            "Score": round(llr, 3)
        })
        
    df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoScore: Zero-Shot Mutation Scanner")
    parser.add_argument("--sequence", required=True, help="Wild-type protein sequence")
    parser.add_argument("--position", required=True, type=int, help="1-indexed position to scan")
    args = parser.parse_args()

    tokenizer, model = load_engine()
    results_df = scan_position(args.sequence, args.position, tokenizer, model)
    
    print(f"\n--- Results for Position {args.position} ---")
    print(results_df.to_string(index=False))
    results_df.to_csv(f"evoscore_pos{args.position}_results.csv", index=False)
    print(f"\nSaved to evoscore_pos{args.position}_results.csv")