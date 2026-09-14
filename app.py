import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 1. Page Configuration
st.set_page_config(page_title="EvoScore Scanner", page_icon="🧬", layout="centered")
st.title("🧬 EvoScore: Zero-Shot Scanner")
st.write("Predict the stability of amino acid substitutions using the ESM-2 protein language model.")

# 2. Cache the heavy model so it only loads ONCE
@st.cache_resource
def load_model():
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Automatically use GPU if available, otherwise use CPU for local testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    return tokenizer, model, device

with st.spinner("Loading AI Engine... (This will download 2.6GB on the very first run)"):
    tokenizer, model, device = load_model()

# 3. Sidebar for User Inputs
st.sidebar.header("Input Parameters")
sequence = st.sidebar.text_input("Wild-Type Sequence", value="MQIFVKTLTG").upper()
position = st.sidebar.number_input("Position to Scan", min_value=1, max_value=len(sequence) if sequence else 100, value=4, step=1)

# 4. The Core AI Logic
def scan_position(sequence, position, tokenizer, model, device):
    original_aa = sequence[position - 1]
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
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
            "Substitute": aa,
            "Score": round(llr, 3)
        })
        
    return pd.DataFrame(results).sort_values(by="Score", ascending=False)

# 5. The User Action (Clicking the Button)
if st.button("Run Evolutionary Scan", type="primary"):
    if not sequence:
        st.error("Please enter a valid sequence.")
    elif position > len(sequence):
        st.error("Position out of bounds.")
    else:
        with st.spinner(f"Scoring all 20 mutations at position {position}..."):
            results_df = scan_position(sequence, int(position), tokenizer, model, device)
            
            st.success(f"Scan complete! Original Amino Acid: {sequence[position-1]}")
            
            # Display colorful table
            def color_scores(val):
                color = '#d4edda' if val > 0 else '#f8d7da' if val < 0 else '#fff3cd'
                return f'background-color: {color}; color: black'
            
            st.dataframe(results_df.style.map(color_scores, subset=['Score']), use_container_width=True)
            
            # Provide a one-click CSV download
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results (CSV)", data=csv, file_name=f"EvoScore_Pos{position}.csv", mime="text/csv")