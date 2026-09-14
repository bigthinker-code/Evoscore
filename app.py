import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os

# 1. Hide default Streamlit UI to let your HTML take over
st.set_page_config(page_title="EvoScore Scanner", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding: 0rem; max-width: 100%;}
    </style>
""", unsafe_allow_html=True)

# 2. Cache the AI engine
@st.cache_resource
def load_model():
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    return tokenizer, model, device

with st.spinner("Warming up ESM-2 Engine..."):
    tokenizer, model, device = load_model()

# 3. Connect to your 'frontend' folder
parent_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(parent_dir, "frontend")
evoscore_ui = components.declare_component("evoscore_ui", path=frontend_dir)

# 4. Initialize Data Trackers
if "results" not in st.session_state:
    st.session_state.results = []
if "results_nonce" not in st.session_state:
    st.session_state.results_nonce = None
if "error" not in st.session_state:
    st.session_state.error = None
if "sequence" not in st.session_state:
    st.session_state.sequence = "MQIFVKTLTG"
if "index" not in st.session_state:
    st.session_state.index = 4

# 5. Render your HTML interface
component_value = evoscore_ui(
    sequence=st.session_state.sequence,
    index=st.session_state.index,
    results=st.session_state.results,
    results_nonce=st.session_state.results_nonce,
    error=st.session_state.error,
    height=1000,
    key="evo_bridge"
)

# 6. Listen for the user clicking "Execute Pipeline"
if component_value and component_value.get("action") == "scan":
    incoming_nonce = component_value.get("nonce")
    
    # Only run the math if this is a new request
    if incoming_nonce != st.session_state.results_nonce:
        seq = component_value.get("sequence", "").upper()
        idx = component_value.get("index")
        
        st.session_state.sequence = seq
        st.session_state.index = idx
        
        try:
            wt_aa = seq[idx - 1]
            inputs = tokenizer(seq, return_tensors="pt").to(device)
            inputs["input_ids"][0, idx] = tokenizer.mask_token_id

            with torch.no_grad():
                output = model(**inputs)

            raw_scores = output.logits[0, idx]
            log_probs = F.log_softmax(raw_scores, dim=-1)
            wt_log_prob = log_probs[tokenizer.convert_tokens_to_ids(wt_aa)].item()

            results = []
            for aa in list("ACDEFGHIKLMNPQRSTVWY"):
                if aa == wt_aa:
                    results.append({"mutation": f"{wt_aa}{idx}{aa}", "substitute": aa, "score": 0.0, "wt": True})
                else:
                    aa_id = tokenizer.convert_tokens_to_ids(aa)
                    llr = log_probs[aa_id].item() - wt_log_prob
                    results.append({
                        "mutation": f"{wt_aa}{idx}{aa}",
                        "substitute": aa,
                        "score": round(llr, 3),
                        "wt": False
                    })
            
            st.session_state.results = sorted(results, key=lambda x: x["score"], reverse=True)
            st.session_state.error = None
            
        except Exception as e:
            st.session_state.error = str(e)
            st.session_state.results = []
            
        # Confirm completion and send data back to HTML
        st.session_state.results_nonce = incoming_nonce
        st.rerun()