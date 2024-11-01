import torch
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained MolFormer model and tokenizer
PATH = '/usr/scratch/danial_stuff/Chromalyzer/MoLFormer-XL-both-10pct/'
model = AutoModel.from_pretrained(PATH, local_files_only=True, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True, trust_remote_code=True)

print('Model loaded!')
def get_sequence_probability(smiles_sequence):
    # Tokenize the input SMILES sequence
    inputs = tokenizer(smiles_sequence, return_tensors='pt',padding=True).to(device)
    tokens = inputs['input_ids'].squeeze()
    token_probs = []

    # For each token, mask it and get the probability of the original token
    for i in range(1, len(tokens) - 1):  # skip special tokens like [CLS] and [SEP] if applicable
        masked_input = tokens.clone()
        masked_input[i] = tokenizer.mask_token_id  # mask the token

        # Get model prediction with masked token
        with torch.no_grad():
            output = model(masked_input.unsqueeze(0), output_hidden_states=True)  # Add batch dimension

        logits = output.hidden_states[0]  # Extract logits from the model's output
        softmax = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get the probability of the original token
        token_prob = softmax[0, i, tokens[i]].item()
        token_probs.append(token_prob)

    # Multiply probabilities to get the sequence probability
    log_probs = torch.log(torch.tensor(token_probs))
    sequence_log_prob = torch.sum(log_probs)
    return sequence_log_prob.item()

# Example SMILES strings
smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]

# Get probability for the first SMILES sequence
prob = get_sequence_probability(smiles)

print(f"Sequence probability for the first SMILES: {prob}")

