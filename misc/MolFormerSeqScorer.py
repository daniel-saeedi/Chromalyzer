import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm

class MolFormerSeqScorer:
    def __init__(self, model, tokenizer, max_length=512, epsilon=1e-8, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.epsilon = epsilon  # Small value to avoid log(0)
        self.batch_size = batch_size

    def score_batch(self, sequences):
        if isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        sequences = [str(seq) for seq in sequences]

        all_scores = []

        for i in tqdm(range(0, len(sequences), self.batch_size), desc="Scoring batches"):
            batch_sequences = sequences[i:i+self.batch_size]
            
            # Tokenize the batch
            inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            seq_scores = [[] for _ in range(len(batch_sequences))]
            
            for mask_pos in range(1, input_ids.shape[1] - 1):  # Skip [CLS] and [SEP] tokens
                masked_input_ids = input_ids.clone()
                original_token_ids = masked_input_ids[:, mask_pos].clone()
                masked_input_ids[:, mask_pos] = self.tokenizer.mask_token_id
                
                with torch.no_grad():
                    outputs = self.model(masked_input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.last_hidden_state
                
                # Compute token probabilities for the masked position
                token_probs = torch.softmax(torch.matmul(hidden_states[:, mask_pos], self.model.embeddings.word_embeddings.weight.t()), dim=-1)
                
                # Extract probabilities for original tokens
                original_token_probs = token_probs[torch.arange(len(batch_sequences)), original_token_ids]
                
                # Compute log probabilities
                log_probs = torch.log(original_token_probs + self.epsilon).detach().cpu().numpy()
                
                # Add log probabilities to sequence scores
                for seq_idx, log_prob in enumerate(log_probs):
                    if attention_mask[seq_idx, mask_pos].item() == 1:  # Only consider non-padding tokens
                        seq_scores[seq_idx].append(log_prob)
            
            batch_scores = [sum(scores) for scores in seq_scores]
            all_scores.extend(batch_scores)

        return all_scores

# Example usage
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     PATH = '/usr/scratch/danial_stuff/Chromalyzer/MoLFormer-XL-both-10pct/'
#     model = AutoModel.from_pretrained(PATH, local_files_only=True, trust_remote_code=True).to(device)
#     tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True, trust_remote_code=True)
#     scorer = MLMScorer(model, tokenizer, batch_size=32)
#     sequences = np.load('/usr/scratch/danial_stuff/Chromalyzer/chem.npy')
    # scores = scorer.score_batch(sequences)
    # for seq, score in zip(sequences, scores):
    #     print(f"Sequence: {seq}")
    #     print(f"Score: {score:.4f}\n")