import torch
from torch.nn.parallel import DataParallel
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch.nn import DataParallel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

class MolFormerDistanceScorer:
    def __init__(self, model, tokenizer, max_length=512, epsilon=1e-8, batch_size=1024, max_gpus=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        
        # Determine the number of GPUs to use
        num_gpus = torch.cuda.device_count()
        num_gpus_to_use = min(num_gpus, max_gpus)
        
        if num_gpus_to_use > 1:
            print(f"Using {num_gpus_to_use} GPUs")
            gpu_ids = list(range(num_gpus_to_use))
            self.model = DataParallel(model, device_ids=gpu_ids).to(self.device)
        else:
            print("Using single GPU or CPU")
            self.model = model.to(self.device)
        
        self.model.eval()
        self.max_length = max_length
        self.epsilon = epsilon  # Small value to avoid log(0)
        self.batch_size = batch_size

    def get_vector_representation(self, sequences):
        # Ensure sequences is a list of strings
        if isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        elif isinstance(sequences, str):
            sequences = [sequences]
        
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # Use the [CLS] token representation as the vector representation of the molecule
            vector_rep = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return vector_rep

    def score_batch(self, target, pool_sequences):
        target_vector = self.get_vector_representation([target])
        
        all_scores = []

        for i in tqdm(range(0, len(pool_sequences), self.batch_size), desc="Scoring batches"):
            batch_sequences = pool_sequences[i:i+self.batch_size]
            
            batch_vectors = self.get_vector_representation(batch_sequences)
            
            # Calculate cosine similarity between target and batch vectors
            similarities = cosine_similarity(target_vector, batch_vectors)
            all_scores.extend(similarities[0])

        return all_scores
# # Example usage
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     PATH = '/usr/scratch/danial_stuff/Chromalyzer/MoLFormer-XL-both-10pct/'
#     model = AutoModel.from_pretrained(PATH, local_files_only=True, trust_remote_code=True).to(device)
#     tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True, trust_remote_code=True)
    
#     # Increase batch size to take advantage of multiple GPUs
#     scorer = MolFormerDistanceScorer(model, tokenizer, batch_size=256)
    
#     target_sequence = 'C1=CC=C2C=CC=CC2=C1'
#     pool_sequences = np.load('/usr/scratch/danial_stuff/Chromalyzer/chem.npy')
#     scores = scorer.score_batch(target_sequence, pool_sequences)
    
#     # Sort sequences by similarity score (highest to lowest)
#     sorted_indices = np.argsort(scores)[::-1]
#     sorted_sequences = pool_sequences[sorted_indices]
#     sorted_scores = np.array(scores)[sorted_indices]

#     # Print top 10 most similar sequences
#     print("Top 10 most similar sequences:")
#     for seq, score in zip(sorted_sequences[:10], sorted_scores[:10]):
#         print(f"Sequence: {seq}")
#         print(f"Similarity Score: {score:.4f}\n")