import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from typing import List, Tuple
import numpy as np

class MLMScorer:
    def __init__(self, path='bert-base-uncased', max_length=512, wwm=False, add_special=True, device=None):
        self.tokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)
        self.model = BertForMaskedLM.from_pretrained(path, local_files_only=True)
        self.max_length = max_length
        self.wwm = wwm
        self.add_special = add_special
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def _ids_to_masked(self, token_ids: List[int]) -> List[Tuple[List[int], List[int]]]:
        token_ids_masked_list = []
        mask_indices = []

        if self.wwm:
            for idx, token_id in enumerate(token_ids):
                if self.tokenizer.convert_ids_to_tokens([token_id])[0].startswith('##'):
                    mask_indices[-1].append(idx)
                else:
                    mask_indices.append([idx])
        else:
            mask_indices = [[idx] for idx in range(len(token_ids))]

        if self.add_special:
            mask_indices = mask_indices[1:-1]
        else:
            mask_indices = mask_indices[1:]

        mask_token_id = self.tokenizer.mask_token_id
        for mask_set in mask_indices:
            token_ids_masked = token_ids.copy()
            for idx in mask_set:
                token_ids_masked[idx] = mask_token_id
            token_ids_masked_list.append((token_ids_masked, mask_set))

        return token_ids_masked_list

    class MLMDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    @staticmethod
    def collate_fn(batch):
        max_len = max([len(item['input_ids']) for item in batch])
        
        input_ids = [item['input_ids'] + [0] * (max_len - len(item['input_ids'])) for item in batch]
        attention_mask = [item['attention_mask'] + [0] * (max_len - len(item['attention_mask'])) for item in batch]
        
        return {
            'sent_idx': torch.tensor([item['sent_idx'] for item in batch]),
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'mask_pos': torch.tensor([item['mask_pos'] for item in batch]),
            'original_token_id': torch.tensor([item['original_token_id'] for item in batch])
        }

    def corpus_to_dataset(self, corpus: List[str]):
        sents_expanded = []

        for sent_idx, sent in enumerate(corpus):
            encoding = self.tokenizer.encode_plus(
                sent,
                add_special_tokens=self.add_special,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            
            token_ids = encoding['input_ids'].squeeze().tolist()
            attention_mask = encoding['attention_mask'].squeeze().tolist()

            if len(token_ids) > self.max_length:
                logging.error(f"Line #{sent_idx+1} is too long; will output score of 0 and omit in token counts.")
                continue

            ids_masked = self._ids_to_masked(token_ids)

            for masked_ids, mask_set in ids_masked:
                for mask_pos in mask_set:
                    sents_expanded.append({
                        'sent_idx': sent_idx,
                        'input_ids': masked_ids,
                        'attention_mask': attention_mask,
                        'mask_pos': mask_pos,
                        'original_token_id': token_ids[mask_pos]
                    })

        return self.MLMDataset(sents_expanded)

    def score(self, corpus: List[str], batch_size=32, num_workers=4, per_token=False):
        dataset = self.corpus_to_dataset(corpus)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn)

        true_tok_lens = [len(self.tokenizer.encode(sent, add_special_tokens=self.add_special)) for sent in corpus]

        if per_token:
            scores_per_token = [[None] * tok_len for tok_len in true_tok_lens]
        else:
            scores = np.zeros((len(corpus),))

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                mask_pos = batch['mask_pos'].to(self.device)
                original_token_ids = batch['original_token_id'].to(self.device)
                sent_idxs = batch['sent_idx'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs[torch.arange(input_ids.size(0)), mask_pos, original_token_ids]

                if per_token:
                    for sent_idx, score, position in zip(sent_idxs.cpu().numpy(), token_log_probs.cpu().numpy(), mask_pos.cpu().numpy()):
                        scores_per_token[sent_idx][position] = score
                else:
                    np.add.at(scores, sent_idxs.cpu().numpy(), token_log_probs.cpu().numpy())

        if per_token:
            return scores_per_token, true_tok_lens
        else:
            return scores.tolist(), true_tok_lens

    def __call__(self, corpus: List[str], **kwargs):
        return self.score(corpus, **kwargs)

# Initialize MLMScorer
scorer = MLMScorer(path='/usr/scratch/danial_stuff/Chromalyzer/misc/bert-base-uncased')

# Example corpus
corpus = ["This is a sample sentence.", "Another example for scoring.","Hey Hey You You shsuygu"]

# Get scores
scores, token_lengths = scorer(corpus)

print("Scores:", scores)
print("Token lengths:", token_lengths)