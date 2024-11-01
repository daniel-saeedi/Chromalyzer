from helper import *
from MolFormerDistanceScorer import *
import pandas as pd
from rdkit.Chem import rdMolDescriptors
from sklearn.neighbors import KernelDensity
import numpy as np
from rdkit import Chem
from sklearn.model_selection import GridSearchCV
from scipy.special import expit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = '/usr/scratch/danial_stuff/Chromalyzer/MoLFormer-XL-both-10pct/'
model = AutoModel.from_pretrained(PATH, local_files_only=True, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True, trust_remote_code=True)
scorer = MolFormerDistanceScorer(model, tokenizer, batch_size=256)

starting_molecule = 'C1=CC=C2C=CC=CC2=C1'
current_molecule = starting_molecule

# Read peaks data
peaks = pd.read_csv('/usr/scratch/chromalyzer/peaks/combined.csv')
peaks = peaks[peaks['label'] == 0]

# Prepare the data
X = peaks['m_z'].values.reshape(-1, 1)

# Define the bandwidth range to search
bandwidths = np.logspace(-1, 1, 20)

# Create a GridSearchCV object
grid = GridSearchCV(
    KernelDensity(kernel='gaussian'),
    {'bandwidth': bandwidths},
    cv=5,  # 5-fold cross-validation
    n_jobs=-1  # Use all available cores
)

# Fit the grid search object to find the best bandwidth
grid.fit(X)

# Get the best bandwidth
best_bandwidth = grid.best_params_['bandwidth']
print(f"Best bandwidth: {best_bandwidth}")

# Train KDE on m/z column of peaks with the best bandwidth
kde = KernelDensity(kernel='gaussian', bandwidth=best_bandwidth).fit(X)

def calculate_combined_scores(cosine_similarities, molecular_weights, kde):
    # Convert cosine similarities to probabilities
    similarity_probabilities = expit(cosine_similarities)
    
    # Get probabilities from KDE
    kde_probabilities = np.exp(kde.score_samples(np.array(molecular_weights).reshape(-1, 1)))
    
    # Calculate the log ratio
    log_ratios = np.log(similarity_probabilities) + np.log(kde_probabilities)
    
    return log_ratios

for time_step in range(50):

    # if time_step == 1: import code; code.interact(local=locals())
    candidates = generate_mutations(starting_mol=current_molecule)

    # Candidate Cosine Similarity
    scores = scorer.score_batch(current_molecule, candidates)

    # Find the molecular weights of all candidates using Chem.MolFromSmiles
    molecular_weights = []
    for smiles in candidates:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecular_weights.append(rdMolDescriptors.CalcExactMolWt(mol))
        else:
            molecular_weights.append(None)

    # Calculate combined scores using the new method
    combined_scores = calculate_combined_scores(scores, molecular_weights, kde)

    # Print results nicely
    results = pd.DataFrame({
        'starting_molecule': [current_molecule]*len(candidates),
        'SMILES': candidates,
        'Molecular Weight': molecular_weights,
        'Cosine Similarity': scores,
        'KDE Log-Likelihood': kde.score_samples(np.array(molecular_weights).reshape(-1, 1)),
        'Combined Score': combined_scores
    })
    results = results.sort_values('Combined Score', ascending=True).reset_index(drop=True)
    results.to_csv(f'misc/timesteps/{time_step}.csv')

    # Only keep results with Molecular Weight equal or above the current_molecule
    current_mol = Chem.MolFromSmiles(current_molecule)
    current_mol_weight = rdMolDescriptors.CalcExactMolWt(current_mol)
    # results = results[results['Molecular Weight'] > current_mol_weight].sort_values('Combined Score', ascending=True).reset_index(drop=True)

    

    # Handle potential issues with probabilities
    probabilities = np.exp(results['Combined Score'].fillna(-np.inf))  # Convert log ratios back to probabilities
    probabilities = np.clip(probabilities, 0, np.inf)  # Clip to non-negative values
    
    probabilities = probabilities / probabilities.sum()  # Normalize to sum to 1
    current_molecule = np.random.choice(results['SMILES'], p=probabilities)

    # current_molecule = results['SMILES'][0]

    print(f"Time step {time_step}: Current molecule = {current_molecule}")