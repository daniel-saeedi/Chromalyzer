from helper import *
from MolFormerDistanceScorer import *
import pandas as pd
from rdkit.Chem import rdMolDescriptors
from sklearn.neighbors import KernelDensity
import numpy as np
from rdkit import Chem
from sklearn.model_selection import GridSearchCV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = '/usr/scratch/danial_stuff/Chromalyzer/MoLFormer-XL-both-10pct/'
model = AutoModel.from_pretrained(PATH, local_files_only=True, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True, trust_remote_code=True)
scorer = MolFormerDistanceScorer(model, tokenizer, batch_size=256)


molecules = ['C1=CC=CC=C1','C1=CC=C2C=CC=CC2=C1','C1=CC=C(C=C1)C2=CC=CC=C2','C1=CC=C2C(=C1)C=CC3=CC=CC=C32','C1=CC=C2C=C3C=CC=CC3=CC2=C1',
'C1CC2=CC=CC3=C2C1=CC=C3','C1=CC=C(C=C1)C2=CC=CC3=CC=CC=C32',
'C1=CC2=C3C(=C1)C=CC4=CC=CC(=C43)C=C2','C1=CC=C2C(=C1)C3=CC=CC4=C3C2=CC=C4','C1=CC=C2C(=C1)C=CC3=C2C=CC4=CC=CC=C43',
'C1=CC=C2C(=C1)C3=CC=CC=C3C4=CC=CC=C24',
'C1=CC=C2C(=C1)C=CC3=CC4=CC=CC=C4C=C32'
]

molecules_name = [
    '(0) - Benzene','(1) - Naphthalene', '(4) - Biphenyl', '(12) - Phenanthrene', '(13) - Anthracene', '(6) - acenaphthene', '(15) - Naphthalene, 1-phenyl-', '(20) - Pyrene', '(19) - Fluoranthene', '(27) - Chrysene',
    '(27) - triphenylene', '(26) - benz(a)anthracene'
]

# Read peaks data
peaks = pd.read_csv('/usr/scratch/chromalyzer/peaks/combined.csv')
peaks = peaks[peaks['label'] == 1]
# peaks = peaks[(peaks['RT1_center'] >= 4000) & (peaks['RT1_center'] <= 6750) & (peaks['RT2_center'] >= 1.5) & (peaks['RT2_center'] <= 2.3)]

#Find the best bandwidth
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

combined_score_per_molecule = []

i = 0
for current_molecule in molecules:
    scores = scorer.score_batch(current_molecule, molecules)

    # Find the molecular weights of all molecules using Chem.MolFromSmiles
    molecular_weights = []
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecular_weights.append(rdMolDescriptors.CalcExactMolWt(mol))
        else:
            molecular_weights.append(None)

    # Compute NLL of each candidate in trained KDE
    nll_scores = -kde.score_samples(np.array(molecular_weights).reshape(-1, 1))

    # Calculate the NLL of current_molecule
    #TODO
    # Then subtract each nll_scores with this current_molecule NLL.


    # For each candidate, compute Cosine Similarity/(NLL from KDE)
    combined_scores = scores / nll_scores

    # Print results nicely
    results = pd.DataFrame({
        'starting_molecule': [current_molecule]*len(molecules),
        'end_point': molecules,
        'end_point_name': molecules_name,
        'Molecular Weight': molecular_weights,
        'Cosine Similarity': scores,
        'NLL Score': nll_scores,
        'Combined Score': combined_scores,
    })

    # Sort ascending Combined Score
    results = results.sort_values('Combined Score', ascending=False)
    results.to_csv(f'misc/results/{i}.csv')

    i += 1

    combined_score_per_molecule.append(combined_scores)

    # print(results)


import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def create_molecule_graph(molecules, molecules_name, start_index, scores, threshold=0.1):
    G = nx.Graph()
    
    # Add nodes
    for i, mol in enumerate(molecules):
        img = Draw.MolToImage(Chem.MolFromSmiles(mol), size=(100, 100))
        G.add_node(i, image=img, name=molecules_name[i])
    
    # Add edges from start_index to other nodes
    for j, score in enumerate(scores):
        if j != start_index and score > threshold:
            G.add_edge(start_index, j, weight=score)
    
    # Use spring layout to position nodes
    pos = nx.spring_layout(G, k=0.7, iterations=50)
    
    return G, pos

def draw_molecule_graph(G, pos, start_index, output_file):
    plt.figure(figsize=(20, 20))
    
    # Get edge weights
    weights = [G[start_index][v]['weight'] for v in G[start_index]]
    
    # Normalize weights for edge widths
    if weights:
        max_weight = max(weights)
        min_weight = min(weights)
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
        
        # Scale edge widths (adjust the multiplier to change the maximum width)
        edge_widths = [1 + 5 * nw for nw in normalized_weights]
    else:
        edge_widths = []
    
    # Draw edges with width based on weight
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(start_index), width=edge_widths, alpha=0.7, edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=['red' if n == start_index else 'white' for n in G.nodes()], edgecolors='black')
    
    # Draw node images and labels
    ax = plt.gca()
    for n in G.nodes():
        img = G.nodes[n]['image']
        imagebox = OffsetImage(img, zoom=0.5)
        ab = AnnotationBbox(imagebox, pos[n], pad=0.0, frameon=False)
        ax.add_artist(ab)
        
        # Add molecule name as label
        plt.text(pos[n][0], pos[n][1]-0.08, G.nodes[n]['name'], ha='center', va='center', wrap=True, fontsize=8)
    
    # Add edge labels (combined scores)
    edge_labels = {(start_index, v): f'{G[start_index][v]["weight"]:.4f}' for v in G[start_index]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"Connections from {G.nodes[start_index]['name']}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph saved as {output_file}")

# Generate graphs for each molecule
for i, mol in enumerate(molecules):
    G, pos = create_molecule_graph(molecules, molecules_name, i, combined_score_per_molecule[i])
    output_file = f'molecule_graph_{i}_{molecules_name[i].replace(" ", "_")}.png'
    draw_molecule_graph(G, pos, i, output_file)
    
    # Print some statistics about the graph
    print(f"\nStatistics for {molecules_name[i]}:")
    print(f"Number of connections: {len(G[i])}")
    if G[i]:
        print(f"Minimum edge weight: {min(G[i][v]['weight'] for v in G[i]):.4f}")
        print(f"Maximum edge weight: {max(G[i][v]['weight'] for v in G[i]):.4f}")
    else:
        print("No connections above threshold.")