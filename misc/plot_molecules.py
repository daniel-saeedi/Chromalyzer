import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# Read CSV data
# df = pd.read_csv('misc/timesteps/0.csv')

SMILES_list = ['C1=CC2C#CCC2C=1'
]

# SMILES_list = df['SMILES'].tolist()[:100]

# Create molecules from SMILES
mols = [Chem.MolFromSmiles(smiles) for smiles in SMILES_list]

# Generate 2D depictions of the molecules
img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200,200), legends=SMILES_list)

# Save the image
img.save('molecule_grid.png')

print("Image saved as 'molecule_grid.png'")