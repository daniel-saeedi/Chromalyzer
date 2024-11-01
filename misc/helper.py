import time 
import selfies
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import RDLogger
import random
import numpy as np
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

RDLogger.DisableLog('rdApp.*')

@lru_cache(maxsize=10000)
def randomize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None
    Chem.Kekulize(mol)
    return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True)

@lru_cache(maxsize=10000)
def sanitize_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)

def get_selfie_chars(selfie):
    return selfie.replace(']', ']\n').split('\n')[:-1]

def mutate_selfie(selfie, max_molecules_len, alphabet):
    chars_selfie = get_selfie_chars(selfie)
    
    while True:
        random_choice = random.choice([1, 2, 3])
        
        if random_choice == 1:
            random_index = random.randint(0, len(chars_selfie))
            random_character = random.choice(alphabet)
            selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]
        elif random_choice == 2:
            random_index = random.randint(0, len(chars_selfie) - 1)
            random_character = random.choice(alphabet)
            selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]
        else:
            if len(chars_selfie) > 1:
                random_index = random.randint(0, len(chars_selfie) - 1)
                selfie_mutated_chars = chars_selfie[:random_index] + chars_selfie[random_index+1:]
            else:
                continue

        selfie_mutated = "".join(selfie_mutated_chars)
        
        try:
            smiles = selfies.decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) <= max_molecules_len and smiles_canon:
                return (selfie_mutated, smiles_canon)
        except:
            pass

def get_mutated_SELFIES(selfie, num_mutations, alphabet):
    str_chars = get_selfie_chars(selfie)
    max_molecules_len = len(str_chars) + num_mutations
    
    for _ in range(num_mutations):
        selfie, _ = mutate_selfie(selfie, max_molecules_len, alphabet)
    
    return selfie

def process_smile(args):
    smile, num_mutations_list, alphabet = args
    selfie = selfies.encoder(smile)
    results = []
    for num_mutations in num_mutations_list:
        mutated_selfie = get_mutated_SELFIES(selfie, num_mutations, alphabet)
        mutated_smile = selfies.decoder(mutated_selfie)
        results.append(mutated_smile)
    return results

def generate_mutations(starting_mol='C1=CC=C2C=CC=CC2=C1', num_random_samples=5000, num_mutation_ls=[1]):
    total_time = time.time()
    
    mol = Chem.MolFromSmiles(starting_mol)
    if mol is None:
        raise Exception('Invalid starting structure encountered')

    start_time = time.time()
    randomized_smile_orderings = [randomize_smiles(starting_mol) for _ in range(num_random_samples)]
    print('Randomized molecules time:', time.time() - start_time)

    alphabet = ['[=C]', '[C]', '[H]', '[Ring1]', '[Ring2]', '[Ring3]', '[Branch1]', '[Branch2]', '[Branch3]', 
                '[=Branch1]', '[=Branch2]', '[=Branch3]', '[=Ring1]', '[=Ring2]', '[=Ring3]']

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=8) as executor:
        all_smiles_collect = list(executor.map(process_smile, [
            (smile, num_mutation_ls, alphabet) for smile in randomized_smile_orderings
        ]))
    all_smiles_collect = [item for sublist in all_smiles_collect for item in sublist]
    print('Mutation obtainment time:', time.time() - start_time)

    start_time = time.time()
    canon_smi_set = set()
    for item in all_smiles_collect:
        _, smi_canon, did_convert = sanitize_smiles(item)
        if smi_canon and did_convert:
            canon_smi_set.add(smi_canon)
    print('Unique mutated structure obtainment time:', time.time() - start_time)

    return list(canon_smi_set)

def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Descriptors.ExactMolWt(mol)
    else:
        return None