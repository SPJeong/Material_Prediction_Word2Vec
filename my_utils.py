##### my_utils.py

import pandas as pd
from rdkit import Chem
import numpy as np

'''
Word2Vec model & parameters download
# !wget https://raw.githubusercontent.com/samoturk/mol2vec/master/examples/models/model_300dim.pkl # pre-trained mol2vec model download
# !pip install git+https://github.com/samoturk/mol2vec
from mol2vec import features 
from mol2vec import helpers
'''

from features import mol2alt_sentence, MolSentence, DfVec, featurize
from helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg




def sentences2vec(sentences, model, unseen=None):
    """
    Generate vectors for each sentence (list) in a list of sentences. Vector is simply a sum of vectors for individual words.
    Parameters
    sentences : list, array (List with sentences)
    model : word2vec.Word2Vec (Gensim word2vec model)
    unseen : None, str (Keyword for unseen words. If None, those words are skipped)

    Returns =>  np.array
    """
    keys = set(model.wv.index_to_key)
    vec = []
    if unseen:
        unseen_vec = model.wv.word_vec(unseen)

    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.word_vec(y) if y in set(sentence) & keys
                            else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.word_vec(y) for y in sentence
                            if y in set(sentence) & keys]))  # vocab.keys() -> index_to_key
    return np.array(vec)




# Assuming the following functions and object are imported/defined elsewhere

def convert_smiles_to_mol2vec(df, smiles_col, w2vmodel):
    """
    Converts SMILES strings in a DataFrame column to RDKit Mol objects,
    then to MolSentence objects, and finally to Mol2vec vectors.
    Rows with invalid SMILES strings are dropped.
    Args:
        df (pd.DataFrame): The input DataFrame containing SMILES strings.
        smiles_col (str): The name of the column in the DataFrame that holds the SMILES strings.
        w2vmodel: The pre-trained Word2Vec model object required by sentences2vec.

    """
    # 1. SMILES -> Mol Conversion
    mols = []
    indices_to_drop = []

    print("Starting SMILES to Mol conversion...")
    for i, smiles in enumerate(df[smiles_col]):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            mol = None
            print(f"Error processing SMILES at original index {i}: {smiles}. Error: {e}")

        if mol is not None:
            mols.append(mol)
        else:
            print(f'Index: {i}, smiles: {smiles} is failed for mol conversion. Dropping row.')
            indices_to_drop.append(i)


    # A cleaner approach is to use the original index for alignment:
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in df[smiles_col]]

    # Filter out None values and record indices to drop
    mols = []
    indices_to_drop = []
    for i, mol in enumerate(mol_list):
        if mol is not None:
            mols.append(mol)
        else:
            indices_to_drop.append(df.index[i])  # Get the actual index label from the DataFrame

    # Create a copy to avoid SettingWithCopyWarning and add 'mol' column
    df_processed = df.copy()
    df_processed['mol_temp'] = mol_list

    # 2. Remove failed Mol conversion rows
    if indices_to_drop:
        print(f"Dropping {len(indices_to_drop)} rows due to failed Mol conversion.")
        df_processed = df_processed.drop(indices_to_drop)


    df_processed = df_processed.rename(columns= {'mol_temp': 'mol'})
    df_processed = df_processed.reset_index(drop= True)

    # 3. Mol -> Sentence -> Vector Conversion
    # The value 1 in mol2alt_sentence corresponds to the radius parameter 'r'
    print("Starting Mol to Sentence conversion (MolSentence)...")
    df_processed['sentence'] = df_processed['mol'].apply(lambda mol: MolSentence(mol2alt_sentence(mol, 1)))

    print("Starting Sentence to Vector conversion (sentences2vec)...")
    # Convert sentences to DfVec objects
    vec_objects = sentences2vec(df_processed['sentence'], w2vmodel, unseen='UNK')
    df_processed['mol2vec_obj'] = [DfVec(x) for x in vec_objects]

    # Extract the vector array from the DfVec object
    print("Extracting vector arrays...")
    df_processed['mol2vec'] = [x.vec for x in df_processed['mol2vec_obj']]

    # Clean up temporary columns
    df_processed = df_processed.drop(columns=['mol2vec_obj'])

    print("Conversion complete.")
    return df_processed
