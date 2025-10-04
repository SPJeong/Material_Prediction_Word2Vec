##### main_training.py (Word2Vec)

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gensim  # !conda install gensim
from gensim.models import word2vec
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # for AMP, float 16 or 32 calculation
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# from features import mol2alt_sentence, MolSentence, DfVec, featurize # !pip install git+https://github.com/samoturk/mol2vec
# from helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg # !pip install git+https://github.com/samoturk/mol2vec

import CONFIG  # custom.py
import chemical_feature_extraction  # custom.py
import data_extraction  # custom.py
import mymodel  # custom.py
import my_utils  # custom.py
from model_trainer import train, validate, test  # custom.py

# parameter setting
filtered_num = CONFIG.filtered_num
random_pick_num = CONFIG.random_pick_num
data_extraction_folder = CONFIG.data_extraction_folder
chemical_feature_extraction_folder = CONFIG.chemical_feature_extraction_folder
plot_save_folder = CONFIG.plot_save_folder
model_save_folder = CONFIG.model_save_folder
os.makedirs(chemical_feature_extraction_folder, exist_ok=True)
os.makedirs(plot_save_folder, exist_ok=True)
os.makedirs(model_save_folder, exist_ok=True)
model_name = 'Word2Vec'
batch_size = CONFIG.batch_size
learning_rate = CONFIG.learning_rate
epochs = CONFIG.epochs
device = CONFIG.device
ROnPlateauLR_factor = CONFIG.ROnPlateauLR_factor
ROnPlateauLR_patience = CONFIG.ROnPlateauLR_patience

Y_total_list = ['Cp', 'Tg', 'Tm', 'Td', 'LOI',
                'YM', 'TSy', 'TSb', 'epsb', 'CED',
                'Egc', 'Egb', 'Eib', 'Ei', 'Eea', 'nc', 'ne',
                'permH2', 'permHe', 'permCH4', 'permCO2', 'permN2', 'permO2',
                'Eat', 'rho', 'Xc', 'Xe']

# load file
file_folder = chemical_feature_extraction_folder
file_name = f'chemical_feature_extraction_len_{filtered_num}_num_{random_pick_num}_scaled_False_ECFP_False_desc_False.csv'
file_raw_path = os.path.join(file_folder, file_name)

if os.path.exists(file_raw_path):
    print(f"Loading existing file from: {file_raw_path}")
    file_raw = pd.read_csv(file_raw_path)

else:
    print(f"File not found. Generating data and saving to: {file_raw_path}")
    file_raw = chemical_feature_extraction.run_feature_extraction(filtered_num=filtered_num,
                                                                  random_pick_num=random_pick_num,
                                                                  data_extraction_folder=data_extraction_folder,
                                                                  ecfp=False,
                                                                  descriptors=False,
                                                                  scale_descriptors=False,
                                                                  ecfp_radius=None,
                                                                  ecfp_nbits=None,
                                                                  chemical_feature_extraction_folder=chemical_feature_extraction_folder,
                                                                  inference_mode=False,
                                                                  new_smiles_list=None)

# pre-trained word2vec model load
w2vmodel = word2vec.Word2Vec.load('./model_300dim.pkl')
# print(len(w2vmodel.wv.index_to_key) # 21003


X_file_processed = my_utils.convert_smiles_to_mol2vec(df=file_raw,
                                                      smiles_col='smiles',
                                                      w2vmodel=w2vmodel)  # Pass your loaded Word2Vec model here

'''
# sanity check 
print(X_file_processed.head())
print(f"Processed shape: {X_file_processed.shape}")
# e.g mol -> sentence
sentence=mol2alt_sentence(mol= X_file_processed['mol'][0], radius= 1) # mol -> sentence conversion 
print(len(sentence), sentence) 
# e.g. identifier image for each component of the sentence
it = IdentifierTable(sentence, [X_file_processed['mol'][0]]*len(sentence), [sentence]*len(sentence), 5, 1) 
print(it)
'''

# Convert the list of lists into a proper numpy array
X_file = np.array(X_file_processed['mol2vec'].tolist(), dtype='float32').copy()

# total targets for Y
start_column_index = X_file_processed.columns.get_loc('Egc')
end_column_index = X_file_processed.columns.get_loc('Tm')
Y_total_file = X_file_processed.iloc[:, start_column_index:end_column_index + 1]

for i, target_name in tqdm(enumerate(Y_total_list), total=len(Y_total_list)):
    y_data = Y_total_file[[str(target_name)]].to_numpy().ravel().astype('float32')  # float16 or 32 conversion

    # split
    X_train, X_temp, y_train, y_temp = train_test_split(X_file, y_data, test_size=0.2, random_state=777)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=777)

    # Print shapes to verify splits
    print(
        f"target: {target_name} | X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")

    # dataloader
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model, loss fn and optimizer define
    my_model = mymodel.Word2Vec_NN_model()
    my_model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=my_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = GradScaler()  # GradScaler for float16 calculation

    # ROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=ROnPlateauLR_factor,
                                                           patience=ROnPlateauLR_patience)

    # Main training and validation loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # initiate val loss as infinite
    best_model_state = None  # for best model save during training

    for epoch in range(0, epochs):
        # load train, val function
        train_loss = train(my_model, optimizer, loss_fn, train_loader, device, scaler)
        val_loss = validate(my_model, loss_fn, val_loader, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check if current validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = my_model.state_dict()
            print(f"New best model found at epoch {epoch} with Val Loss: {best_val_loss:.4f}. Model state saved.")

        if epoch % 1 == 0:
            print(
                f"Target: {target_name} | Epoch {epoch} | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")

    # Final evaluation on the test set after training is complete
    final_metrics = test(my_model, test_loader, device, plot_save_folder, model_name, target_name)
    print(f'\nFinal Metrics on Test Set:')
    for metric_name, value in final_metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")

    # Dictionary to save all parameters and metrics
    results = {'model_type': model_name,
               'target_variable': target_name,
               'model_state_dict': best_model_state,
               'optimizer_state_dict': optimizer.state_dict(),
               'train_losses': train_losses,
               'val_losses': val_losses,
               'final_test_metrics': final_metrics,
               'epochs': epoch, }

    # Save the entire package (best model + metadata)
    model_file_name = f'{model_name}_model_len_{filtered_num}_num_{random_pick_num}_{target_name}.pth'
    model_save_file_name = os.path.join(model_save_folder, model_file_name)
    torch.save(results, model_save_file_name)
    print(f'Best model and training results saved to {model_save_file_name}')