##### mymodel.py

import torch
import torch.nn as nn

class Word2Vec_NN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(300, 2048),
            nn.LeakyReLU(0.1),
            torch.nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.1),
            torch.nn.BatchNorm1d(1024),
            nn.Dropout(0.2),  # First Dropout layer
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            torch.nn.BatchNorm1d(1024),
            nn.Dropout(0.2),  # Second Dropout layer
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.1),
            torch.nn.BatchNorm1d(256),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.stack(x)


def count_parameters(model):
    """
    Calculates the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calculate and print the total parameters
if __name__ == '__main__':
    my_deep_model = Word2Vec_NN_model()
    total_params = count_parameters(my_deep_model)
    print(f"Total trainable parameters: {total_params:,}")