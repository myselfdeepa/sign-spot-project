import torch
import torch.nn as nn
import torch.nn.functional as F
from CFE import run_cfe  # Import the CFE runner function

class Mlp(torch.nn.Module):
    def __init__(
        self,
        input_dim=1024,
        adaptation_dims=[512, 256],
        with_norm=True,
        num_classes=1064,
        with_classification=False,
        dropout_keep_prob=0.0,
    ):
        super(Mlp, self).__init__()
        self.input_dim = input_dim
        self.with_norm = with_norm
        self.with_classification = with_classification
        self.dropout = nn.Dropout(dropout_keep_prob)

        # 1. Layers learning a 1024d residual
        self.res = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.LeakyReLU(0.2, inplace=True)
        )
        # 2. Layers learning the embeddings from 1024d -> 256d
        layers = []
        layers.append(nn.Linear(input_dim, adaptation_dims[0]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(1, len(adaptation_dims)):
            layers.append(nn.Linear(adaptation_dims[i - 1], adaptation_dims[i]))
            # layers.append(nn.BatchNorm1d(adaptation_dims[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.adaptor = nn.Sequential(*layers)
        output_dim = adaptation_dims[-1]
        # 3. Classifier
        self.logits = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        x = x + self.res(x)
        x = self.adaptor(x)
        # L2 normalize each feature vector
        if self.with_norm:
            x = F.normalize(x, p=2, dim=1)
        out = {"embds": x}
        if self.with_classification:
            logits = self.logits(self.dropout(x))
            out["logits"] = logits
        return out

def run_mlp_on_features(features, mlp_model_path):
    # Load MLP model
    mlp_model = Mlp(input_dim=1024, adaptation_dims=[512, 256], num_classes=1064, with_classification=True)
    mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=torch.device('cpu')))
    mlp_model.eval()

    # Run MLP on the features
    with torch.no_grad():
        mlp_output = mlp_model(features)['logits']

    return mlp_output

def main():
    # Extract features using CFE
    cfe_output = run_cfe()  # Ensure this returns a tensor of shape [batch_size, seq_len, feature_dim]
    
    # Path to your MLP model weights
    mlp_model_path = '/home/uday/Desktop/SignSpotting/bsldict/models/mlp.pth.tar'

    # Run MLP on the extracted features
    mlp_output = run_mlp_on_features(cfe_output, mlp_model_path)

    # Process mlp_output as needed
    print("MLP Output:", mlp_output)

if __name__ == "__main__":
    main()
