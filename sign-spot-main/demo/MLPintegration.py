import torch
from mlp import Mlp
from CFE import run_cfe  # Import the run_cfe function from CFE.py
from ISFE import run_isfe
class MLPIntegration:
    def __init__(self, checkpoint_path, input_dim=1024, adaptation_dims=[512, 256], num_classes=1064):
        self.model = Mlp(input_dim=input_dim, adaptation_dims=adaptation_dims, num_classes=num_classes, with_classification=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def process_and_identify(self, features):
        features = features.to(self.device)
        
        with torch.no_grad():
            output = self.model(features)['logits']
        
        probabilities = torch.softmax(output, dim=1)
        highest_prob_class_indices = torch.argmax(probabilities, dim=1)
        return highest_prob_class_indices, probabilities

if __name__ == "__main__":
    checkpoint_path = '/home/uday/Desktop/SignSpotting/bsldict/models/mlp.pth.tar'
    mlp_integration = MLPIntegration(checkpoint_path)

    # Get features from CFE
   # cfe_output = run_cfe()
    isfe_output = run_isfe()
    # Process features through MLP
    highest_prob, probabilities = mlp_integration.process_and_identify(isfe_output)
    print("Highest probability class index:", highest_prob)
    # Optional: print probabilities if needed
    # print("Probabilities:", probabilities)





"""
Transformer MLP
import torch
from mlp import Mlp
from SignSpotter import run_sign_spotter  # Import the function from SignSpotter.py
from CFE import 

class MLPIntegration:
    def __init__(self, checkpoint_path, input_dim=1024, adaptation_dims=[512, 256], num_classes=1064):
        self.model = Mlp(input_dim=input_dim, adaptation_dims=adaptation_dims, num_classes=num_classes,with_classification=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    
    def process_and_identify(self, transformer_output):
        # Assuming transformer_output is of shape [159, 1, 1024]
        if transformer_output.dim() == 3:
            transformer_output = transformer_output.squeeze(1)

        transformer_output = transformer_output.to(self.device)
        
        with torch.no_grad():
            output = self.model(transformer_output)['logits']
        
        probabilities = torch.softmax(output, dim=1)  # Apply softmax to get probabilities
        highest_prob_class_indices = torch.argmax(probabilities, dim=1)  # Find the indices of the highest probability class for each segment
        return highest_prob_class_indices, probabilities

if __name__ == "__main__":
    checkpoint_path = '/home/uday/Desktop/SignSpotting/bsldict/models/mlp.pth.tar'
    mlp_integration = MLPIntegration(checkpoint_path)

    # Example transformer output (replace with actual output from SignSpotter.py)
    #transformer_output = run_sign_spotter()
    #query_class_index = 123  # Replace with the actual class index of the query sign
    transformer_output = run_cfe()

    highest_prob,probabilities = mlp_integration.process_and_identify(transformer_output)
    print("highest probability class index",highest_prob)
    #print("Probabilities:",probabilities)
"""
