import torch
from torch import nn
from preprocess import TestPreprocessor


class Model(nn.Module):
    r"""
    This is a dummy model just for illustrtation. Your own model should have an 'inference' function as defined below. 
    The 'inference' function should do all necessary data pre-processing and the prediction process of your NN model. 
    When grading, we will call the 'inference' function of your own model.
    You do not need a GPU to train your model. When grading, however, we might use a GPU to make a faster work.
    """

    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.device = device
        # TODO: define your modules
        self.test_preprocessor = TestPreprocessor(device)

    def forward(self, x):
        x = x.to(self.device)
        # TODO: Implement your own forward function
        pass

    def inference(self, PID, centroid, LIDs, ligands):
        r"""
        Your own model should have this 'inference' function, which does all necessary data pre-processing and the prediction process of your NN model. 
        We will call this function to run your model and grade its performance. Please note that the input to this funciton is strictly defined as follows.
        Args:
            PID: str, one single protein ID, e.g., '112M'.
            centroid: float tuple, the x-y-z binding location of protein PID, e.g., (34.8922, 7.174, 12.4984).
            LIDs: str list, a list of ligand IDs, e.g., ['3', '3421']. You can regard len(LIDs) as the batch size during inference.
            ligands: str list, a list of SIMLEs formulas of the ligands in LIDs, e.g., ['NCCCCCCNCCCCCCN', 'C1CC1(N)P(=O)(O)O']
        Return:
            A Torch Tensor in the shape of (len(LIDs), 1), representing the predicted binding score (or likelihood) for the protein PID and each ligand in LIDs.

        About GPU:
            Again, you do not need a GPU to train your model. However, We might use GPU to accelerate out grading work. 
            So please send all your processed inputs to self.device.
            If you define any object that is not a torch.nn module, you should also explicitly send this object to self.device.
        """
        # TODO: Implement the inference function
        data = self.test_preprocessor.preprocess(PID, centroid, LIDs, ligands)
        return torch.rand(len(LIDs), 1)
