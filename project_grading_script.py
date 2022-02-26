import torch
from torch import nn
import pandas as pd
#from student_submission_folder.model import Model


class Model(nn.Module):
    r"""
    This is a dummy model just for illustrtation. Your own model should have an 'inference' function as defined below. 
    The 'inference' function should do all necessary data pre-processing and the forward computation of your NN model. 
    When grading, we will call this 'inference' function of your own model.
    You do not need a GPU to train your model. When grading, however, we might use a GPU to make a faster work.
    """
    def __init__(self, device='cpu'):
        super(Model, self).__init__()
        self.device = device
        # TODO: define your modules
        pass

    def forward(self, x):   
        x = x.to(self.device)
        # TODO: Implement your own forward function
        pass

    def inference(self, PID, PDBS_DIR, centroid, ligands):
        r"""
        Your own model should have this 'inference' function, which does all necessary data pre-processing and the forward computation of your NN model. 
        We will call this function to run your model and grade its performance. Please note that the input to this funciton is strictly defined as follows:
        Args:
            PID: str, one single protein ID, e.g., '112M'. You should use this ID to query the corresponding PDB file and process it.
            PDBS_DIR: str, the PDF directory from which to read PDB files.
            centroid: float tuple, the x-y-z binding location of protein PID, e.g., (34.8922, 7.174, 12.4984).
            ligands: str list, a list of SIMLEs formulas of ligands, e.g., ['NCCCCCCNCCCCCCN', 'C1CC1(N)P(=O)(O)O']
        Return:
            A Torch Tensor in the shape of (len(ligands), 1), representing the predicted binding score (or likelihood) for protein PID and each ligand.

        About GPU:
            Again, you do not need a GPU to train your model. However, We might use GPU to accelerate our grading work. 
            So please send all your pre-processed inputs to self.device.
            If you define any object that is not a torch.nn module, you should also explicitly send this object to self.device.
        """
        # TODO: Implement the inference function
        return torch.rand(len(ligands), 1)


if __name__ == '__main__':
    CENTROIDS_DIR = './project_test_data/centroids.csv'
    PDBS_DIR = './project_test_data/pdbs'
    LIGAND_DIR = './project_test_data/ligand.csv'
    GT_PAIR_DIR = './project_test_data/pair.csv'
    MODEL_PATH = './student_submission_folder/parameters.pt'    #pre-trained parameters of your model

    #read centroids.csv
    centroids = {}
    df = pd.read_csv(CENTROIDS_DIR)
    for i in range(len(df)):
        centroids[str(df.PID[i])] = (float(df.x[i]), float(df.y[i]), float(df.z[i]))
    #centroids dict format: {'112M': (34.8922, 7.174, 12.4984), ...}


    #read ligand.csv
    ligands = {}    
    df = pd.read_csv(LIGAND_DIR)
    for i in range(len(df)):
        ligands[str(df.LID[i])] = (str(df.Smiles[i]))
    #ligands dict format: {'3':'NCCCCCCNCCCCCCN', '3421':'C1CC1(N)P(=O)(O)O', ...}
    LIDs =[LID for LID in ligands]
    #LIDs format: ['3', '3421', ...], i.e., the key list of ligands dict.


    #read groundtruth pair.csv for grading
    gt_pairs = {}
    df = pd.read_csv(GT_PAIR_DIR)
    for i in range(len(df)):
        gt_pairs[str(df.PID[i])] = (str(df.LID[i]))
    #gt_pairs dict format: {'112M': '3421', ...}

    
    BS = 100                        #Batch size for inference
    TOPK = 10                       #Set top-10 accuracy
    DEVICE = 'cpu'                  #You do not necessarily need a GPU. Of course, you are free to use a GPU if you have one.

    model = Model(device=DEVICE)    #When grading, we will import and call your own model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.to(DEVICE)
    model.eval()


    #inference
    prediction_correctness = []
    #traverse through all test proteins
    for PID in centroids:
        binding_scores = torch.empty(0, 1)
        #traverse through all test ligands, covering BS ligands each time:
        for i in range(0, len(LIDs)-BS+1, BS):
            batch_pred = model.inference(PID, PDBS_DIR, centroids[PID], [ligands[LID] for LID in LIDs[i: i+BS]])
            binding_scores = torch.cat([binding_scores, batch_pred], dim=0)
        if i < len(LIDs)-BS:
            batch_pred = model.inference(PID, PDBS_DIR, centroids[PID], [ligands[LID] for LID in LIDs[i+BS: ]])
            binding_scores = torch.cat([binding_scores, batch_pred], dim=0)

        #transform torch.tensor to list
        binding_scores = binding_scores.squeeze(-1).cpu().detach().numpy().tolist()

        #get top-k scores and corresponding LIDs
        topk_pred = sorted(zip(binding_scores, LIDs), reverse=True)[: TOPK]
        topk_scores, topk_LIDs = zip(*topk_pred)
        #print(topk_LIDs)
        
        #compare with groundtruth
        if str(gt_pairs[PID]) in topk_LIDs:
            prediction_correctness.append(1)
        else:
            prediction_correctness.append(0)

    accuracy = sum(prediction_correctness) / len(prediction_correctness)

    print(f"Inference Prediction Score: {'{:.5f}'.format(accuracy)}.")