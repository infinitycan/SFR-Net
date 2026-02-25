import torch
import torch.nn.functional as F
from clip import clip

from utils.model_utils import is_main_process

def build_adjacency_matrix(class_exp, clip_model, threshold=0.85):
    """
    Constructs an adjacency matrix based on the semantic similarity of defect class descriptions.
    """
    with torch.no_grad():
        # 1. Tokenize the textual descriptions of classes
        tokens = clip.tokenize(class_exp)
        device = next(clip_model.parameters()).device 
        tokens = tokens.to(device)
        # 2. Encode text into embeddings using CLIP
        class_embeddings = clip_model.encode_text(tokens)
        
    # 3. Compute the cosine similarity matrix
    class_embeddings_norm = F.normalize(class_embeddings, p=2, dim=1)
    # Matrix multiplication: [Nc, D] @ [D, Nc] -> [Nc, Nc]
    cosine_sim_matrix = class_embeddings_norm @ class_embeddings_norm.t()
    
    # 4. Generate adjacency matrix A based on the similarity threshold
    adj = (cosine_sim_matrix > threshold).float()
    
    # 5. Subtract the identity matrix to remove self-loops 
    # (These will be re-added uniformly in the next step)
    adj = adj - torch.eye(len(class_exp), device=adj.device)
    adj[adj < 0] = 0  # Ensure no negative values
    
    # 6. Standard GCN operation: Add self-loops (A' = A + I)
    adj = adj + torch.eye(len(class_exp), device=adj.device)
    
    # 7. Normalize the adjacency matrix (D^-0.5 * A' * D^-0.5) 
    # This follows the standard GCN symmetric normalization.
    row_sum = adj.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    if is_main_process():
        print("Adjacency matrix (normalized) for GR:")
        print(norm_adj.cpu().numpy())

    return norm_adj
