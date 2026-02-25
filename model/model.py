import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from clip import clip
from utils.model_utils import is_main_process
from .base import BaseModel

class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.layer_norm(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)

        out = out + identity
        return out

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: [Batch, Class_Num, Input_Dim]
        # adj: [Class_Num, Class_Num]

        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        
        return x

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x: [B, N, D_in]
        # adj: [N, N]
        # support: A * X -> [B, N, D_in]
        support = torch.matmul(adj, x)
        # output: A * X * W -> [B, N, D_out]
        output = torch.matmul(support, self.weight)
        return output

class SFRNet(BaseModel):
    def __init__(
            self,
            cfg,
            clip_model,
            classnames_seen, 
            classnames_unseen
    ):
        super().__init__()
        self.cfg = cfg
        self.classnames_seen = classnames_seen
        self.classnames_unseen = classnames_unseen
        self.classnames = self.classnames_seen+self.classnames_unseen
        self.num_classes_seen = len(classnames_seen)
        self.num_classes_unseen = len(classnames_unseen)
        self.device = dist.get_rank() if cfg.MODEL.DIST_TRAIN else 'cuda'

        # add clip model and text tokens
        self.clip_model = clip_model
        self.text_tokens = self.get_text_templates()

        # MEF Module
        self.mlp = nn.Sequential(
            ResidualMLPBlock(input_dim=512, hidden_dim=1024), 
            ResidualMLPBlock(input_dim=512, hidden_dim=1024), 
            nn.Linear(512, 1)
        )   
        # GR Module
        self.gcn_corrector = GCN(input_dim=1, hidden_dim=64, output_dim=1) 

        # freeze the model
        self.freeze(cfg.MODEL.TRANSFER_TYPE)       

    def get_text_templates(self):
        file_name = 'sewerml_de.txt'
        if is_main_process():
            print("Loading text prompts from:", file_name)
        with open(f'datasets/text/{file_name}', 'r') as f:
            expert_prompts = f.readlines()
        expert_prompts = [prompt.strip() for prompt in expert_prompts]
        
        text_tokens = clip.tokenize(expert_prompts)
        return text_tokens.cuda()
    
    def forward(self, img, class_matrix, zsl=False, gzsl=False):

        # text slice
        seen = True if not zsl and not gzsl else False
        if seen:
            text_tokens = self.text_tokens[:self.num_classes_seen].clone()
            class_matrix = class_matrix[:self.num_classes_seen, :self.num_classes_seen]
        elif zsl:
            text_tokens = self.text_tokens[self.num_classes_seen:self.num_classes_seen+self.num_classes_unseen].clone()
            class_matrix = class_matrix[self.num_classes_seen:, self.num_classes_seen:]
        else:
            text_tokens = self.text_tokens[:self.num_classes_seen+self.num_classes_unseen].clone()
            class_matrix = class_matrix[:self.num_classes_seen+self.num_classes_unseen, :self.num_classes_seen+self.num_classes_unseen]

        # encode text 
        text_features = self.clip_model.encode_text(text_tokens)
        # text_features:[num_classes_seen, 512] -> [num_classes_seen, 1, 512] -> [1, num_classes_seen, 512]
        text_features = text_features.unsqueeze(1).permute(1, 0, 2) 
        
        # encode image
        image_fea = self.clip_model.encode_image(img)
        img_cls = image_fea[:, :1] # [bs, 1, 512]
        img_loc = image_fea[:, 1:] # [bs, num_patches(196), 512]

        # normalize
        img_cls = F.normalize(img_cls, dim=-1)
        img_loc = F.normalize(img_loc, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # global match score
        logits_glb = img_cls @ text_features.transpose(1, 2)
        logits_glb = logits_glb.squeeze(1)

        # local score = sum of patch scores weighted by affinity
        patch_match_score = img_loc @ text_features.transpose(1, 2)
        affinity_matrix = patch_match_score.softmax(dim=1)
        logits_loc = self.mlp(affinity_matrix.transpose(1, 2) @ img_loc) 
        logits_loc = logits_loc.squeeze(-1)

        logits = logits_glb + logits_loc

        # GR correction
        gcn_correction = self.gcn_corrector(logits.unsqueeze(-1), class_matrix)
        gcn_correction = gcn_correction.squeeze(-1)

        logits = logits + gcn_correction

        return logits   
