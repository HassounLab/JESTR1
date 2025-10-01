import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(v1, v2, tau=1.0) -> torch.Tensor:
        v1_norm = torch.norm(v1, dim=1, keepdim=True)
        v2_norm = torch.norm(v2, dim=1, keepdim=True)
        
        v2T = torch.transpose(v2, 0, 1)
        
        inner_prod = torch.matmul(v1, v2T)
        
        v2_normT = torch.transpose(v2_norm, 0, 1)
        
        norm_mat = torch.matmul(v1_norm, v2_normT)
        
        loss_mat = torch.div(inner_prod, norm_mat)
        
        loss_mat = loss_mat * (1/tau)
        
        loss_mat = torch.exp(loss_mat)
        
        numerator = torch.diagonal(loss_mat)
        numerator = torch.unsqueeze(numerator, 0)
        
        Lv1_v2_denom = torch.sum(loss_mat, dim=1, keepdim=True)
        Lv1_v2_denom = torch.transpose(Lv1_v2_denom, 0, 1)
        #Lv1_v2_denom = Lv1_v2_denom - numerator
        
        Lv2_v1_denom = torch.sum(loss_mat, dim=0, keepdim=True)
        #Lv2_v1_denom = Lv2_v1_denom - numerator
        
        Lv1_v2 = torch.div(numerator, Lv1_v2_denom)
        
        Lv1_v2 = -1 * torch.log(Lv1_v2)
        Lv1_v2 = torch.mean(Lv1_v2)
        
        Lv2_v1 = torch.div(numerator, Lv2_v1_denom)
        
        Lv2_v1 = -1 * torch.log(Lv2_v1)
        Lv2_v1 = torch.mean(Lv2_v1)
        
        return Lv1_v2 + Lv2_v1 , torch.mean(numerator), torch.mean(Lv1_v2_denom+Lv2_v1_denom)

def cand_spec_sim_loss(spec_enc, cand_enc):
        cand_enc = torch.transpose(cand_enc, 0, 1) # C x B x d
        spec_enc = spec_enc.unsqueeze(0) # 1 x B x d

        sim = nn.functional.cosine_similarity(spec_enc, cand_enc, dim=2)
        loss = torch.mean(sim)

        return loss

class cons_spec_loss:
        def __init__(self, loss_type) -> None:
                self.loss_compute = {'cosine': self.cos_loss,
                                     'l2':torch.nn.MSELoss()}[loss_type]
        def __call__(self,cons_spec, ind_spec):
                return self.loss_compute(cons_spec, ind_spec)
        
        def cos_loss(self, cons_spec, ind_spec):
                sim = nn.functional.cosine_similarity(cons_spec, ind_spec)
                loss = 1-torch.mean(sim) 
                return loss

class fp_loss:
        def __init__(self, loss_type) -> None:
                self.loss_compute = {'cosine': self.fp_loss_cos,
                                        'bce': nn.BCELoss()}[loss_type]
        
        def __call__(self, predicted_fp, target_fp):
                return self.loss_compute(predicted_fp, target_fp)
        
        def fp_loss_cos(self, predicted_fp, target_fp):
                sim = nn.functional.cosine_similarity(predicted_fp, target_fp)
                return 1 - torch.mean(sim)


