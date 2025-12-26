import torch
import torch.nn.functional as F

class SemanticSimilarityNormalizer:
    def __init__(self, alpha=0.05, eps=1e-6):

        self.alpha = alpha
        self.eps = eps
        self.o_star = None

    def compute_batch_o_star(self, report_feats):


        r = F.normalize(report_feats, dim=-1)  # (B, D)
        p_b = r.mean(dim=0, keepdim=True)      # (1, D) batch prototype
        sim = F.cosine_similarity(r, p_b, dim=-1)  # (B,)
        o_star_batch = sim.mean()  # scalar
        return o_star_batch

    def update_ema(self, o_star_batch):

        if self.o_star is None:
            self.o_star = o_star_batch.detach()
        else:
            self.o_star = self.alpha * o_star_batch.detach() + (1 - self.alpha) * self.o_star

    def normalize_similarity(self, report_feats):

        r = F.normalize(report_feats,p=2, dim=-1)  # (B, D)
        S = torch.matmul(r, r.T)               # (B, B)

        o_star_batch = self.compute_batch_o_star(report_feats)
        self.update_ema(o_star_batch)

        S_normalized = (S - self.o_star) / (1 - self.o_star + self.eps)  # 避免除零
        return S_normalized.detach()