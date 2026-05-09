import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
#                Prototype Head
# =====================================================

class PrototypeHead3D(nn.Module):

    def __init__(self, in_channels, num_classes=3, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.proj = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, feat, ptv_s, oar_s, bg_s):

        B, C, D, H, W = feat.shape

        feat = self.proj(feat)
        feat = F.normalize(feat, dim=1)

        masks = [
            (ptv_s > 0),
            (oar_s > 0),
            (bg_s > 0)
        ]

        prototypes = []

        for mask in masks:
            if mask.sum() > 0:
                proto = feat.permute(0,2,3,4,1)[mask].mean(dim=0)
            else:
                proto = torch.zeros(C, device=feat.device)

            proto = F.normalize(proto, dim=0)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes, dim=0)  # [3,C]

        feat_flat = feat.view(B, C, -1)
        logits = torch.einsum("kc,bcn->bkn", prototypes, feat_flat)
        logits = logits.view(B, self.num_classes, D, H, W)

        return logits / self.temperature, prototypes


# =====================================================
#                Light Decoder
# =====================================================

class LightDecoder3D(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv3d(num_classes, num_classes, 3, padding=1),
            nn.BatchNorm3d(num_classes),
            nn.ReLU(inplace=False),
            nn.Conv3d(num_classes, num_classes, 3, padding=1)
        )

    def forward(self, logits):
        return self.refine(logits)


# =====================================================
#                Full Network
# =====================================================

class PrototypeLightDecoderNet3D(nn.Module):

    def __init__(self, in_channels=1, feat_dim=64, num_classes=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=False),
            nn.Conv3d(32, feat_dim, 3, padding=1),
            nn.BatchNorm3d(feat_dim),
            nn.ReLU(inplace=False),
        )

        self.prototype_head = PrototypeHead3D(
            in_channels=feat_dim,
            num_classes=num_classes
        )

        self.decoder = LightDecoder3D(num_classes=num_classes)

    def forward(self, x, ptv_s, oar_s, bg_s):

        feat = self.encoder(x)

        coarse_logits, prototypes = self.prototype_head(
            feat, ptv_s, oar_s, bg_s
        )

        refined_logits = self.decoder(coarse_logits)

        compact_loss = prototype_compactness_loss(
            feat, ptv_s, oar_s, bg_s, prototypes
        )

        sep_loss = prototype_separation_loss(prototypes)

        return refined_logits, compact_loss, sep_loss


# =====================================================
#          Prototype Compactness Loss
# =====================================================

def prototype_compactness_loss(feat, ptv_s, oar_s, bg_s, prototypes):

    B, C, D, H, W = feat.shape
    feat_flat = feat.permute(0,2,3,4,1)

    masks = [
        (ptv_s > 0),
        (oar_s > 0),
        (bg_s > 0)
    ]

    loss = 0.0
    cnt = 0

    for cls, mask in enumerate(masks):
        if mask.sum() > 0:
            f = feat_flat[mask]
            p = prototypes[cls]
            loss += ((f - p)**2).mean()
            cnt += 1

    return loss / max(cnt, 1)


# =====================================================
#          Prototype Separation Loss
# =====================================================

def prototype_separation_loss(prototypes):

    loss = 0.0
    K = prototypes.shape[0]

    for i in range(K):
        for j in range(i+1, K):
            loss += torch.exp(-((prototypes[i] - prototypes[j])**2).mean())

    return loss


# =====================================================
#                Scribble Loss
# =====================================================

def scribble_loss_3class(logits, ptv_s, oar_s, bg_s):

    logits_perm = logits.permute(0,2,3,4,1)

    loss = 0.0
    cnt = 0

    masks = [
        (ptv_s > 0),
        (oar_s > 0),
        (bg_s > 0)
    ]

    for cls, mask in enumerate(masks):
        if mask.sum() > 0:
            l = logits_perm[mask]
            target = torch.full(
                (l.shape[0],),
                cls,
                dtype=torch.long,
                device=logits.device
            )
            loss += F.cross_entropy(l, target)
            cnt += 1

    return loss / max(cnt, 1)


# =====================================================
#                MAIN TEST
# =====================================================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 2
    C_in = 3
    D, H, W = 16, 64, 64

    x = torch.randn(B, C_in, D, H, W).to(device)

    ptv_s = torch.zeros(B, D, H, W, device=device)
    oar_s = torch.zeros(B, D, H, W, device=device)
    bg_s  = torch.zeros(B, D, H, W, device=device)

    ptv_s[:, 4:6, 20:30, 20:30] = 1
    oar_s[:, 8:10, 35:45, 35:45] = 1
    bg_s[:, 12:14, 5:15, 5:15] = 1

    model = PrototypeLightDecoderNet3D(
        in_channels=C_in,
        feat_dim=64,
        num_classes=3
    ).to(device)

    model.eval()

    with torch.no_grad():
        logits, compact_loss, sep_loss = model(x, ptv_s, oar_s, bg_s)

    print("Logits shape:", logits.shape)
    print("Compactness loss:", compact_loss.item())
    print("Separation loss:", sep_loss.item())

    ce_loss = scribble_loss_3class(
        logits, ptv_s, oar_s, bg_s
    )

    print("CE loss:", ce_loss.item())

    total_loss = ce_loss + 0.1 * compact_loss + 0.05 * sep_loss
    print("Total loss:", total_loss.item())

    print("Corrected Prototype + Light Decoder version works.")
