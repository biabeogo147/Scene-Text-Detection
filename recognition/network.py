import torch
import torch.nn as nn
import timm

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3):
        super(CRNN, self).__init__()
        # Load pretrained ResNet-34
        backbone = timm.create_model('resnet34', in_chans=1, pretrained=True)
        # remove last 2 layers and replace with adaptive avg pooling layer
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        # Unfreeze the last few layers
        for param in self.backbone[-unfreeze_layers:].parameters():
            param.requires_grad = True

        self.mapSeq = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    @torch.autocast(device_type='cuda')
    def forward(self, x):
        # Input shape: (b, c, h, w)
        x = self.backbone(x) # (b, 2048, 1, w)
        x = x.permute(0, 3, 2, 1) # (b, w, 1, 2048)
        x = x.view(x.size(0), x.size(1), -1) # (b, w, 2048)
        x = self.mapSeq(x) # (b, w, 512)
        x, _ = self.gru(x) # (b, w, hidden * 2)
        x = self.layer_norm(x)
        x = self.out(x) # (b, w, vocab)
        x = x.permute(1, 0, 2) # (w, b, vocab) for CTC loss
        return x