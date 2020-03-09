import torch
import torch.nn as nn

from ..registry import registry
from .modeling_utils import ProteinConfig
from .modeling_utils import ProteinModel

URL_PREFIX = "https://s3.amazonaws.com/proteindata/pytorch-models/"
TRROSETTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xaa': URL_PREFIX + "trRosetta-xaa-pytorch_model.bin",
    'xab': URL_PREFIX + "trRosetta-xab-pytorch_model.bin",
    'xac': URL_PREFIX + "trRosetta-xac-pytorch_model.bin",
    'xad': URL_PREFIX + "trRosetta-xad-pytorch_model.bin",
    'xae': URL_PREFIX + "trRosetta-xae-pytorch_model.bin",
}
TRROSETTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xaa': URL_PREFIX + "trRosetta-xaa-config.json",
    'xab': URL_PREFIX + "trRosetta-xab-config.json",
    'xac': URL_PREFIX + "trRosetta-xac-config.json",
    'xad': URL_PREFIX + "trRosetta-xad-config.json",
    'xae': URL_PREFIX + "trRosetta-xae-config.json",
}


class TRRosettaConfig(ProteinConfig):

    pretrained_config_archive_map = TRROSETTA_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 num_features: int = 64,
                 kernel_size: int = 3,
                 num_layers: int = 61,
                 dropout: float = 0.15,
                 msa_cutoff: float = 0.8,
                 penalty_coeff: float = 4.5,
                 initializer_range: float = 0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.msa_cutoff = msa_cutoff
        self.penalty_coeff = penalty_coeff
        self.initializer_range = initializer_range


class MSAFeatureExtractor(nn.Module):

    def __init__(self, config: TRRosettaConfig):
        super().__init__()
        self.msa_cutoff = config.msa_cutoff
        self.penalty_coeff = config.penalty_coeff

    def forward(self, msa1hot):
        # Convert to float, then potentially back to half
        # These transforms aren't well suited to half-precision
        initial_type = msa1hot.dtype

        msa1hot = msa1hot.float()
        seqlen = msa1hot.size(2)

        weights = self.reweight(msa1hot)
        features_1d = self.extract_features_1d(msa1hot, weights)
        features_2d = self.extract_features_2d(msa1hot, weights)

        left = features_1d.unsqueeze(2).repeat(1, 1, seqlen, 1)
        right = features_1d.unsqueeze(1).repeat(1, seqlen, 1, 1)
        features = torch.cat((left, right, features_2d), -1)
        features = features.type(initial_type)
        features = features.permute(0, 3, 1, 2)
        features = features.contiguous()
        return features

    def reweight(self, msa1hot, eps=1e-9):
        # Reweight
        seqlen = msa1hot.size(2)
        id_min = seqlen * self.msa_cutoff
        id_mtx = torch.stack([torch.tensordot(el, el, [[1, 2], [1, 2]]) for el in msa1hot], 0)
        id_mask = id_mtx > id_min
        weights = 1.0 / (id_mask.type_as(msa1hot).sum(-1) + eps)
        return weights

    def extract_features_1d(self, msa1hot, weights):
        # 1D Features
        f1d_seq = msa1hot[:, 0, :, :20]
        batch_size = msa1hot.size(0)
        seqlen = msa1hot.size(2)

        # msa2pssm
        beff = weights.sum()
        f_i = (weights[:, :, None, None] * msa1hot).sum(1) / beff + 1e-9
        h_i = (-f_i * f_i.log()).sum(2, keepdims=True)
        f1d_pssm = torch.cat((f_i, h_i), dim=2)
        f1d = torch.cat((f1d_seq, f1d_pssm), dim=2)
        f1d = f1d.view(batch_size, seqlen, 42)
        return f1d

    def extract_features_2d(self, msa1hot, weights):
        # 2D Features
        batch_size = msa1hot.size(0)
        num_alignments = msa1hot.size(1)
        seqlen = msa1hot.size(2)
        num_symbols = 21

        if num_alignments == 1:
            # No alignments, predict from sequence alone
            f2d_dca = torch.zeros(
                batch_size, seqlen, seqlen, 442,
                dtype=torch.float,
                device=msa1hot.device)
            return f2d_dca

        # compute fast_dca
        # covariance
        x = msa1hot.view(batch_size, num_alignments, seqlen * num_symbols)
        num_points = weights.sum(1) - weights.mean(1).sqrt()
        mean = (x * weights.unsqueeze(2)).sum(1, keepdims=True) / num_points[:, None, None]
        x = (x - mean) * weights[:, :, None].sqrt()
        cov = torch.matmul(x.transpose(-1, -2), x) / num_points[:, None, None]

        # inverse covariance
        reg = torch.eye(seqlen * num_symbols,
                        device=weights.device,
                        dtype=weights.dtype)[None]
        reg = reg * self.penalty_coeff / weights.sum(1, keepdims=True).sqrt().unsqueeze(2)
        cov_reg = cov + reg
        inv_cov = torch.stack([torch.inverse(cr) for cr in cov_reg.unbind(0)], 0)

        x1 = inv_cov.view(batch_size, seqlen, num_symbols, seqlen, num_symbols)
        x2 = x1.permute(0, 1, 3, 2, 4)
        features = x2.reshape(batch_size, seqlen, seqlen, num_symbols * num_symbols)

        x3 = (x1[:, :, :-1, :, :-1] ** 2).sum((2, 4)).sqrt() * (
            1 - torch.eye(seqlen, device=weights.device, dtype=weights.dtype)[None])
        apc = x3.sum(1, keepdims=True) * x3.sum(2, keepdims=True) / x3.sum(
            (1, 2), keepdims=True)
        contacts = (x3 - apc) * (1 - torch.eye(
            seqlen, device=x3.device, dtype=x3.dtype).unsqueeze(0))

        f2d_dca = torch.cat([features, contacts[:, :, :, None]], axis=3)
        return f2d_dca

    @property
    def feature_size(self) -> int:
        return 526


class DilatedResidualBlock(nn.Module):

    def __init__(self, num_features: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = self._get_padding(kernel_size, dilation)
        self.conv1 = nn.Conv2d(
            num_features, num_features, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.InstanceNorm2d(num_features, affine=True, eps=1e-6)
        self.actv1 = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            num_features, num_features, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.InstanceNorm2d(num_features, affine=True, eps=1e-6)
        self.actv2 = nn.ELU(inplace=True)
        self.apply(self._init_weights)
        nn.init.constant_(self.norm2.weight, 0)

    def _get_padding(self, kernel_size: int, dilation: int) -> int:
        return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()

        # elif isinstance(module, DilatedResidualBlock):
            # nn.init.constant_(module.norm2.weight, 0)

    def forward(self, features):
        shortcut = features
        features = self.conv1(features)
        features = self.norm1(features)
        features = self.actv1(features)
        features = self.dropout(features)
        features = self.conv2(features)
        features = self.norm2(features)
        features = self.actv2(features + shortcut)
        return features


class TRRosettaAbstractModel(ProteinModel):

    config_class = TRRosettaConfig
    base_model_prefix = 'trrosetta'
    pretrained_model_archive_map = TRROSETTA_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config: TRRosettaConfig):
        super().__init__(config)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DilatedResidualBlock):
            nn.init.constant_(module.norm2.weight, 0)


class TRRosettaPredictor(TRRosettaAbstractModel):

    def __init__(self, config: TRRosettaConfig):
        super().__init__(config)
        layers = [
            nn.Conv2d(526, config.num_features, 1),
            nn.InstanceNorm2d(config.num_features, affine=True, eps=1e-6),
            nn.ELU(),
            nn.Dropout(config.dropout)]

        dilation = 1
        for _ in range(config.num_layers):
            block = DilatedResidualBlock(
                config.num_features, config.kernel_size, dilation, config.dropout)
            layers.append(block)

            dilation *= 2
            if dilation > 16:
                dilation = 1

        self.resnet = nn.Sequential(*layers)
        self.predict_theta = nn.Conv2d(config.num_features, 25, 1)
        self.predict_phi = nn.Conv2d(config.num_features, 13, 1)
        self.predict_dist = nn.Conv2d(config.num_features, 37, 1)
        self.predict_bb = nn.Conv2d(config.num_features, 3, 1)
        self.predict_omega = nn.Conv2d(config.num_features, 25, 1)

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)
        nn.init.constant_(self.predict_theta.weight, 0)
        nn.init.constant_(self.predict_phi.weight, 0)
        nn.init.constant_(self.predict_dist.weight, 0)
        nn.init.constant_(self.predict_bb.weight, 0)
        nn.init.constant_(self.predict_omega.weight, 0)

    def forward(self,
                features,
                theta=None,
                phi=None,
                dist=None,
                omega=None):
        batch_size = features.size(0)
        seqlen = features.size(2)
        embedding = self.resnet(features)

        # anglegrams for theta
        logits_theta = self.predict_theta(embedding)

        # anglegrams for phi
        logits_phi = self.predict_phi(embedding)

        # symmetrize
        sym_embedding = 0.5 * (embedding + embedding.transpose(-1, -2))

        # distograms
        logits_dist = self.predict_dist(sym_embedding)

        # beta-strand pairings (not used)
        # logits_bb = self.predict_bb(sym_embedding)

        # anglegrams for omega
        logits_omega = self.predict_omega(sym_embedding)

        logits_dist = logits_dist.permute(0, 2, 3, 1).contiguous()
        logits_theta = logits_theta.permute(0, 2, 3, 1).contiguous()
        logits_omega = logits_omega.permute(0, 2, 3, 1).contiguous()
        logits_phi = logits_phi.permute(0, 2, 3, 1).contiguous()

        probs = {}
        probs['p_dist'] = nn.Softmax(-1)(logits_dist)
        probs['p_theta'] = nn.Softmax(-1)(logits_theta)
        probs['p_omega'] = nn.Softmax(-1)(logits_omega)
        probs['p_phi'] = nn.Softmax(-1)(logits_phi)
        outputs = (probs,)

        metrics = {}
        total_loss = 0

        if dist is not None:
            logits_dist = logits_dist.reshape(batch_size * seqlen * seqlen, 37)
            loss_dist = nn.CrossEntropyLoss(ignore_index=-1)(logits_dist, dist.view(-1))
            metrics['dist'] = loss_dist
            total_loss += loss_dist
        if theta is not None:
            logits_theta = logits_theta.reshape(batch_size * seqlen * seqlen, 25)
            loss_theta = nn.CrossEntropyLoss(ignore_index=0)(logits_theta, theta.view(-1))
            metrics['theta'] = loss_theta
            total_loss += loss_theta
        if omega is not None:
            logits_omega = logits_omega.reshape(batch_size * seqlen * seqlen, 25)
            loss_omega = nn.CrossEntropyLoss(ignore_index=0)(logits_omega, omega.view(-1))
            metrics['omega'] = loss_omega
            total_loss += loss_omega
        if phi is not None:
            logits_phi = logits_phi.reshape(batch_size * seqlen * seqlen, 13)
            loss_phi = nn.CrossEntropyLoss(ignore_index=0)(logits_phi, phi.view(-1))
            metrics['phi'] = loss_phi
            total_loss += loss_phi

        if len(metrics) > 0:
            outputs = ((total_loss, metrics),) + outputs

        return outputs


@registry.register_task_model('trrosetta', 'trrosetta')
class TRRosetta(TRRosettaAbstractModel):

    def __init__(self, config: TRRosettaConfig):
        super().__init__(config)
        self.extract_features = MSAFeatureExtractor(config)
        self.trrosetta = TRRosettaPredictor(config)

    def forward(self,
                msa1hot,
                theta=None,
                phi=None,
                dist=None,
                omega=None):
        features = self.extract_features(msa1hot)
        return self.trrosetta(features, theta, phi, dist, omega)
