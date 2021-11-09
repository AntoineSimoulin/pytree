import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity(nn.Module):
    def __init__(self, encoder, config):
        super(Similarity, self).__init__()
        self.config = config
        self.encoder = encoder
        self.hidden_similarity_size = config.hidden_similarity_size
        self.hidden_size = encoder.config.hidden_size
        self.num_classes = config.num_classes
        self.wh = nn.Linear(2 * self.hidden_size, self.hidden_similarity_size)
        # self.wi = nn.Linear(self.hidden_similarity_size, self.hidden_similarity_size)  # added from Conneau, et al. (2018)
        # self.wii = nn.Linear(self.hidden_similarity_size,
        #                     self.hidden_similarity_size)  # added from Choi, et al. (2018)
        self.wp = nn.Linear(self.hidden_similarity_size, self.num_classes)

        # self.bn_mlp_input = nn.BatchNorm1d(num_features=4 * self.hidden_size)  # added from Choi, et al. (2018)
        # self.bn_mlp_output = nn.BatchNorm1d(num_features=self.hidden_similarity_size)  # added from Choi, et al. (2018)
        # self.dropout = nn.Dropout(0.2)  # added from Choi, et al. (2018)
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.device = next(self.encoder.parameters()).device

    def forward(self, **kwargs):
        inputs = {a: kwargs[a] for a in kwargs}
        targets = inputs['labels']
        
        lhidden = self.encoder({k[:-2]: v.to(self.device) if torch.is_tensor(v) else v for (k, v) in inputs.items() if k.endswith('_A')})
        rhidden = self.encoder({k[:-2]: v.to(self.device) if torch.is_tensor(v) else v for (k, v) in inputs.items() if k.endswith('_B')})
        # lhidden = F.normalize(lhidden, p=2, dim=1)
        # rhidden = F.normalize(rhidden, p=2, dim=1)
        # output = self.similarity(lhidden, rhidden)
        mult_dist = torch.mul(lhidden, rhidden)
        abs_dist = torch.abs(torch.add(lhidden, -rhidden))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)  # lvec, rvec added from Conneau, et al. (2018)
        # vec_dist = torch.cat((mult_dist, abs_dist, lhidden, rhidden), 1)  # lvec, rvec added from Conneau, et al. (20

        # mlp_input = self.bn_mlp_input(vec_dist)  # added from Choi, et al. (2018)
        # mlp_input = self.dropout(mlp_input)  # added from Choi, et al. (2018)
        # outputs = torch.relu(self.wh(mlp_input))  # added from Choi, et al. (2018)
        # outputs = torch.relu(self.wi(outputs))  # added from Choi, et al. (2018)
        # outputs = torch.relu(self.wii(outputs))  # added from Choi, et al. (2018)
        # mlp_output = self.bn_mlp_output(outputs)  # added from Choi, et al. (2018)
        # mlp_output = self.dropout(mlp_output)  # added from Choi, et al. (2018)
        # outputs = F.log_softmax(self.wp(mlp_output), dim=1)  # added from Choi, et al. (2018)

        outputs = torch.relu(self.wh(vec_dist))  # added from Conneau, et al. (2018)
        # outputs = torch.relu(self.wi(outputs))  # added from Conneau, et al. (2018)
        outputs = F.log_softmax(self.wp(outputs), dim=1)  # added from Conneau, et al. (2018)
        # outputs = self.wp(outputs)
        loss = self.criterion(outputs, targets.to(self.device))
        return {'logits': outputs, 'loss': loss}