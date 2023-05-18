from typing import List, Tuple, Callable, Union, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.mlp import MLP
from model.sequence import AutoregressiveFlow


class FlowNet(nn.Module):
    def __init__(self, future_lens,
                 input_embedding_size=16,
                 hidden_size=32,
                 output_sizes=[32,4]):

        super().__init__()

        self.future_lens = future_lens
        self.hidden_size = hidden_size

        self.output_shape = (future_lens, 2)

        self.hist_embedding = nn.Linear(self.output_shape[-1], input_embedding_size)
        self.history_encoder = nn.LSTM(input_size=input_embedding_size,
                                      hidden_size=hidden_size)

        self._decoder = AutoregressiveFlow(
                            output_shape=self.output_shape,
                            hidden_size=hidden_size,
                            output_sizes=output_sizes
                        )
        self._decoder = self._decoder.cuda()


    # def to(self, *args, **kwargs):
    #     """Handles non-parameter tensors when moved to a new device."""
    #     self = super().to(*args, **kwargs)
    #     self._decoder = self._decoder.to(*args, **kwargs)
    #     return self

    def encode_history(self, history_tensor):
        assert (len(history_tensor.size()) == 3)
        hist_batch = history_tensor.transpose(0, 1)     # (obs_len, B, 2)
        output, (z, c) = self.history_encoder(F.leaky_relu(self.hist_embedding(hist_batch)))        # output: (obs_len, B, 32)
                                                                                                    # z: (1, B, 32)   c: (1, B, 32)
        return z[0]

    def forward(self, history_tensor, future_tensor):
        y_tm1 = history_tensor[:, -1]       # (B, 2)
        history_features = self.encode_history(history_tensor)      # (B, 32)
        _, log_prob, logabsdet = self._decoder._inverse(y_tm1, y=future_tensor, z=history_features)

        return log_prob, logabsdet

    def predict(self, history_tensor, deterministic=False):
        y_tm1 = history_tensor[:, -1]
        history_features = self.encode_history(history_tensor)
        if deterministic:
            x = self._decoder._base_dist.loc.repeat(history_features.shape[0], 1)
            x = x.reshape(-1, self.future_lens, 2)
            agent_future_hat = self._decoder._forward(y_tm1, x, history_features)[0]
        else:
            agent_future_hat = self._decoder.forward(y_tm1, history_features)
        return agent_future_hat

    def predict_n(self, history_tensor, n=5, sigma=1.0):
        y_tm1 = history_tensor[:, -1]       # (B, 2)
        history_features = self.encode_history(history_tensor)      # (B, 32)

        y_tm1 = torch.repeat_interleave(y_tm1, repeats=n, dim=0)        # (B*n, 2)
        history_features_n = torch.repeat_interleave(history_features, repeats=n, dim=0)        # (B*n, 32)
        agent_future_hat = self._decoder.forward(y_tm1, history_features_n, sigma=sigma)        # (B*n, pred_len, 2)
        return agent_future_hat