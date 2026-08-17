import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen


class PathSNN(nn.Module):
    """
    SNN shallow e fully-connected, ispirata alla struttura del PPT:
    input -> Linear -> LIF -> Linear -> LIF/output.

    Input:  RGB image flattened to 32*32*3 = 3072 features.
    Output: 5 classes:
        0 FAR_LEFT
        1 LEFT
        2 CENTER
        3 RIGHT
        4 FAR_RIGHT
    """

    def __init__(
        self,
        input_size=32 * 32 * 3,
        hidden_size=128,
        num_outputs=5,
        beta=0.9,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs

        # Connections generated with PyTorch.
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism="subtract",
        )

        self.fc2 = nn.Linear(hidden_size, num_outputs)
        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=surrogate.fast_sigmoid(),
            reset_mechanism="subtract",
        )

    def forward(self, x, num_steps=20):
        """
        x shape:
            [batch, input_size], values in [0, 1].

        The input is converted to spikes using rate coding,
        as described in the PPT.
        """
        if x.ndim != 2:
            raise ValueError(
                f"Expected x with shape [batch, features], got {tuple(x.shape)}"
            )

        if x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, got {x.shape[1]}"
            )

        x = x.clamp(0.0, 1.0)

        # Rate coding: pixel intensity becomes spike probability.
        spike_train = spikegen.rate(
            x,
            num_steps=num_steps,
        )

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        # Different input element for every timestep.
        for step in range(num_steps):
            cur1 = self.fc1(spike_train[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


CLASS_NAMES = [
    "FAR_LEFT",
    "LEFT",
    "CENTER",
    "RIGHT",
    "FAR_RIGHT",
]

# Signed lateral positions used by the controller.
CLASS_POSITION = torch.tensor(
    [-1.0, -0.5, 0.0, 0.5, 1.0],
    dtype=torch.float32,
)
