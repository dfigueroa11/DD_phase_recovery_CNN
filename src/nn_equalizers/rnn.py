import torch
import torch.jit as jit
from torch import nn
from typing import List

############ code adapted from https://github.com/DPlabst/NN-MI
# 
# * ------ Time-varying recurrent neural network class ----------
#   _________      _______  _   _ _   _
#  |__   __\ \    / /  __ \| \ | | \ | |
#     | |   \ \  / /| |__) |  \| |  \| |
#     | |    \ \/ / |  _  /| . ` | . ` |
#     | |     \  /  | | \ \| |\  | |\  |
#     |_|      \/   |_|  \_\_| \_|_| \_|
# * ---------------------------------------------------------------
class RNNRX(jit.ScriptModule):
    def __init__(
        self,
        input_size,
        hidden_states_size,
        output_size,
        N_tv_cells: int,
        dev,
    ):
        super().__init__()
        self.dev = dev  # Save device
        self.mult_BIRNN = 2
        self.N_tv_cells = N_tv_cells
        self.TVRNN_layers = nn.ModuleList()  # Variable layer list
        for hidden_sz in hidden_states_size:
            self.TVRNN_layers.append(TVRNN(input_size, hidden_sz, self.N_tv_cells, dev))
            # Mult x 2 for next input dimension, because bidirectional RNN concatenates two previous output vectors
            input_size = hidden_sz * self.mult_BIRNN
    
        self.lin_layer_in = input_size
        self.Lin_layer = nn.Linear(self.lin_layer_in, output_size)

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for TVRNN_layer in self.TVRNN_layers:
            x = TVRNN_layer(x)
        # apply the lin layer only to the s-th stage ass in the papper
        # out = self.Lin_layer(x[:,::self.N_tv_cells,:]) 
        out = x.contiguous().view(-1, self.lin_layer_in)
        out = self.Lin_layer(out)
        return out

# * ------------------ Time-Varying RNN Layer ---------------------
class TVRNN(jit.ScriptModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        N_tv_cells: int,
        dev: str,
    ):
        super().__init__()
        self.dev = dev

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.N_tv_cells = N_tv_cells

        self.cells_fw = nn.ModuleList([CRNNCell(self.input_size, self.hidden_size, self.dev) for count in range(self.N_tv_cells)])
        self.cells_bw = nn.ModuleList([CRNNCell(self.input_size, self.hidden_size, self.dev) for count in range(self.N_tv_cells)])
            
    @jit.script_method
    def recurFW(self, x: torch.Tensor) -> torch.Tensor:
        # Input dim(x) = nBatch x N_Trnn x Nin
        # Initialize state with uniform random numbers
        # [https://pytorch.org/docs/stable/generated/torch.nn.RNN.html]
        ksc = torch.sqrt(torch.tensor(1 / self.hidden_size))
        h = 2 * ksc * torch.rand(x.size(dim=0), self.hidden_size, device=self.dev) - ksc

        inputs = x.unbind(dim=1)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs) // self.N_tv_cells):
            for k, tvrnn_cell in enumerate(self.cells_fw):
                h = tvrnn_cell(inputs[i * self.N_tv_cells + k], h)
                outputs += [h]
        # Because formerly, we unbound dim=1 (Trnn)
        return torch.stack(outputs, dim=1)

    @jit.script_method
    def recurBW(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize with uniform random numbers
        # [https://pytorch.org/docs/stable/generated/torch.nn.RNN.html]
        ksc = torch.sqrt(torch.tensor(1 / self.hidden_size))
        h = 2 * ksc * torch.rand(x.size(dim=0), self.hidden_size, device=self.dev) - ksc

        inputs = reverse(x.unbind(dim=1))  # Unbind Trnn
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs) // self.N_tv_cells):
            for k, tvrnn_cell in enumerate(self.cells_bw):
                h = tvrnn_cell(inputs[i * self.N_tv_cells + k], h)
                outputs += [h]
        # Because formerly, we unbound dim=1 (Trnn)
        return torch.stack(reverse(outputs), dim=1)

    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Parallelize FW and BW path, as they are independent
        # [https://pytorch.org/tutorials/advanced/torch-script-parallelism.html]

        ## --- Forward path
        future_f = torch.jit.fork(self.recurFW, x)

        ## --- Backward path
        out_bw = self.recurBW(x)

        out_fw = torch.jit.wait(future_f)  # Wait for FW path to finish

        ## -- Return concatenated
        return torch.cat((out_fw, out_bw), dim=2)

# * ----------------------- Custom RNN Cell ---------------------
# References for JIT; Code inspired by:
# [https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns/custom_lstms.py]
# [https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/]
class CRNNCell(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, dev):
        super().__init__()
        self.Lin_layer_input_to_hidden = nn.Linear(input_size, hidden_size, device=dev)
        self.Lin_layer_hidden_to_hidden = nn.Linear(hidden_size, hidden_size, device=dev)

    @jit.script_method
    def forward(self, input: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        hx = self.Lin_layer_input_to_hidden(input) + self.Lin_layer_hidden_to_hidden(hx)
        return torch.relu(hx)

def reverse(lst: List[torch.Tensor]) -> List[torch.Tensor]:
    return lst[::-1]
