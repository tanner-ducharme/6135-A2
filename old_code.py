
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def forward(self, inputs, hidden_states):

    all_sequence_encoding = torch.zeros(inputs.shape[0], inputs.shape[1], hidden_states.shape[2])
    final_hidden_states = torch.zeros(hidden_states.shape)

    # iterate over batches
    for i, (seq, h) in enumerate(zip(inputs, hidden_states[0])):

        seq_encoding = torch.zeros(inputs.shape[1], h.shape[0])
        for j, x in enumerate(seq): # x is a word vector of length 'input_size'
            # x = x.reshape((1, x.shape[0]))


            # reset gate
            r = torch.sigmoid(torch.matmul(x, self.w_ir.T) + self.b_ir + torch.matmul(h, self.w_hr) + self.b_hr) 

            # update gate
            z = torch.sigmoid(torch.matmul(x, self.w_iz.T) + self.b_iz + torch.matmul(h, self.w_hz) + self.b_hz) 

            # reset gate output
            n = torch.tanh(torch.matmul(x, self.w_in.T) + self.b_in + r*(torch.matmul(h,self.w_hn.T) + self.b_hn))

            h = (1-z)*n + z*h

            seq_encoding[j] = h

    all_sequence_encoding[i] = seq_encoding
    final_hidden_states[:, i] = h

    return all_sequence_encoding, final_hidden_states
