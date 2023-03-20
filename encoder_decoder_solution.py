import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_in = nn.Parameter(torch.empty(hidden_size, input_size))

        self.b_ir = nn.Parameter(torch.empty(hidden_size))
        self.b_iz = nn.Parameter(torch.empty(hidden_size))
        self.b_in = nn.Parameter(torch.empty(hidden_size))

        self.w_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.b_hr = nn.Parameter(torch.empty(hidden_size))
        self.b_hz = nn.Parameter(torch.empty(hidden_size))
        self.b_hn = nn.Parameter(torch.empty(hidden_size))
        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/hidden_size)**0.5, b=(1/hidden_size)**0.5)

        # self.x2h = nn.Linear(input_size, 3 * hidden_size,)
        # self.h2h = nn.Linear(hidden_size, 3 * hidden_size,)

    
    def forward(self, inputs, hidden_states):
        """GRU.
        
        This is a Gated Recurrent Unit
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`)
          The input tensor containing the embedded sequences. 
          input_size corresponds to embedding size.
          
        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The (initial) hidden state.
          
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
          A feature tensor encoding the input sentence. 
          
        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The final hidden state. 
        """
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]
        hidden_size = hidden_states.shape[2]

        # initialize to zeros for later updates
        all_sequence_encoding = torch.zeros(batch_size, seq_length, hidden_size)
        final_hidden_states = torch.zeros(hidden_states.shape)


        # iterate over batches
        for i, seq in enumerate(inputs):
            # seq is input matrix of dimension sequence_length x input size
            # h is hidden state vector of length hidden_size
            
            # initialize encoding to be zeros for later update
            h = hidden_states[:, i]
            seq_encoding = torch.zeros(seq_length, hidden_size)
            for j, x in enumerate(seq): # x is a word vector of length 'input_size'
                
                # h = hidden_states[:, i]
                # reset gate
                r = torch.sigmoid(torch.matmul(x, self.w_ir.T) + self.b_ir + torch.matmul(h, self.w_hr.T) + self.b_hr) 

                # update gate
                z = torch.sigmoid(torch.matmul(x, self.w_iz.T) + self.b_iz + torch.matmul(h, self.w_hz.T) + self.b_hz) 

                # reset gate output
                n = torch.tanh(torch.matmul(x, self.w_in.T) + self.b_in + r*(torch.matmul(h,self.w_hn.T) + self.b_hn))

                h = (1-z)*n + z*h

                seq_encoding[j] = h

            all_sequence_encoding[i] = seq_encoding
            final_hidden_states[:, i] = h

        return all_sequence_encoding, final_hidden_states


class Attn(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        dropout=0.0 # note, this is an extrenous argument
        ):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(hidden_size*2, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size) # in the forwards, after multiplying
                                                     # do a torch.sum(..., keepdim=True), its a linear operation

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs, hidden_states, mask = None):
        """Soft Attention mechanism.

        This is a one layer MLP network that implements Soft (i.e. Bahdanau) Attention with masking
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence with attention applied.

        x_attn (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`)
            The attention vector.

  
        """
        # ==========================
        # TODO: Write your code here
        # ==========================


        # WHERE DOES MASKING COME IN?

        # iterate over batches

        batch_size, seq_length, hidden_size = inputs.shape

        # reshaping to get 'sequence_length' many copies of the hidden states vector
  
        hidden_reshaped = hidden_states[0].reshape(batch_size, 1, hidden_size).repeat(1, seq_length, 1)


        input_attn = torch.cat([inputs, hidden_reshaped], dim = 2)



        # hidden size is 64, W input is hidden_size *2 x hidden_size
        # why is input allowed to be 256 x 128???

        # multiply attention input times W matrix, get tanh
        Q = self.tanh(self.W(input_attn))
        K = self.V(Q)
        attn_sum = torch.sum(K, dim=2, keepdim=True)

        attn_vector = self.softmax(attn_sum)

        output = attn_vector * inputs

        return output, attn_vector





class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0
        ):
        super(Encoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(input_size=self.embedding_size, 
                          hidden_size=self.hidden_size, 
                          num_layers=self.num_layers,
                          bidirectional=True, 
                          batch_first=True)

    def forward(self, inputs, hidden_states):
        """GRU Encoder.

        This is a Bidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            The input tensor containing the token sequences.

        hidden_states(`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)
            The (initial) hidden state for the bidrectional GRU.
            
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # batch_size, seq_length, vocab_size = inputs.shape
        _, _ , hidden_size= hidden_states.shape
        input_embedding = self.embedding(inputs)
        #embedding dropout
        x = self.dropout(input_embedding)

        rnn_out, hidden_states = self.rnn(x, hidden_states)
        
        x_forward = rnn_out[:, :, 0:hidden_size]
        x_backward = rnn_out[:, :, hidden_size:]
        output =  x_forward + x_backward



        h_forward = hidden_states[0]
        h_backward = hidden_states[1]
        final_hidden_states = (h_forward + h_backward).unsqueeze(0)
        return output, final_hidden_states
    


    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers*2, batch_size, self.hidden_size)
        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        return h_0

class DecoderAttn(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0, 
        ):

        super(DecoderAttn, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(input_size=self.embedding_size, 
                          hidden_size=self.hidden_size, 
                          num_layers=self.num_layers,
                        #   bidirectional=True, 
                          batch_first=True)
        
        self.mlp_attn = Attn(hidden_size, dropout)

    def forward(self, inputs, hidden_states, mask=None):
        """GRU Decoder network with Soft attention

        This is a Unidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the encoded input sequence.

        hidden_states(`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)
            The (initial) hidden state for the bidrectional GRU.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        inputs = self.dropout(inputs)
        attn_output, _ = self.mlp_attn(inputs, hidden_states, mask)
        decoder_output, hidden_states = self.rnn(attn_output, hidden_states)
        return decoder_output, hidden_states
        
        
class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout = 0.0,
        encoder_only=False
        ):
        super(EncoderDecoder, self).__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(vocabulary_size, embedding_size, hidden_size,
                num_layers, dropout=dropout)
        if not encoder_only:
          self.decoder = DecoderAttn(vocabulary_size, embedding_size, hidden_size, num_layers, dropout=dropout)
        
    def forward(self, inputs, mask=None):
        """GRU Encoder-Decoder network with Soft attention.

        This is a Gated Recurrent Unit network for Sentiment Analysis. This
        module returns a decoded feature for classification. 
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The final hidden state. 
        """
        hidden_states = self.encoder.initial_states(inputs.shape[0])
        x, hidden_states = self.encoder(inputs, hidden_states)
        if self.encoder_only:
          x = x[:, 0]
          return x, hidden_states
        x, hidden_states = self.decoder(x, hidden_states, mask)
        x = x[:, 0]
        return x, hidden_states
