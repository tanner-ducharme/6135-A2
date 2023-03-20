import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gradescope_utils.autograder_utils.decorators import weight
from encoder_decoder_solution import Attn, EncoderDecoder, GRU

torch.random.manual_seed(0)

class TestEncoderDecoder(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(6135)
        self.vocabulary_size = 30522
        self.embedding_size = 256
        self.hidden_size = 256
        self.num_layers = 1
        self.dropout = 0.0
        self.batch_size = 5

        self.model = EncoderDecoder(
            self.vocabulary_size,
            self.embedding_size,
            self.hidden_size,
            self.num_layers,
            self.dropout)

        self._x = torch.randint(0, self.vocabulary_size, (self.batch_size, 256))
        
    def _encoder_forward(self, inputs, hidden_states):
        x = self.model.encoder.embedding(inputs)
        #embedding dropout
        x = self.model.encoder.dropout(x)
        x, hidden_states = self.model.encoder.rnn(x, hidden_states)
        x = x[..., 0:x.shape[-1]//2] +x[..., x.shape[-1]//2:]
        hidden_states = (hidden_states[0] + hidden_states[1]).unsqueeze(0)
        return x, hidden_states

    def _attn_forward(self, inputs, hidden_states, mask=None):
        shape = inputs.shape
        hidden = hidden_states.reshape(shape[0], shape[-1]).unsqueeze(1).repeat(1, shape[1], 1)
        attn_input = torch.cat([inputs, hidden], dim = -1)
        a = self.model.decoder.mlp_attn.W(attn_input)
        b = self.model.decoder.mlp_attn.tanh(a)
        c = self.model.decoder.mlp_attn.V(b)
        c = torch.sum(c, -1, keepdim=True)

        x_attn = self.model.decoder.mlp_attn.softmax(c)
        x = x_attn * inputs
        return x, x_attn


    def _decoder_forward(self, inputs, hidden_states, mask=None):
        inputs = self.model.decoder.dropout(inputs)
        x_new, x_attn = self._attn_forward(inputs, hidden_states, mask)
        x, hidden_states = self.model.decoder.rnn(x_new, hidden_states)
        return x, hidden_states

    @weight(2)
    def test_encoder_forward(self):
        print("Testing if the output of Encoder matches the expected output.")
        hidden_states = self.model.encoder.initial_states(self._x.shape[0])
        inputs = self._x
        outputs_sol, out_hidden = self.model.encoder(inputs, hidden_states)
        outputs_gt, out_hidden_gt  = self._encoder_forward(inputs, hidden_states)

        np.testing.assert_almost_equal(
            outputs_sol.detach().numpy(), outputs_gt.detach().numpy(), decimal=3
        )
        np.testing.assert_almost_equal(
            out_hidden.detach().numpy(), out_hidden_gt.detach().numpy(), decimal=3
        )


    @weight(1)
    def test_encoder_decoder_forward(self):
        self.setUp()
        print("Testing if the output of Encoder Decoder matches the expected output.")
        hidden_states = self.model.encoder.initial_states(self._x.shape[0])
        inputs = self._x
        outputs_sol, out_hidden = self.model.encoder(inputs, hidden_states)
        outputs_sol, out_hidden  = self.model.decoder(outputs_sol, out_hidden)


        outputs_gt, out_hidden_gt  = self._encoder_forward(inputs, hidden_states)
        outputs_gt, out_hidden_gt  = self._decoder_forward(outputs_gt, out_hidden_gt)


        np.testing.assert_almost_equal(
            outputs_sol.detach().numpy(), outputs_gt.detach().numpy(), decimal=3
        )
        np.testing.assert_almost_equal(
            out_hidden.detach().numpy(), out_hidden_gt.detach().numpy(), decimal=3
        )


    @weight(3) # 1 point missing for masked attn
    def test_attn_unmasked_forward(self):
        self.setUp()
        print("Testing if the output of Attn matches the expected output.")
        hidden_states = self.model.encoder.initial_states(self._x.shape[0])
        inputs = self._x
        inputs[:, 200:] = 0.0

        outputs_sol, out_hidden = self.model.encoder(inputs, hidden_states)
        outputs_sol, out_attn  = self.model.decoder.mlp_attn(outputs_sol,
                out_hidden)

        outputs_gt, out_hidden_gt  = self._encoder_forward(inputs, hidden_states)
        outputs_gt, out_attn_gt  = self._attn_forward(outputs_gt,
                out_hidden_gt)

        np.testing.assert_almost_equal(
            outputs_sol.detach().numpy(), outputs_gt.detach().numpy(), decimal=3
        )
        np.testing.assert_almost_equal(
            out_hidden.detach().numpy(), out_hidden_gt.detach().numpy(), decimal=3
        )
        np.testing.assert_almost_equal(
            out_attn.detach().numpy(), out_attn_gt.detach().numpy(), decimal=3
        )


    @weight(1)
    def test_bidirectional_gru_encoder(self):
        self.setUp()
        print(
            "Testing if the encoder rnn is a GRU that is also Bidirectional."
        )
        assert self.model.encoder.rnn.bidirectional
        assert isinstance(self.model.encoder.rnn, nn.GRU)


    @weight(1)
    def test_unidirectional_gru_decoder(self):
        self.setUp()
        print(
            "Testing if the decoder rnn is a GRU that is also Unidirectional."
        )
        assert not self.model.decoder.rnn.bidirectional
        assert isinstance(self.model.decoder.rnn, nn.GRU)

class TestGRU(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(6135)
        gru = nn.GRU(4, 5, 1, batch_first=True)
        self.model = GRU(4, 5)
        self._x = torch.rand(3, 2, 4)
        hid = 5
        self._gru_gt = gru
        self._hid = torch.zeros(1, 3, 5)

        self.model.w_ir, self.model.w_iz, self.model.w_in = nn.Parameter(gru.weight_ih_l0[0:hid]), nn.Parameter(gru.weight_ih_l0[hid:hid*2]), nn.Parameter(gru.weight_ih_l0[hid*2:])
        self.model.w_hr, self.model.w_hz, self.model.w_hn = nn.Parameter(gru.weight_hh_l0[0:hid]), nn.Parameter(gru.weight_hh_l0[hid:hid*2]), nn.Parameter(gru.weight_hh_l0[hid*2:])
        self.model.b_ir, self.model.b_iz, self.model.b_in = nn.Parameter(gru.bias_ih_l0[0:hid]), nn.Parameter(gru.bias_ih_l0[hid*1:hid*2]), nn.Parameter(gru.bias_ih_l0[hid*2:])
        self.model.b_hr, self.model.b_hz, self.model.b_hn = nn.Parameter(gru.bias_hh_l0[0:hid]), nn.Parameter(gru.bias_hh_l0[hid:hid*2]), nn.Parameter(gru.bias_hh_l0[hid*2:])
         
    @weight(6)
    def test_forward_gru(self):
        self.setUp()
        print(
            "Testing if the GRU returns the proper outputs"
        )
        outputs_gt, hid_states_gt = self._gru_gt(self._x, self._hid)
        outputs_student, hid_states_student = self.model(self._x, self._hid)
        np.testing.assert_almost_equal(
            outputs_student.detach().numpy(), outputs_gt.detach().numpy(), decimal=3
        )
        np.testing.assert_almost_equal(
            hid_states_student.detach().numpy(), hid_states_gt.detach().numpy(), decimal=3
        )
