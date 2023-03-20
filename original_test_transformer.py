import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math 
from contextlib import contextmanager

from gradescope_utils.autograder_utils.decorators import weight

from transformer_solution import LayerNorm, MultiHeadedAttention,PreNormAttentionBlock, Transformer#, MiniGPT1


class ForbiddenFunction(Exception):
    pass


class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(6135)

        self.hidden_size = 13
        self.shape = (5, 7, 11)

        self.layer_norm = LayerNorm(self.hidden_size, eps=1e-5)

        self._layer_norm = nn.LayerNorm(
            self.hidden_size, eps=1e-5, elementwise_affine=True
        )
        self._inputs = 3 + 2 * torch.randn(*self.shape, self.hidden_size)
        self._weight = torch.randn(self.hidden_size)
        self._bias = torch.randn(self.hidden_size)

    @contextmanager
    def _disable_torch_layer_norm(self):
        def torch_layer_norm(*args, **kwargs):
            raise ForbiddenFunction(
                "The function torch.nn.functional.layer_norm "
                "has been disabled. You are not allowed to use this function, "
                "please implement Layer Normalization yourself."
            )

        original_torch_layer_norm = torch.layer_norm
        torch.layer_norm = torch_layer_norm
        yield
        torch.layer_norm = original_torch_layer_norm

    @weight(2)
    def test_forward_no_scaling(self):
        self.setUp()
        print(
            "Testing if the output of LayerNorm matches the expected output when weight = 1 and bias = 0."
        )
        # Set the scaling to weight = 1 / bias = 0
        nn.init.ones_(self.layer_norm.weight)
        nn.init.ones_(self._layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
        nn.init.zeros_(self._layer_norm.bias)

        with self._disable_torch_layer_norm():
            outputs = self.layer_norm(self._inputs)

        expected_outputs = self._layer_norm(self._inputs)

        np.testing.assert_almost_equal(
            outputs.detach().numpy(), expected_outputs.detach().numpy(), decimal=3
        )

    @weight(2)
    def test_forward_normalization(self):
        self.setUp()
        print(
            "Testing if the output of LayerNorm has approximately zero mean and unit variance on the last dimension (when weight = 1 and bias = 0)."
        )
        # Set the scaling to weight = 1 / bias = 0
        nn.init.ones_(self.layer_norm.weight)
        nn.init.ones_(self._layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
        nn.init.zeros_(self._layer_norm.bias)

        with self._disable_torch_layer_norm():
            outputs = self.layer_norm(self._inputs)

        mean = torch.mean(outputs, dim=-1)
        variance = torch.var(outputs, dim=-1, unbiased=False)

        np.testing.assert_almost_equal(mean.detach().numpy(), 0.0, decimal=4)
        np.testing.assert_almost_equal(variance.detach().numpy(), 1.0, decimal=4)

    @weight(1)
    def test_forward_scaling(self):
        self.setUp()
        print("Testing if the output of LayerNorm matches the expected output.")
        # Set the scaling weight / bias
        self.layer_norm.weight.data = self._weight
        self._layer_norm.weight.data = self._weight
        self.layer_norm.bias.data = self._bias
        self._layer_norm.bias.data = self._bias

        with self._disable_torch_layer_norm():
            outputs = self.layer_norm(self._inputs)

        expected_outputs = self._layer_norm(self._inputs)

        np.testing.assert_almost_equal(
            outputs.detach().numpy(), expected_outputs.detach().numpy(), decimal=3
        )


class TestMultiHeadedAttention(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(6135)

        self.head_size = 13
        self.num_heads = 17
        self.sequence_length = 23
        self.batch_size = 7

        self.attention = MultiHeadedAttention(
            self.head_size, self.num_heads, self.sequence_length
        )

        self._dim = 11
        self._tensor_merged = torch.randn(
            self.batch_size, self.sequence_length, self.num_heads * self._dim
        )
        self._tensor_split = self._split_heads(self._tensor_merged)
        self._hidden_states = torch.randn(
            self.batch_size, self.sequence_length, self.num_heads * self.head_size
        )
        self._queries = torch.randn(
            self.batch_size, self.num_heads, self.sequence_length, self.head_size
        )
        self._keys = torch.randn(
            self.batch_size, self.num_heads, self.sequence_length, self.head_size
        )
        self._values = torch.randn(
            self.batch_size, self.num_heads, self.sequence_length, self.head_size
        )

    def _get_attention_weights(self, queries, keys):
        # causal_mask = torch.triu(
        #     torch.ones((self.sequence_length, self.sequence_length), dtype=torch.bool),
        #     diagonal=1,
        # )
        dot_products = torch.matmul(queries, keys.transpose(2, 3))
        dot_products /= math.sqrt(self.head_size)

        #dot_products.masked_fill_(causal_mask, -1e4)
        attention_weights = F.softmax(dot_products, dim=3)
        return attention_weights

    def _apply_attention(self, queries, keys, values):
        attention_weights = self._get_attention_weights(queries, keys)
        outputs = torch.matmul(attention_weights, values)
        return self._merge_heads(outputs)

    def _split_heads(self, tensor):
        tensor = tensor.view((*tensor.shape[:2], self.num_heads, -1))
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor):
        tensor = tensor.transpose(1, 2)
        return tensor.reshape((tensor.size(0), self.sequence_length, -1))

    @contextmanager
    def _disable_F_multi_head_attention_forward(self):
        def F_multi_head_attention_forward(*args, **kwargs):
            raise ForbiddenFunction(
                "The function torch.nn.functional.multi_head_attention_forward "
                "has been disabled. You are not allowed to use this function, "
                "please implement Multi-headed attention yourself."
            )

        original_F_multi_head_attention_forward = F.multi_head_attention_forward
        F.multi_head_attention_forward = F_multi_head_attention_forward
        yield
        F.multi_head_attention_forward = original_F_multi_head_attention_forward

    # Split heads & Merge heads
    # -------------------------
    @weight(1.5)
    def test_split_heads(self):
        self.setUp()
        print("Testing if the output of `split_heads` matches the expected output.")
        output = self.attention.split_heads(self._tensor_merged)
        np.testing.assert_equal(output.numpy(), self._tensor_split.numpy())

    @weight(1.5)
    def test_merge_heads(self):
        self.setUp()
        print("Testing if the output of `merge_heads` matches the expected output.")
        output = self.attention.merge_heads(self._tensor_split)
        np.testing.assert_equal(output.numpy(), self._tensor_merged.numpy())

    @weight(1)
    def test_split_merge_inverse(self):
        self.setUp()
        print(
            "Testing if `split_heads` and `merge_heads` are inverse from one another (e.g. merge_heads(split_heads(tensor)) == tensor)."
        )
        split_output = self.attention.split_heads(self._tensor_merged)
        merged_output = self.attention.merge_heads(split_output)
        np.testing.assert_equal(merged_output.numpy(), self._tensor_merged.numpy())

        merged_output = self.attention.merge_heads(self._tensor_split)
        split_output = self.attention.split_heads(merged_output)
        np.testing.assert_equal(split_output.numpy(), self._tensor_split.numpy())

    # Attention weights
    # -----------------

    @weight(3)
    def test_get_attention_weights_valid_attention(self):
        self.setUp()
        print("Testing if the output of `get_attention_weights` are valid attentions.")
        with self._disable_F_multi_head_attention_forward():
            weights = self.attention.get_attention_weights(self._queries, self._keys)

        assert torch.all(weights >= 0.0)
        assert torch.all(weights <= 1.0)

        sum_weights = torch.sum(weights, dim=3)
        np.testing.assert_almost_equal(sum_weights.detach().numpy(), 1.0, decimal=5)

    # @weight(2)
    # def test_get_attention_weights_causal_attention(self):
    #     print(
    #         "Testing if the output of `get_attention_weights` are valid causal attentions."
    #     )
    #     with self._disable_F_multi_head_attention_forward():
    #         weights = self.attention.get_attention_weights(self._queries, self._keys)

    #     # Check if the (strict) upper triangle is zero
    #     upper_triangle = np.triu(
    #         np.ones((self.sequence_length, self.sequence_length)), k=1
    #     )
    #     np.testing.assert_almost_equal(
    #         weights.detach().numpy() * upper_triangle, 0.0, decimal=5
    #     )
    @weight(5)
    def test_get_attention_weights(self):
        self.setUp()
        print(
            "Testing if the output of `get_attention_weights` matches the expected output."
        )
        with self._disable_F_multi_head_attention_forward():
            weights = self.attention.get_attention_weights(self._queries, self._keys)

        expected_weights = self._get_attention_weights(self._queries, self._keys)

        np.testing.assert_almost_equal(
            weights.detach().numpy(), expected_weights.detach().numpy(), decimal=3
        )

    # Apply attention
    # ---------------
    @weight(2)
    def test_apply_attention(self):
        self.setUp()
        print("Testing if the output of `apply_attention` matches the expected output.")
        with self._disable_F_multi_head_attention_forward():
            outputs = self.attention.apply_attention(
                self._queries, self._keys, self._values
            )

        expected_outputs = self._apply_attention(
            self._queries, self._keys, self._values
        )

        np.testing.assert_almost_equal(
            outputs.detach().numpy(), expected_outputs.detach().numpy(), decimal=3
        )

    # Forward function
    # ----------------
    @weight(2)
    def test_num_parameters(self):
        self.setUp()
        print(
            "Testing if the number of learnable parameters matches the expected number."
        )
        num_parameters = sum(
            param.numel()
            for param in self.attention.parameters()
            if param.requires_grad
        )

        # 4 linear layers, with weights & biases
        hidden_size = self.num_heads * self.head_size
        expected_num_parameters = 4 * (hidden_size * hidden_size + hidden_size)

        assert num_parameters == expected_num_parameters

    @weight(1)
    def test_forward_backward(self):
        self.setUp()
        print(
            "Testing if backpropagation through the module works as intended (i.e. the gradients wrt. the module parameters are defined when `backward` is called)."
        )
        self.attention.zero_grad()
        with self._disable_F_multi_head_attention_forward():
            outputs = self.attention(self._hidden_states)

        loss = outputs.sum()  # Dummy loss
        loss.backward()

        for name, param in self.attention.named_parameters():
            if param.requires_grad:
                assert (
                    param.grad is not None
                ), "Parameter {0} has no gradient " "(.grad = None).".format(name)
        self.attention.zero_grad()


class TestPreNormAttentionBlock(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(6135)
        self.head_size = 13
        self.num_heads = 17
        self.sequence_length = 23
        self.batch_size = 7

        
        self.num_heads = 17
        self.embed_dim = 23 * self.num_heads
        
        self.hidden_dim = 23 * self.num_heads*2
        self.prenorm = PreNormAttentionBlock(self.embed_dim, self.hidden_dim, self.num_heads,self.sequence_length)
       
        self._x = torch.randn(
            self.batch_size, self.sequence_length, self.embed_dim
        )
        
    def _forward(self, x):
        inp_x = self.prenorm.layer_norm_1(x)
        x = x + self.prenorm.attn(inp_x)
        x = x + self.prenorm.linear(self.prenorm.layer_norm_2(x))
        return x

    @weight(1)
    def test_forward_expected(self):
        self.setUp()
        print(
            "Testing if the output of the PreNorm forward function matches the expected output."
        )
        log_probas = self.prenorm(self._x)

        # Expected results (using the same model weights from self.model)
        expected = self._forward(self._x)

        # Testing log-probabilities
        np.testing.assert_almost_equal(
            log_probas.detach().numpy(), expected.detach().numpy(), decimal=4
        )


class TestTransformer(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(6135)
        self.vocab_size = 30522
        self.num_heads = 17
        self.num_layers = 3
        self.batch_size = 5
        self.embed_dim=24 * self.num_heads
        self.hidden_dim= 24 * self.num_heads *2
        self.block = 'prenorm'
        self.dropout = 0.0

        self.model = Transformer(
            self.vocab_size,
            self.embed_dim,
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
            self.block,
            self.dropout)

        self._x = torch.randint(0, self.vocab_size, (self.batch_size, 256))
        
    def _forward(self, inputs):
         # Preprocess input
        x = self.model.embedding(inputs)
        B, T, _ = x.shape
        
        # Add CLS token and positional encoding
        cls_token = self.model.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.model.pos_embedding[:,:T+1]
        
        # Apply Transforrmer
        x = self.model.dropout(x)
        for encoder in self.model.transformer:
            x = encoder(x)

        # Perform classification prediction
        x = x.transpose(0, 1)
        cls = x[0]
        out = self.model.mlp_head(cls)
        #out = self.model.mlp_head(x[:, 0])
        return out

    # Forward function
    # ----------------

    # @weight(1)
    # def test_forward_valid_log_probabilities(self):
    #     print(
    #         "Testing if the output of the forward function are valid log-probabilities."
    #     )
    #     log_probas = self.model(self._x)

    #     assert torch.all(log_probas <= 0.0)
    #     sum_probas = torch.sum(log_probas.exp(), dim=2)
    #     np.testing.assert_almost_equal(sum_probas.detach().numpy(), 1.0, decimal=5)
    
    @weight(2)
    def test_forward_expected_rep(self):
        self.setUp()
        print(
            "Testing if the output of the forward function matches the expected output."
        )
        torch.random.manual_seed(6135)
        rep = self.model(self._x)

        # Expected results (using the same model weights from self.model)
        torch.random.manual_seed(6135)
        expected_rep = self._forward(self._x)

        # Testing log-probabilities
        np.testing.assert_almost_equal(
            rep.detach().numpy(), expected_rep.detach().numpy(), decimal=4
        )
#     @weight(0.5)
#     def test_loss_expected_mask(self):
#         print("Testing if the output of the loss function matches the expected loss.")
#         # Build a random mask
#         lengths = torch.randint(1, self.sequence_length, size=(self.batch_size,))
#         mask = torch.zeros((self.batch_size, self.sequence_length))
#         for i, length in enumerate(lengths):
#             mask[i, :length].fill_(1.0)

#         # Ensure that masked values are zero-padding
#         targets = self._targets.clone().mul_(mask.long())

#         loss = self.model.loss(self._log_probas, targets, mask)

#         # Expected loss
#         expected_loss = self._loss(self._log_probas, targets, mask)

#         np.testing.assert_almost_equal(loss.item(), expected_loss.item(), decimal=4)