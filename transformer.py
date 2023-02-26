import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


#German (de) to English (en) Translation


de_vocab_size = 20000     #input vocab size
en_vocab_size = 10000     #output vocab size
max_in_len = 128             #input sequence length: maximum words in the input sequence
max_out_len = 256            #output sequence length: maximum words in the output sequence
embed_dim=512
enc_layers=6
dec_layers=6
num_heads=8
batch_size = 1

enc_tensor_shape = (batch_size, max_in_len, de_vocab_size)
dec_tensor_shape = (batch_size, max_out_len, en_vocab_size)

class TransformerModel(nn.Module):
    def __init__(
        self,
        en_vocab_size = en_vocab_size,
        de_vocab_size = de_vocab_size,
        max_in_len = max_in_len,
        max_out_len = max_out_len,
        embed_dim = embed_dim,
        enc_layers = enc_layers,
        dec_layers = dec_layers ,
        num_heads = num_heads,
        enc_tensor_shape = enc_tensor_shape,
        dec_tensor_shape = dec_tensor_shape,
    ):
        """
        Initializes the TransformerModel.

        Args:
            en_vocab_size (int): size of the English vocabulary used in this exercise.
            de_vocab_size (int): size of the German vocabulary used in this exercise.
            max_len (int): maximum number of words in the input sentence.
                Notice that during training the input to the decoder
                will be shifted one token to the right starting
                with <sos>, and will therefore have a maximum length of max_len + 1.
            embed_dim (int): the dimension of the model.
            enc_layers (int): number of encoder blocks.
            dec_layers (int): number of decoder blocks.
            num_heads (int): number of heads in Multi-head attention blocks.

        """
        super().__init__()

        self.en_vocab_size = en_vocab_size
        self.de_vocab_size = de_vocab_size
        self.max_in_len = max_in_len
        self.max_out_len = max_out_len
        self.embed_dim = embed_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.num_heads = num_heads

        # Linear transformation used for embedding the input to the encoder
        self.enc_input_dense = nn.Linear(self.de_vocab_size, self.embed_dim)

        # (Learned) positional encoding to be added to the embedded encoder input
        self.enc_pos_enc = nn.Parameter(torch.zeros((1, self.max_in_len, self.embed_dim)))

        # List of hidden layers in the feed-forward network of different encoder blocks
        self.enc_increase_hidden = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim * 2) for _ in range(self.enc_layers)]
        )

        # List of output layers used in the feed-forward network of different encoder blocks
        self.enc_decrease_hidden = nn.ModuleList(
            [nn.Linear(self.embed_dim * 2, self.embed_dim) for _ in range(self.enc_layers)]
        )

        # List of final layer normalizations used in different encoder blocks
        self.enc_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(self.enc_layers)]
        )

        # List of Multi-head attention blocks used in different encoder blocks
        self.enc_att = nn.ModuleList(
            [
                MultiHeadAttention(self.num_heads, self.embed_dim)
                for _ in range(self.enc_layers)
            ]
        )

        # Linear transformation used for embedding the input to the decoder
        self.dec_input_dense = nn.Linear(self.en_vocab_size, self.embed_dim)

        # (Learned) positional encoding to be added to the embedded decoder input
        self.dec_pos_enc = nn.Parameter(torch.zeros((1, self.max_out_len, self.embed_dim)))

        # List of hidden layers in the feed-forward network of different decoder blocks
        self.dec_increase_hidden = nn.ModuleList(
            [nn.Linear(self.embed_dim, self.embed_dim * 2) for _ in range(self.dec_layers)]
        )

        # List of output layers in the feed-forward network of different decoder blocks
        self.dec_decrease_hidden = nn.ModuleList(
            [nn.Linear(self.embed_dim * 2, self.embed_dim) for _ in range(self.dec_layers)]
        )

        # List of final layer normalizations used in different decoder blocks
        self.dec_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(self.dec_layers)]
        )

        # List of (first) Multi-head attention blocks used in different decoder blocks
        self.dec_att = nn.ModuleList(
            [
                MultiHeadAttention(self.num_heads, self.embed_dim)
                for _ in range(self.dec_layers)
            ]
        )

        # List of (second) Multi-head attention blocks used in different decoder blocks
        self.enc_dec_att = nn.ModuleList(
            [
                MultiHeadAttention(self.num_heads, self.embed_dim)
                for _ in range(self.dec_layers)
            ]
        )

        # Final fully connected layer converting the final decoder output to predictions over the output vocabulary
        self.decoder_final_dense = nn.Linear(self.embed_dim, self.en_vocab_size)

    def forward(self, x1, x2):
        """Implement the forward pass of the Transformer model.

        Args:
            x1: with shape (batch_size, self.max_in_len, self.de_vocab_size)
            x2: with shape (batch_size, self.max__out_len + 1, self.en_vocab_size)

        Returns:
            Tuple of:
                decoding: final output of the model,
                          with shape (batch_size, self.max_out_len + 1, self.en_vocab_size)
                attention: attention_weights of the last multi-head attention block in decoder,
                           with shape (batch_size, self.max_out_len + 1, self.max_out_len)
        """

        # Embed inputs to embed dimension
        # START TODO #############
        # enc_input_emb = ...
        # dec_input_emb = ...
        enc_input_emb = self.enc_input_dense(x1)
        dec_input_emb = self.dec_input_dense(x2)
        # END TODO #############

        # Add positional encodings
        # START TODO #############
        # encoding = ...
        # decoding = ...
        encoding = enc_input_emb + self.enc_pos_enc
        decoding = dec_input_emb + self.dec_pos_enc
        # END TODO #############

        # Loop over the encoder blocks
        for i in range(self.enc_layers):
            # Encoder Self-Attention
            # In the ith encoder block:
            # 1) Pass encoding through the MultiHeadAttention block
            # 2) Pass the output through the FeedForward block
            # FeedForward block consists of:
            # 1 - a Linear layer doubling the embedding size
            # 2 - a ReLU activation function
            # 3 - a Linear layer halving the embedding size
            # 3) Sum up the outputs of the previous two steps
            # 4) pass the result through the encoder layer-normalization
            # START TODO #############
            q = encoding.clone().detach()
            v = encoding.clone().detach()
            k = encoding.clone().detach()
            enc_mult_out, _ = self.enc_att[i](q, v, k)
            enc_ffwd_out = self.enc_increase_hidden[i](enc_mult_out)
            enc_ffwd_out = torch.relu(enc_ffwd_out)
            enc_ffwd_out = self.enc_decrease_hidden[i](enc_ffwd_out)
            encoding = self.enc_layer_norm[i](enc_ffwd_out + enc_mult_out)
            # END TODO #############

        # Without decoder blocks, attention would be None
        attention = None
        # Loop over the decoder blocks
        for i in range(self.dec_layers):
            # Decoder Self-Attention
            # Repeat the steps in the encoder, but with three main differences:
            # 1) In each decoder block, there are two successive MultiHeadAttention blocks
            # 2) The first MultiHeadAttention is masked (you therefore need to set mask=True)
            # 3) The three inputs to the second MultiHeadAttention block are mixed:
            # two coming from the encoder, and
            # one coming from the previous MultiHeadAttention block in the decoder
            # START TODO #############
            dec_mult1_out, _ = self.dec_att[i](decoding.clone().detach(), decoding.clone().detach(),
                                               decoding.clone().detach(), mask=True)
            dec_mult2_out, attention = self.enc_dec_att[i](dec_mult1_out, encoding.clone().detach(),
                                                           encoding.clone().detach())
            dec_ffwd_out = self.dec_increase_hidden[i](dec_mult2_out)
            dec_ffwd_out = torch.relu(dec_ffwd_out)
            dec_ffwd_out = self.dec_decrease_hidden[i](dec_ffwd_out)
            decoding = self.dec_layer_norm[i](dec_ffwd_out + dec_mult2_out)
            # END TODO #############
        # Map the embedding dimension of the decoder output back to self.en_vocab_size
        # START TODO #############
        decoding = self.decoder_final_dense(decoding)
        # END TODO #############

        print("Total number of self-attention blocks: ", (enc_layers+dec_layers))
        print("Total Number of skip connections: ", (2*enc_layers + 3*dec_layers))
        print("Total number of elements in the output tensor of the last encoder: (input sequence length * embed_dims)", torch.numel(encoding))
        print("Encoder-Decoder Attention Weights dimensions softmax((Q * K.t)/√dk)): (num_heads, batch_size, output sequence length, input sequence length)", attention.shape)
        print("Encoder-Decoder Attention Weights matrix (for a single head) number of elements: (max_in_len * max_out_len)", torch.numel(attention)/num_heads)
        print("Number of output units of the final fully connected layer = number of features in a decoder output vector = output vocab size: ", decoding.shape[-1])
        print("Total number of weights in a single multi-headed self attention module: For all Q, K and V combined: (embed_dims * embed_dims*3) and for W_O: (embed_dims*embed_dims)", embed_dim*embed_dim*4)

        return decoding, attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, num_heads, embed_dim):
        """
        Initialize Multi-Head Attention module

        Args:
            num_heads (int): number of attention heads.
            embed_dim (int): the dimension of the model.
        """
        super().__init__()

        # Number of heads
        self.num_heads = num_heads

        # The dimension of the model
        self.embed_dim = embed_dim

        # W_Q of all heads stacked together for efficiency of dimensionality
        # (num_heads * d_head, num_heads * d_head) which is equal to (embed_dim, embed_dim)
        self.W_query = nn.Parameter(
            torch.normal(mean=torch.zeros((self.embed_dim, self.embed_dim)), std=1e-2)
        )

        # W_K of all heads stacked together for efficiency of dimensionality
        # (num_heads * d_head, num_heads * d_head) which is equal to (embed_dim, embed_dim)
        self.W_key = nn.Parameter(
            torch.normal(mean=torch.zeros((self.embed_dim, self.embed_dim)), std=1e-2)
        )
        # W_V of all heads stacked together for efficiency of dimensionality
        # (num_heads * d_head, num_heads * d_head) which is equal to (embed_dim, embed_dim)
        self.W_value = nn.Parameter(
            torch.normal(mean=torch.zeros((self.embed_dim, self.embed_dim)), std=1e-2)
        )
        # W_O of all heads stacked together for efficiency of dimensionality
        # (num_heads * d_head, num_heads * d_head) which is equal to (embed_dim, embed_dim)
        self.W_output = nn.Parameter(
            torch.normal(mean=torch.zeros((self.embed_dim, self.embed_dim)), std=1e-2)
        )

        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, query, key, value, mask=False):
        # Chunk size is equal to the dimensionality of each head (d_head)                           
        chunk_size = int(self.embed_dim / self.num_heads)                                           #512/8 = 64 => d_head = 64

        # Tuple of length equal to the attention heads, each containing the query tensor of
        # the corresponding head with dimensionality (1, max_len, d_head)
        multi_query = torch.matmul(query, self.W_query).split(
            split_size=chunk_size, dim=-1
        )
        # Queries tensor from all attention heads stacked together resulting in a torch.tensor
        # of shape (num_heads, 1, max_len, d_head)
        multi_query = torch.stack(multi_query, dim=0)

        # Tuple of length equal to the attention heads, each containing the key tensor of
        # the corresponding head with dimensionality (1, max_len, d_head)
        multi_key = torch.matmul(key, self.W_key).split(split_size=chunk_size, dim=-1)
        # Keys tensor from all attention heads stacked together resulting in a torch.tensor
        # of shape (num_heads, 1, max_len, d_head)
        multi_key = torch.stack(multi_key, dim=0)

        # Tuple of length equal to the attention heads, each containing the value tensor of
        # the corresponding head with dimensionality (1, max_len, d_head)
        multi_value = torch.matmul(value, self.W_value).split(
            split_size=chunk_size, dim=-1
        )
        # Values tensor from all attention heads stacked together resulting in a torch.tensor
        # of shape (num_heads, 1, max_len, d_head)
        multi_value = torch.stack(multi_value, dim=0)

        # Scaling factor used in scaled dot product attention formula calculated as
        # square root of d_head
        scaling_factor = torch.tensor(np.sqrt(multi_query.shape[-1]))                               #sq root of dk (which is basically d_head with value 64) = 8
        # Scaled dot product attention matrix for all attention heads
        # of dimensionality (num_heads, 1, max_len, max_len)
        dotp = torch.matmul(multi_query, multi_key.transpose(2, 3)) / scaling_factor
        # attention weights of all attention heads of dimensionality
        # (num_heads, 1, max_len, max_len)
        attention_weights = F.softmax(dotp, dim=-1)

        if mask:
            attention_weights = attention_weights.tril()
            attention_weights = attention_weights / attention_weights.sum(
                dim=3, keepdim=True
            )

        # sum of values weighted by the attention weights for all attention heads of
        # dimensionality (num_heads, 1, max_len, d_head)
        weighted_sum = torch.matmul(attention_weights, multi_value)
        # Tuple of weighted sums for each attention head (total length = num_heads)
        # each entry has dimensionality (1, 1, max_len, dim_head)
        weighted_sum = weighted_sum.split(1, dim=0)
        # All weighted sums stacked together for efficiency this gives a torch.tensor
        # with dimensionality (max_len, num_heads * dim_head) which is equal to (max_len, embed_dim)
        weighted_sum = torch.cat(weighted_sum, dim=-1).squeeze()
        # Calculate output of multi-head attention module before layer normalization
        multi_head = torch.matmul(weighted_sum, self.W_output)

        # Sum multi-head attention plus query
        output = multi_head + query
        # Pass through layer normalization
        output = self.layer_norm(output)
        # Return output and attention weights
        return output, attention_weights
