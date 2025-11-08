import torch
import torch.nn as nn
import torch.nn.functional as F


# Extract the [CLS] token representation
class CLSPooling(nn.Module):
    # Dummy initializer
    def __init__(self):
        super(CLSPooling, self).__init__()

    def forward(self, hidden_state, attention_mask=None):
        return hidden_state[:, 0, :]


# Extract the hidden representation of the last non-padded token
class LastTokenPooling(nn.Module):
    # Dummy initializer
    def __init__(self):
        super(LastTokenPooling, self).__init__()

    def forward(self, hidden_state, attention_mask=None):
        if attention_mask is None:
            return hidden_state[:, -1, :]

        # Select the last non-padded token per sequence
        seq_len_idx = attention_mask.sum(dim=1) - 1
        last_hidden = hidden_state[torch.arange(hidden_state.size(0)), seq_len_idx, :]

        return last_hidden


# Computes the mean-pooled embedding of hidden state
class MeanPooling(nn.Module):
    # Dummy initializer
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, hidden_state, attention_mask=None):
        """
        hidden_state : [B, L, H]
        attention_mask : [B, L]
        """

        if attention_mask is None:
            return hidden_state.mean(dim=1)

        # Expand mask for broadcasting
        mask = attention_mask.unsqueeze(-1).to(
            dtype=hidden_state.dtype, device=hidden_state.device
        )

        # Apply mask and sum token embeddings
        masked_sum = (mask * hidden_state).sum(dim=1)

        # Sum valid tokens and clamp to avoid division by zero
        valid_token_count = mask.sum(dim=1).clamp_min(torch.finfo(mask.dtype).tiny)

        # Divide summed vectors by valid token count
        mean_pooled = masked_sum / valid_token_count

        return mean_pooled


# Apply attention-based pooling over sequence representations
class AttentionPooling(nn.Module):
    # Initializer
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()

        self.attn_score_fc = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_state, attention_mask=None):
        """
        hidden_state : [B, L, H]
        attention_mask : [B, L]
        """

        # Compute attention scores for each token
        attn_logits = self.attn_score_fc(hidden_state)
        attn_logits = attn_logits.squeeze(-1).to(
            dtype=hidden_state.dtype, device=hidden_state.device
        )

        # Apply attention mask to ignore padding tokens
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            neg_inf = torch.finfo(attn_logits.dtype).min
            masked_attn_logits = attn_logits.masked_fill(~attention_mask, neg_inf)

        # Code for numerical stability before softmax
        masked_attn_logits = (
            masked_attn_logits - masked_attn_logits.max(dim=1, keepdim=True).values
        )

        # Apply softmax to obtain normalized attention weights
        weights = F.softmax(masked_attn_logits, dim=1)

        # Compute attention-weighted sum of hidden states
        attn_pooled = torch.einsum("bl,blh->bh", weights, hidden_state)

        return attn_pooled
