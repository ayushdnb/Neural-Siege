from typing import Tuple
# ^ Type hints:
#   Tuple[torch.Tensor, torch.Tensor] is used for (logits, value) return types.

import math                # Used for sqrt(2) gain in orthogonal initialization.
import torch
import torch.nn as nn
import torch.nn.functional as F
import config               # Global configuration parameters (imported but not actively referenced below).


# ============================================================================
# Attention building blocks
# ============================================================================
# This file defines a "TransformerBrain" policy/value network suitable for
# actor-critic reinforcement learning (e.g., PPO) on *flat* observations.
#
# The core idea is to treat part of the observation (ray features) as a token
# sequence and apply attention mechanisms:
#   - Cross-attention: ray tokens attend to a "rich state token"
#   - Self-attention: ray tokens attend among themselves
#
# The network produces:
#   - Policy logits: (B, act_dim) for a discrete action space
#   - Value:         (B, 1) for critic estimation
#
# Notation:
#   B  = batch size
#   T  = number of ray tokens (num_rays)
#   Fr = ray feature dimension (ray_feat_dim)
#   Dr = rich feature dimension (rich_feat_dim)
#   D  = embedding dimension (embed_dim)
#   A  = number of actions (act_dim)
# ============================================================================


class CrossAttentionBlock(nn.Module):
    """
    A single block of Cross-Attention followed by a Feed-Forward network (FFN).
    Includes residual connections and LayerNorm.

    Cross-attention definition (mathematical view):
      - Queries (Q) come from one sequence (here: ray tokens).
      - Keys (K) and Values (V) come from another sequence (here: rich state token).

    Standard scaled dot-product attention:
        Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V

    For multi-head attention:
      - Q, K, V are projected into multiple "heads"
      - attention is computed in each head independently
      - head outputs are concatenated and projected back to embed_dim

    In this block:
      - The ray token sequence is updated using information from the rich token.
      - The FFN then applies a position-wise non-linear transformation.

    Residual + LayerNorm pattern:
      - After attention:  x = LN(query + attn(query, memory))
      - After FFN:        x = LN(x + ffn(x))
    """

    def __init__(self, embed_dim: int, num_heads: int = 1):
        """
        Args:
            embed_dim:
                Token embedding dimension D. Must match across query, key, and value projections.
            num_heads:
                Number of attention heads.
                - embed_dim must be divisible by num_heads in nn.MultiheadAttention.
                - Default is 1 for simplicity and lower compute cost.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # MultiheadAttention with batch_first=True expects:
        #   query:     (B, Tq, D)
        #   key/value: (B, Tk, D)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # LayerNorm after attention residual.
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward network (FFN):
        # Common transformer pattern is expansion to a larger hidden size, non-linearity, then contraction.
        # Here expansion factor is 2 (D -> 2D -> D).
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # LayerNorm after FFN residual.
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:
                Ray tokens of shape (B, num_rays, embed_dim).
                These are the "queries" (the sequence requesting information).

            key_value:
                Rich token of shape (B, 1, embed_dim).
                These act as both keys and values (the contextual memory).

        Returns:
            Updated ray tokens of shape (B, num_rays, embed_dim).

        Important shape reasoning:
            - query length Tq = num_rays
            - memory length Tk = 1
            The attention weights for each ray token are computed over a single memory token.
            Conceptually, each ray token receives a context-dependent transformation derived
            from the rich token.
        """
        # Cross-attention: ray tokens attend to the rich token.
        # need_weights=False avoids returning attention matrices (saves memory and compute).
        attn_output, _ = self.attn(query, key_value, key_value, need_weights=False)

        # Residual + LayerNorm.
        x = self.norm1(query + attn_output)

        # Feed-forward network.
        ffn_output = self.ffn(x)

        # Second residual + LayerNorm.
        x = self.norm2(x + ffn_output)
        return x


class SelfAttentionBlock(nn.Module):
    """
    A single block of Self-Attention followed by a Feed-Forward network (FFN).
    Includes residual connections and LayerNorm.

    Self-attention definition:
      - Queries, keys, values are all derived from the same sequence x.

    Purpose in this design:
      - After each ray token has incorporated global context via cross-attention,
        self-attention allows ray tokens to exchange information with each other.
      - This supports relational reasoning across ray directions (for example,
        local obstacles and openings in different directions influencing each other).

    Residual + LayerNorm structure:
      x = LN(x + SelfAttn(x))
      x = LN(x + FFN(x))
    """

    def __init__(self, embed_dim: int, num_heads: int = 1):
        """
        Args:
            embed_dim: Token embedding dimension D.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Ray tokens (B, num_rays, embed_dim)

        Returns:
            Self-attended ray tokens (B, num_rays, embed_dim)
        """
        # Self-attention: Q=K=V=x.
        attn_output, _ = self.attn(x, x, x, need_weights=False)

        # Residual + LayerNorm.
        x = self.norm1(x + attn_output)

        # Feed-forward + second residual + LayerNorm.
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


# ============================================================================
# The Main Transformer Brain
# ============================================================================

class TransformerBrain(nn.Module):
    """
    A transformer-based policy/value network that processes observations by treating
    raycasts as a sequence of tokens and enriching them with the agent's state via attention.

    Observation structure (as assumed in this implementation):
        - Ray features: num_rays * ray_feat_dim = 32 * 8 = 256 features
        - Rich features: the remaining (obs_dim - 256) features

    Tokenization strategy:
        - Each ray becomes one token with ray_feat_dim features.
        - The rich features become a single token.

    Processing pipeline:
        1) Embed ray tokens to (B, 32, D) and add a learnable positional encoding.
        2) Embed rich features to (B, 1, D).
        3) Cross-attention: ray tokens attend to rich token.
        4) Self-attention: ray tokens attend among themselves.
        5) Pool ray tokens (mean over ray dimension) to get one vector (B, D).
        6) Concatenate pooled ray summary with rich token (B, 2D).
        7) Apply MLP to produce:
            - logits: (B, act_dim)
            - value:  (B, 1)

    Why mean pooling?
        - It produces a fixed-size representation regardless of sequence length.
        - It is simple and stable.
        - It treats rays symmetrically, which can be desirable when ray order is
          only distinguished by positional encoding.
    """

    def __init__(self, obs_dim: int, act_dim: int, embed_dim: int = 32, mlp_hidden: int = 128):
        """
        Args:
            obs_dim:
                Total observation dimension. Must be > num_rays * ray_feat_dim.
            act_dim:
                Number of discrete actions.
            embed_dim:
                Token embedding dimension D.
            mlp_hidden:
                Hidden width of the final MLP.
        """
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.embed_dim = int(embed_dim)

        # Fixed assumptions about observation structure.
        # This is a hard-coded schema:
        #   - 32 ray directions
        #   - 8 features per ray
        self.num_rays = 32
        self.ray_feat_dim = 8

        # Rich feature dimension is the remaining part of the observation.
        self.rich_feat_dim = self.obs_dim - (self.num_rays * self.ray_feat_dim)

        # Validate the observation is large enough to contain rays and at least 1 rich feature.
        if self.rich_feat_dim <= 0:
            raise ValueError(f"obs_dim ({obs_dim}) is not large enough for {self.num_rays} rays.")

        # ------------------------------------------------------------------
        # Embedding layers
        # ------------------------------------------------------------------
        # Each embedding uses:
        #   LayerNorm (stabilize feature scales)
        #   Linear projection (map raw feature dims -> embed_dim)
        #
        # This is analogous to "token embedding" in NLP, except tokens are numeric feature vectors.

        # Ray features: normalize over last dimension Fr, then project Fr -> D.
        self.ray_embed_norm = nn.LayerNorm(self.ray_feat_dim)
        self.ray_embed_proj = nn.Linear(self.ray_feat_dim, self.embed_dim)

        # Rich features: normalize over Dr, then project Dr -> D.
        self.rich_embed_norm = nn.LayerNorm(self.rich_feat_dim)
        self.rich_embed_proj = nn.Linear(self.rich_feat_dim, self.embed_dim)

        # Learnable positional encoding for ray tokens:
        # - Shape (1, 32, D) broadcast across batch.
        # - Allows the network to distinguish ray index/direction even after mean pooling.
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_rays, self.embed_dim))

        # Attention blocks.
        # Cross-attention uses rich token as context; self-attention models ray-ray relations.
        self.cross_attention = CrossAttentionBlock(self.embed_dim)
        self.self_attention = SelfAttentionBlock(self.embed_dim)

        # ------------------------------------------------------------------
        # Final MLP and heads
        # ------------------------------------------------------------------
        # Input to MLP is concatenation of:
        #   - pooled ray summary: (B, D)
        #   - rich token:         (B, D)
        # yielding (B, 2D).
        mlp_input_dim = self.embed_dim * 2
        self.fc_in = nn.Linear(mlp_input_dim, mlp_hidden)
        self.fc1 = nn.Linear(mlp_hidden, mlp_hidden)

        # Actor head: logits for discrete actions.
        self.actor = nn.Linear(mlp_hidden, self.act_dim)

        # Critic head: scalar value.
        self.critic = nn.Linear(mlp_hidden, 1)

        # Initialize weights with a PPO-friendly scheme.
        self.init_weights()

    def init_weights(self):
        """
        Orthogonal initialization (commonly used in PPO-style actor-critic models).

        What is orthogonal initialization?
            - For a matrix W, "orthogonal" means its columns (or rows) are orthonormal.
            - For square matrices: W^T W = I.
            - For rectangular matrices, PyTorch's orthogonal_ constructs a matrix where
              the smaller dimension forms an orthonormal basis.

        Why orthogonal initialization is used:
            - It helps preserve the variance of activations across layers.
            - It can improve optimization stability in deep networks.
            - It often performs well in reinforcement learning, where training signals
              (advantages/returns) are noisy.

        What is the "gain"?
            - The gain scales the initialized weights.
            - A common heuristic for layers with ReLU-like activations is sqrt(2),
              which helps keep forward/backward magnitudes stable.
            - GELU is often treated similarly in practice.

        Head-specific gains:
            - Actor head gain = 0.01:
                Makes initial logits small in magnitude.
                As a consequence, softmax(logits) is closer to uniform distribution:
                    softmax(0 vector) = uniform distribution
                This yields high-entropy initial policies, encouraging exploration.
            - Critic head gain = 1.0:
                Standard scale for value estimation.

        Bias initialization:
            - All biases are set to 0.
        """
        gain_hidden = math.sqrt(2.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain_hidden)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Override actor head with small gain.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        if self.actor.bias is not None:
            nn.init.zeros_(self.actor.bias)

        # Override critic head with gain 1.0.
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        if self.critic.bias is not None:
            nn.init.zeros_(self.critic.bias)

    # ------------------------------------------------------------------
    # Embedding helpers (kept separate for TorchScript friendliness)
    # ------------------------------------------------------------------

    def _embed_rays(self, rays_raw: torch.Tensor) -> torch.Tensor:
        """
        Embed raw ray features into token embeddings.

        Args:
            rays_raw:
                (B, num_rays, ray_feat_dim) = (B, 32, 8)

        Returns:
            (B, num_rays, embed_dim) = (B, 32, D)

        Steps:
            1) Apply LayerNorm in fp32 for numerical stability.
            2) Cast to the projection layer's weight dtype.
            3) Apply linear projection to embed_dim.

        Practical mixed-precision note:
            - LayerNorm in float32 reduces risk of instability with float16.
        """
        x_norm = self.ray_embed_norm(rays_raw.float())
        return self.ray_embed_proj(x_norm.to(dtype=self.ray_embed_proj.weight.dtype))

    def _embed_rich(self, rich_raw: torch.Tensor) -> torch.Tensor:
        """
        Embed raw rich features into a single token embedding.

        Args:
            rich_raw:
                (B, rich_feat_dim)

        Returns:
            (B, embed_dim)

        Same numerical pattern as _embed_rays.
        """
        x_norm = self.rich_embed_norm(rich_raw.float())
        return self.rich_embed_proj(x_norm.to(dtype=self.rich_embed_proj.weight.dtype))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the transformer brain.

        Args:
            obs:
                Observation tensor of shape (B, obs_dim).

        Returns:
            logits:
                (B, act_dim) unnormalized action scores for a categorical policy.
            value:
                (B, 1) scalar state value estimate.

        Conceptual flow:
            - Tokenize observation into ray tokens + one rich token.
            - Apply attention blocks to produce contextualized ray tokens.
            - Pool and combine into a single representation.
            - Map to policy logits and value through an MLP.
        """
        B = obs.shape[0]

        # ------------------------------------------------------------------
        # Split observation into ray portion and rich portion
        # ------------------------------------------------------------------
        # Rays are the first 32 * 8 = 256 values of the observation.
        # They are reshaped into (B, 32, 8) to become a token sequence.
        rays_raw = obs[:, :self.num_rays * self.ray_feat_dim].view(B, self.num_rays, self.ray_feat_dim)

        # Rich features are the remaining values.
        rich_raw = obs[:, self.num_rays * self.ray_feat_dim:]

        # ------------------------------------------------------------------
        # Embed raw features into token space
        # ------------------------------------------------------------------
        ray_tokens = self._embed_rays(rays_raw)                 # (B, 32, D)
        rich_token = self._embed_rich(rich_raw).unsqueeze(1)    # (B, 1, D)

        # Add positional encoding so each ray index/direction is distinguishable.
        # Broadcast: (1, 32, D) + (B, 32, D) -> (B, 32, D)
        ray_tokens = ray_tokens + self.positional_encoding

        # ------------------------------------------------------------------
        # Attention processing
        # ------------------------------------------------------------------
        # Step 1: Cross-attention
        # Each ray token queries the rich token to incorporate global state context.
        contextual_ray_tokens = self.cross_attention(query=ray_tokens, key_value=rich_token)

        # Step 2: Self-attention among rays
        # Rays exchange information with each other, enabling relational reasoning.
        processed_ray_tokens = self.self_attention(contextual_ray_tokens)

        # ------------------------------------------------------------------
        # Pooling and final MLP
        # ------------------------------------------------------------------
        # Mean pooling across rays yields a single vector summary:
        #   processed_ray_tokens: (B, 32, D)
        #   pooled:              (B, D)
        pooled_ray_summary = processed_ray_tokens.mean(dim=1)

        # Concatenate pooled rays with rich token (remove token dimension first).
        # rich_token.squeeze(1): (B, D)
        # mlp_input:             (B, 2D)
        mlp_input = torch.cat([pooled_ray_summary, rich_token.squeeze(1)], dim=-1)

        # MLP with GELU activations.
        h = F.gelu(self.fc_in(mlp_input))
        h = F.gelu(self.fc1(h))

        # Actor and critic outputs.
        logits = self.actor(h)    # (B, act_dim)
        value = self.critic(h)    # (B, 1)

        return logits, value

    def param_count(self) -> int:
        """
        Count the number of trainable parameters.

        This is useful for:
          - verifying model size constraints
          - comparing architectures
          - estimating memory / compute requirements
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def scripted_transformer_brain(obs_dim: int, act_dim: int) -> torch.jit.ScriptModule:
    """
    Create a TorchScript version of TransformerBrain.

    TorchScript motivation:
      - Reduces Python overhead in tight loops.
      - Enables deployment in environments without Python interpretation.
      - Can improve portability of inference pipelines.

    Returns:
      A torch.jit.ScriptModule produced via torch.jit.script(model).

    Note:
      - torch.jit.script traces and compiles the Python module into a TorchScript graph
        with control-flow support.
      - This requires TorchScript-compatible code patterns in the model.
    """
    model = TransformerBrain(obs_dim, act_dim)
    return torch.jit.script(model)