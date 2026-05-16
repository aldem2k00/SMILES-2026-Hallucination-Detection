"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations
import math

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from model import MAX_LENGTH


device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_name = 'Qwen/Qwen2.5-0.5B'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True,
    torch_dtype=torch.bfloat16,
)
model = model.eval()

lm_head = nn.Linear(896, 151936, bias=False)
lm_head.load_state_dict(model.lm_head.state_dict())
lm_head = lm_head.to(device)

assistant_embedding = model.get_input_embeddings()(torch.tensor([77091])).to(device)
im_start_embedding = model.get_input_embeddings()(torch.tensor([151644])).to(device)
endoftext_embedding = model.get_input_embeddings()(torch.tensor([151643])).to(device)

del model

def detect_response_start(hidden_in):
    """
    Find the position in the sequence where the generation starts.
    """
    zero_tensor = torch.tensor([0.0]).to(device)
    a_dist = torch.norm(hidden_in - assistant_embedding, dim=1)
    a_indices = set(torch.where(a_dist.isclose(zero_tensor))[0].tolist())
    i_dist = torch.norm(hidden_in - im_start_embedding, dim=1)
    i_indices = set((torch.where(i_dist.isclose(zero_tensor))[0] + 1).tolist())
    singleton_set = a_indices.intersection(i_indices)
    if len(singleton_set) < 1:
        return None
    index = singleton_set.pop() + 2
    if index >= MAX_LENGTH:
        return None
    return index

def embedding_is_eos(e):
    zero_tensor = torch.tensor([0.0]).to(device)
    dist = torch.norm(e - endoftext_embedding)
    ret = torch.isclose(dist, zero_tensor).item()
    return ret

def get_output_entropy(logits):
    probs = torch.softmax(logits, dim=1)
    logprobs = torch.log(probs)
    entropy = -torch.sum(probs * logprobs, dim=1)
    return entropy

def get_entropies(hidden, start_index, end_index):
    with torch.no_grad():
        logits = lm_head(hidden[-1][start_index:end_index])
    entropies = get_output_entropy(logits)
    return entropies

def extract_llm_check_features(hidden_states):
    # hidden_states shape: (n_layers + 1, seq_len, hidden_dim)
    n_layers = hidden_states.shape[0] - 1
    seq_len = hidden_states.shape[1]
    features = []
    # Skip index 0 (input embeddings), iterate through actual layers
    for l in range(1, n_layers + 1):
        H_l = hidden_states[l]
        singular_values = torch.linalg.svdvals(H_l)
        singular_values = singular_values[singular_values > 1e-7]
        hidden_score = (2.0 / seq_len) * torch.sum(torch.log(singular_values))
        features.append(hidden_score.item())
    return torch.tensor(features)

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    # Default: last real token of the final transformer layer.
    layer = hidden_states[-1]          # (seq_len, hidden_dim)

    # Find the index of the last real (non-padding) token.
    real_positions = attention_mask.nonzero(as_tuple=False)  # (n_real, 1)
    last_pos = int(real_positions[-1].item())                 # scalar index

    feature_0 = layer[last_pos].cpu()          # (hidden_dim,)

    end_index = attention_mask.sum()
    start_index = detect_response_start(hidden_states[0,:end_index])

    if isinstance(start_index, int):
        entropies = get_entropies(hidden_states, start_index, end_index)
        feature_1 = torch.cat((
            torch.amin(entropies, keepdim=True),
            torch.quantile(
                entropies,
                torch.tensor(
                    [0.25, 0.5, 0.75],
                    dtype=entropies.dtype,
                    device=entropies.device
                )
            ),
            torch.amax(entropies, keepdim=True),
            torch.mean(entropies, dim=0, keepdim=True),
            torch.std(entropies, dim=0, unbiased=False, keepdim=True),
            torch.tensor(
                [math.log(len(entropies))],
                dtype=entropies.dtype,
                device=entropies.device
            ),
            entropies[:1]
        ))
        feature_2 = torch.tensor([0.0,])
    else:
        feature_1 = torch.zeros(size=(9,))
        feature_2 = torch.tensor([1.0,])

    if end_index < MAX_LENGTH or embedding_is_eos(hidden_states[0,-1]):
        feature_3 = torch.tensor([0.0,])
    else:
        feature_3 = torch.tensor([1.0,])

    feature_4 = extract_llm_check_features(hidden_states[:,:end_index])

    all_features = (
        feature_0.to(device),
        feature_1.to(device),
        feature_2.to(device),
        feature_3.to(device),
        feature_4.to(device)
    )

    return torch.cat(all_features)
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
