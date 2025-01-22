from pathlib import Path
import torch as th
from .html_utils import (
    create_token_html,
    create_example_html,
    create_base_html,
)
from .utils import sanitize_tokens, sanitize_token
from typing import Optional, Union, List


def activation_visualization(
    tokens: list[str],
    activations: th.Tensor,
    tokenizer,
    highlight_idx: Optional[Union[int, List[int]]] = None,
    title: str = "",
    color1: tuple[int, int, int] = (255, 0, 0),
    color2: tuple[int, int, int] = (0, 0, 255),
) -> str:
    """Create HTML with highlighted tokens based on activation values.

    Args:
        tokens: List of tokens to display
        activations: Tensor of activation values. Shape: [num_features, seq_len] or [seq_len]
        tokenizer: Tokenizer for getting token IDs
        highlight_idx: Which feature(s) to highlight. If None:
                      - For 1D tensor: use as is
                      - For 2D tensor: use features [0] or [0,1]
        title: Title for the visualization
        color1: RGB tuple for primary color (default red)
        color2: RGB tuple for secondary color (default blue)

    Returns:
        HTML string containing the visualization
    """
    html_parts = []

    # Handle different activation cases
    if activations.dim() == 1:
        if highlight_idx is not None:
            raise ValueError("highlight_idx should be None for 1D activation tensor")
        highlight_acts = activations
        activations = activations.unsqueeze(0)
        other_features = []
        secondary_acts = None
    else:  # 2D case
        if highlight_idx is None:
            # Default to first feature or first two features
            highlight_idx = [0] if activations.shape[0] == 1 else [0, 1]

        if isinstance(highlight_idx, int):
            highlight_idx = [highlight_idx]

        if len(highlight_idx) == 1:
            # Single feature case
            highlight_acts = activations[highlight_idx[0]]
            other_features = [
                i for i in range(activations.shape[0]) if i != highlight_idx[0]
            ]
            secondary_acts = None
        elif len(highlight_idx) == 2:
            # Two feature case
            highlight_acts = activations[highlight_idx[0]]
            secondary_acts = activations[highlight_idx[1]]
            other_features = [
                i for i in range(activations.shape[0]) if i not in highlight_idx
            ]
        else:
            raise ValueError("highlight_idx must contain 1 or 2 indices")

    # Find overall max for normalization
    max_primary = highlight_acts[highlight_acts.isnan() == False].max()
    if secondary_acts is not None:
        max_secondary = secondary_acts[secondary_acts.isnan() == False].max()
    else:
        max_secondary = max_primary

    # Normalize activations for color intensity
    norm_acts = highlight_acts / (max_primary + 1e-6)
    if secondary_acts is not None:
        norm_secondary = secondary_acts / (max_secondary + 1e-6)

    # Create HTML spans with activation values
    sanitized_tokens = sanitize_tokens(tokens, non_breaking_space=False)
    for i, (san_token, token) in enumerate(zip(sanitized_tokens, tokens)):
        # Calculate primary color
        intensity_primary = norm_acts[i].item() if not norm_acts[i].isnan() else 0
        primary_color = (
            f"rgba({color1[0]}, {color1[1]}, {color1[2]}, {intensity_primary:.3f})"
        )

        # Calculate secondary color
        if secondary_acts is not None:
            intensity_secondary = (
                norm_secondary[i].item() if not norm_secondary[i].isnan() else 0
            )
            secondary_color = f"rgba({color2[0]}, {color2[1]}, {color2[2]}, {intensity_secondary:.3f})"
        else:
            secondary_color = primary_color

        # Create tooltip content
        tok_id = tokenizer.convert_tokens_to_ids(token)
        tooltip_token = sanitize_token(
            token, keep_newline=False, non_breaking_space=False
        )
        tooltip_lines = [f"Token {tok_id}: '{tooltip_token}'"]

        # Add activations to tooltip
        if secondary_acts is not None:
            tooltip_lines.extend(
                [
                    f"Feature {highlight_idx[0]}: {highlight_acts[i].item():.3f}",
                    f"Feature {highlight_idx[1]}: {secondary_acts[i].item():.3f}",
                ]
            )
        elif activations.shape[0] > 1:  # 2D tensor with single highlight
            tooltip_lines.append(
                f"Feature {highlight_idx[0]}: {highlight_acts[i].item():.3f}"
            )
        else:  # 1D tensor
            tooltip_lines.append(f"Activation: {highlight_acts[i].item():.3f}")

        # Add other feature activations to tooltip
        for feat in other_features:
            act_value = activations[feat, i].item()
            tooltip_lines.append(f"Feature {feat}: {act_value:.3f}")

        tooltip_content = "\n".join(tooltip_lines)
        html_parts.append(
            create_token_html(
                san_token, (primary_color, secondary_color), tooltip_content
            )
        )

    html = "".join(html_parts)
    max_val = max(max_primary.item(), max_secondary.item())
    html = create_example_html(max_val, html, static=True)
    return create_base_html(title, html)
