from pathlib import Path
import torch as th
from .html_utils import (
    create_token_html,
    create_example_html,
    create_base_html,
    create_highlighted_tokens_html,
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
    color2: tuple[int, int, int] = None,
    activation_names: list[str] = None,
    relative_normalization: bool = True,
    tooltip_features: Optional[Union[int, List[int]]] = None,
    highlight_features_in_tooltip: bool = True,
) -> str:
    """Create HTML with highlighted tokens based on activation values.

    Args:
        tokens: List of tokens to display
        activations: Tensor of activation values. Shape: [seq_len, num_features] or [seq_len]
        tokenizer: Tokenizer for getting token IDs
        highlight_idx: Which feature(s) to highlight. If None:
                      - For 1D tensor: use as is
                      - For 2D tensor: use features [0] or [0,1]
        title: Title for the visualization
        color1: RGB tuple for primary color (default red)
        color2: RGB tuple for secondary color (default None = same as color1)
        activation_names: List of names for each feature (optional)
        relative_normalization: If True, normalize each feature independently
        tooltip_features: Which features to show in tooltip (None = all)
        highlight_features_in_tooltip: If True, ensure that the highlighted features are in the tooltip
    Returns:
        HTML string containing the visualization
    """

    # Create HTML with highlighted tokens
    html, max_acts_str = create_highlighted_tokens_html(
        tokens=tokens,
        activations=activations,
        tokenizer=tokenizer,
        highlight_features=highlight_idx,
        color1=color1,
        color2=color2,
        relative_normalization=relative_normalization,
        activation_names=activation_names,
        tooltip_features=tooltip_features,
        highlight_features_in_tooltip=highlight_features_in_tooltip,
        return_max_acts_str=True,
    )
    html = create_example_html(max_acts_str, html, static=True)
    return create_base_html(title, html)
