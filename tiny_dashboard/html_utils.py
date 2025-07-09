from pathlib import Path
import warnings
from .utils import (
    sanitize_html_content,
    update_template_string,
    sanitize_token,
    sanitize_tokens,
)
import torch as th

template_dir = Path(__file__).parent / "templates"
with open(template_dir / "styles.css", "r") as f:
    styles = f.read()
with open(template_dir / "listeners.js", "r") as f:
    scripts = f.read()
with open(template_dir / "base.html", "r") as f:
    base_template = f.read()
with open(template_dir / "feature_section.html", "r") as f:
    feature_template = f.read()
with open(template_dir / "example_container.html", "r") as f:
    example_template = f.read()
with open(template_dir / "click_to_expand.html", "r") as f:
    click_to_expand_template = f.read()
with open(template_dir / "end_of_collapsed.html", "r") as f:
    end_of_collapsed_template = f.read()
with open(template_dir / "token.html", "r") as f:
    token_template = f.read()


def create_example_html(
    max_act: float | str, collapsed_html: str, full_html: str = "", static=False
) -> str:
    """Create HTML for a single example using the example template."""
    if isinstance(max_act, float):
        max_act = f"{max_act:.2f}"
    max_act = str(max_act)
    if static or full_html == collapsed_html:
        click_to_expand = ""
        end_of_collapsed = ""
        is_static = "true"
        full_html = collapsed_html
    else:
        click_to_expand = click_to_expand_template
        end_of_collapsed = end_of_collapsed_template
        is_static = "false"
    return update_template_string(
        example_template,
        {
            "max_act": max_act,
            "collapsed_html": collapsed_html + end_of_collapsed,
            "full_html": full_html,
            "click_to_expand_html": click_to_expand,
            "static": is_static,
        },
    )


def create_base_html(
    title: str, content: str | list[str], styles: str = styles, scripts: str = scripts
) -> str:
    """Create the base HTML with title, content, styles and scripts."""
    if isinstance(content, (list, tuple)):
        content = "\n".join(content)
    return update_template_string(
        base_template,
        {"title": title, "content": content, "styles": styles, "scripts": scripts},
    )


def create_feature_section_html(
    feature_idx: int, max_act: float, full_html: str
) -> str:
    """Create HTML for a feature section using the feature template."""
    return update_template_string(
        feature_template,
        {
            "feature_idx": str(feature_idx),
            "max_activation": f"{max_act:.2f}",
            "full_html": full_html,
        },
    )


def create_token_html(
    token: str, tokenstr: str, color: str | tuple[str, str], tooltip_content: str
) -> str:
    """Create HTML for a single token span with tooltip.

    Args:
        token: The token text to display
        color: Either a single color string for single feature,
               or a tuple of (top_color, bottom_color) for dual feature visualization
        tooltip_content: Text to show in tooltip
    """
    if isinstance(color, tuple):
        top_color, bottom_color = color
    else:
        # Single feature case - use same color for top and bottom
        top_color = bottom_color = color
    if "<br>" in tokenstr:
        raise ValueError("Token contains <br> tag")
    return update_template_string(
        token_template,
        {
            "token": token.replace(" ", "&nbsp;"),
            "top_color": top_color,
            "bottom_color": bottom_color,
            "tooltip_content": tooltip_content,
            "token_str": tokenstr.replace("\\n", "&#10;"),
        },
    )


def create_highlighted_tokens_html(
    tokens: list[str],
    activations: th.Tensor,
    tokenizer,
    *,
    # Feature selection
    highlight_features: (
        list[int] | int | None
    ) = None,  # Indices into activations tensor
    tooltip_features: list[int] | int | None = None,  # None = show all features
    # Visualization options
    min_max_act: float | None = None,
    min_max_act_negative: float | None = None,
    color1: tuple[int, int, int] = (255, 0, 0),
    color1_negative: tuple[int, int, int] = (0, 0, 255),
    color2: tuple[int, int, int] | None = None,
    color2_negative: tuple[int, int, int] | None = None,
    relative_normalization: bool = True,  # False = normalize against global max
    separate_positive_negative_normalization: bool = False,  # If True, normalize positive and negative values separately
    activation_names: list[str] | None = None,
    return_max_acts_str: bool = False,
    highlight_features_in_tooltip: bool = True,
) -> str | tuple[str, str]:
    """Create HTML with highlighted tokens based on activation values.

    Args:
        tokens: List of tokens to display
        activations: Tensor of shape [seq_len, num_features] or [seq_len] of floats.
            Activations can be 'nan' for tokens that should not be highlighted.
        tokenizer: Tokenizer for getting token IDs
        highlight_features: Which features to highlight (max 2)
        tooltip_features: Which features to show in tooltip (None = all)
        min_max_act: Maximum absolute value for normalization
        min_max_act_negative: Maximum absolute value for negative normalization (if None or separate_positive_negative_normalization is False, min_max_act is used)
        color1: RGB color tuple for primary feature
        color1_negative: RGB color tuple for negative primary feature
        color2: RGB color tuple for secondary feature (None = same as color1)
        color2_negative: RGB color tuple for negative secondary feature
        separate_positive_negative_normalization: If True, normalize positive and negative values separately
        relative_normalization: If True, normalize each feature independently
        activation_names: List of names for each feature (optional)
        return_max_acts_str: If True, return a string with the max activation values
        highlight_features_in_tooltip: If True, ensure that the highlighted features are in the tooltip

    Returns:
        HTML string containing the highlighted tokens
        If return_max_acts_str is True, return a tuple with the HTML string and the max activation values string
    """
    if activations.dim() == 1:
        activations = activations.unsqueeze(1)
    elif activations.dim() != 2:
        raise ValueError("Activations must be 1D or 2D")
    if highlight_features is None:
        highlight_features = list(range(activations.shape[1]))
    if isinstance(highlight_features, int):
        highlight_features = [highlight_features]
    if len(highlight_features) > 2:
        raise ValueError("Can only highlight up to 2 features")
    if len(highlight_features) == 2 and (
        activations.dim() != 2 or activations.shape[1] < 2
    ):
        raise ValueError(
            "Cannot highlight 2 features if activations are not 2D or have less than 2 features"
        )
    if activation_names is None:
        activation_names = [f"Feature {i}" for i in list(range(activations.shape[1]))]
    if color2 is None:
        color2 = color1
    if color2_negative is None:
        color2_negative = color1_negative
    if isinstance(tooltip_features, int):
        tooltip_features = [tooltip_features]
    if tooltip_features is None:
        tooltip_features = list(range(activations.shape[1]))
    if highlight_features_in_tooltip:
        tooltip_features = list(dict.fromkeys(highlight_features + tooltip_features))

    # Get activation values for highlighted features
    highlight_acts = [activations[:, idx] for idx in highlight_features]
    # Handle normalization
    if min_max_act is None:
        if relative_normalization:
            if separate_positive_negative_normalization:
                # Separate normalization for positive and negative values
                max_vals_pos = [acts[~acts.isnan() & (acts > 0)].max() if (acts[~acts.isnan() & (acts > 0)]).numel() > 0 else th.tensor(1.0) for acts in highlight_acts]
                max_vals_neg = [acts[~acts.isnan() & (acts < 0)].abs().max() if (acts[~acts.isnan() & (acts < 0)]).numel() > 0 else th.tensor(1.0) for acts in highlight_acts]
            else:
                # Standard normalization using absolute maximum
                max_vals = [acts[~acts.isnan()].abs().max() for acts in highlight_acts]
        else:
            if separate_positive_negative_normalization:
                # Global positive and negative normalization
                all_acts = th.cat([acts[~acts.isnan()] for acts in highlight_acts])
                max_val_pos = all_acts[all_acts > 0].max() if (all_acts > 0).any() else th.tensor(1.0)
                max_val_neg = all_acts[all_acts < 0].abs().max() if (all_acts < 0).any() else th.tensor(1.0)
                max_vals_pos = [max_val_pos] * len(highlight_acts)
                max_vals_neg = [max_val_neg] * len(highlight_acts)
            else:
                # Standard global normalization
                max_val = max(acts[~acts.isnan()].abs().max() for acts in highlight_acts)
                max_vals = [max_val] * len(highlight_acts)
    else:
        if separate_positive_negative_normalization:
            if min_max_act_negative is not None:
                max_vals_pos = [min_max_act] * len(highlight_acts)
                max_vals_neg = [min_max_act_negative] * len(highlight_acts)
            else:
                max_vals_pos = [min_max_act] * len(highlight_acts)
                max_vals_neg = [min_max_act] * len(highlight_acts)
        else:
            max_vals = [min_max_act] * len(highlight_acts)

    # Normalize activations
    if separate_positive_negative_normalization:
        norm_acts = []
        for acts, max_pos, max_neg in zip(highlight_acts, max_vals_pos, max_vals_neg):
            norm_act = th.zeros_like(acts)
            pos_mask = acts > 0
            neg_mask = acts < 0
            norm_act[pos_mask] = acts[pos_mask] / (max_pos + 1e-6)
            norm_act[neg_mask] = acts[neg_mask] / (max_neg + 1e-6)
            norm_act[acts.isnan()] = acts[acts.isnan()]  # Preserve NaN values
            norm_acts.append(norm_act)
    else:
        norm_acts = [
            acts / (max_val + 1e-6) for acts, max_val in zip(highlight_acts, max_vals)
        ]

    # Generate HTML for each token
    html_parts = []
    sanitized_tokens = sanitize_tokens(tokens, non_breaking_space=False)

    for i, (san_token, token) in enumerate(zip(sanitized_tokens, tokens)):
        # Generate colors for token
        token_colors = []
        for norm_act, positive_color, negative_color in zip(
            norm_acts, [color1, color2], [color1_negative, color2_negative]
        ):
            act_val = norm_act[i].item() if not (norm_act[i].isnan()) else 0
            if abs(act_val) > 1:
                if min_max_act is None:
                    raise RuntimeError(
                        "Got an activation > max normalization value but no min_max_act was provided"
                    )
                warnings.warn(
                    f"Warning:\nGot an activation > max normalization value ({act_val}), clamping to ±1"
                )
                act_val = 1 if act_val > 0 else -1

            # Choose color based on sign of activation
            if act_val >= 0:
                r, g, b = positive_color
                intensity = act_val
            else:
                r, g, b = negative_color
                intensity = abs(act_val)

            token_colors.append(f"rgba({r}, {g}, {b}, {intensity:.3f})")

        # If only one feature, duplicate the color
        if len(token_colors) == 1:
            token_colors = (token_colors[0], token_colors[0])

        # Generate tooltip content
        tooltip_lines = [
            f"Token {tokenizer.convert_tokens_to_ids(token)}: "
            f"'{sanitize_token(token, keep_newline=False, non_breaking_space=False)}'"
        ]

        for feat_idx in tooltip_features:
            act_value = activations[i, feat_idx]
            if not th.isnan(act_value):
                act_value = act_value.item()
                tooltip_lines.append(f"{activation_names[feat_idx]}: {act_value:.3f}")

        html_parts.append(
            create_token_html(
                san_token,
                sanitize_token(token, keep_newline=False, non_breaking_space=False),
                tuple(token_colors),
                "\n".join(tooltip_lines),
            )
        )

    html = "".join(html_parts)
    if return_max_acts_str:
        max_acts_str = "<br>".join(
            f"{activation_names[idx]}: {mv.item():.3f}"
            for idx, mv in enumerate(
                [acts[~acts.isnan()].max() for acts in highlight_acts]
            )
        )
        tooltip_max_acts = [
            activations[:, feat][~activations[:, feat].isnan()].max().item()
            for feat in tooltip_features
        ]
        for feat, max_val in zip(tooltip_features, tooltip_max_acts):
            if feat in highlight_features:
                continue
            max_acts_str += f"<br>{activation_names[feat]}: {max_val:.3f}"
        return html, "<br>" + max_acts_str
    return html
