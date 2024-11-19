from pathlib import Path
from .utils import sanitize_html_content, update_template_string, sanitize_token

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


def create_token_html(token: str, color: str, tooltip_content: str) -> str:
    """Create HTML for a single token span with tooltip."""
    tooltip_content = sanitize_html_content(tooltip_content)
    return update_template_string(
        token_template,
        {
            "token": token.replace(" ", "&nbsp;"),
            "color": color,
            "tooltip_content": tooltip_content,
        },
    )
