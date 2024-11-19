import time
import traceback
import warnings
from pathlib import Path
import ipywidgets as widgets
import numpy as np
import torch as th
from IPython.display import HTML, display
from abc import ABC, abstractmethod
from typing import Callable

from transformers import AutoTokenizer
from nnsight import LanguageModel
from .utils import sanitize_tokens, apply_chat, parse_list_str, DummyModel
from .html_utils import (
    create_example_html,
    create_base_html,
    create_token_html,
)


class OfflineFeatureCentricDashboard:
    """
    This Dashboard is composed of a feature selector and a feature viewer.
    The feature selector allows you to select a feature and view the max activating examples for that feature.
    The feature viewer displays the max activating examples for the selected feature. Text is highlighted with a gradient based on the activation value of each token.
    An hover text showing the activation value, token id is also shown. When the mouse passes over a token, the token is highlighted in light grey.
    By default, the text sample is not displayed entirely, but only a few tokens before and after the highest activating token. If the user clicks on the text sample, the entire text sample is displayed.
    """

    def __init__(
        self,
        max_activation_examples: dict[int, list[tuple[float, list[str], list[float]]]],
        tokenizer,
        window_size: int = 50,
        max_examples: int = 30,
    ):
        """
        Args:
            max_activation_examples: Dictionary mapping feature indices to lists of tuples
                (max_activation_value, list of tokens, list of activation values)
            tokenizer: HuggingFace tokenizer for the model
            window_size: Number of tokens to show before/after the max activation token
        """
        self.max_activation_examples = max_activation_examples
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_examples = max_examples
        # Load templates at initialization

        self._setup_widgets()

    def _setup_widgets(self):
        """Initialize the dashboard widgets"""

        self.available_features = sorted(self.max_activation_examples.keys())
        self.feature_selector = widgets.Text(
            placeholder="Type a feature number...",
            description="Feature:",
            continuous_update=False,  # Only trigger on Enter/loss of focus
            style={"description_width": "initial"},
        )

        self.examples_output = widgets.Output()
        self.feature_selector.observe(self._handle_feature_selection, names="value")

    def _handle_feature_selection(self, change):
        """Handle feature selection, including validation of typed input"""
        try:
            feature_idx = int(change["new"])
            if feature_idx in self.max_activation_examples:
                self._update_examples({"new": feature_idx})
            else:
                with self.examples_output:
                    self.examples_output.clear_output()
                    print(
                        f"Feature {feature_idx} not found. Available features: {self.available_features}"
                    )
        except ValueError:
            with self.examples_output:
                self.examples_output.clear_output()
                print("Please enter a valid feature number")

    def _create_html_highlight(
        self,
        tokens: list[str],
        activations: list[float],
        max_idx: int,
        show_full: bool = False,
    ) -> str:
        html_parts = []
        # Determine window bounds
        if show_full:
            start_idx = 0
            end_idx = len(tokens)
        else:
            start_idx = max(0, max_idx - self.window_size)
            end_idx = min(len(tokens), max_idx + self.window_size + 1)

        # Normalize activations for color intensity
        act_array = np.array(activations)
        max_act = np.max(np.abs(act_array))
        norm_acts = act_array / max_act if max_act > 0 else act_array

        # Create HTML spans with activation values
        tokens = sanitize_tokens(tokens)
        for i in range(start_idx, end_idx):
            act = activations[i]
            norm_act = norm_acts[i]
            token = tokens[i]
            color = f"rgba(255, 0, 0, {abs(norm_act):.3f})"
            tok_id = self.tokenizer.convert_tokens_to_ids(tokens[i])
            tooltip_content = f"Token {tok_id}: '{token}'\nActivation: {act:.3f}"
            html_parts.append(create_token_html(token, color, tooltip_content))

        return "".join(html_parts)

    def generate_html(self, feature_idx: int) -> str:
        examples = self.max_activation_examples[feature_idx]

        content_parts = []
        for max_act, tokens, token_acts in list(examples)[: self.max_examples]:
            max_idx = np.argmax(token_acts)

            # Create both versions
            collapsed_html = self._create_html_highlight(
                tokens, token_acts, max_idx, False
            )
            full_html = self._create_html_highlight(tokens, token_acts, max_idx, True)

            content_parts.append(
                create_example_html(max_act, collapsed_html, full_html)
            )

        # Display the HTML content all at once
        html_content = create_base_html(
            title=f"Feature {feature_idx} Examples",
            content=content_parts,
        )
        return html_content

    def _update_examples(self, change):
        """Update the examples display when a new feature is selected"""
        # Clear the output first
        self.examples_output.clear_output(
            wait=True
        )  # wait=True for smoother transition

        feature_idx = change["new"]
        with self.examples_output:
            display(HTML(self.generate_html(feature_idx)))

    def display(self):
        """Display the dashboard"""

        dashboard = widgets.VBox([self.feature_selector, self.examples_output])
        display(dashboard)

    def export_to_html(self, output_path: str, feature_to_export: int):
        """
        Export the dashboard data to a static HTML file.
        Creates a single self-contained HTML file with embedded CSS and JavaScript.
        """
        html_content = self.generate_html(feature_to_export)

        # Create output directory and write file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)


class AbstractOnlineFeatureCentricDashboard(ABC):
    """
    Abstract base class for real-time feature analysis dashboards.
    Users can input text, select a feature, and see the activation patterns
    highlighted directly in the text.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: LanguageModel | None = None,
        window_size: int = 50,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.window_size = window_size
        self.use_chat_formatting = False
        self.current_html = None
        self._setup_widgets()

    @abstractmethod
    def get_feature_activation(
        self, text: str, feature_indicies: tuple[int, ...]
    ) -> th.Tensor:
        """Get the activation values for given features
        Args:
            text: Input text
            feature_indicies: Indices of features to compute
        Returns:
            Activation values for the given features as a tensor of shape (seq_len, num_features)
        """
        pass

    @th.no_grad
    def generate_model_response(self, text: str) -> str:
        """Generate model's response using the instruct model"""
        if self.model is None:
            raise ValueError("Model is not set")
        with self.model.generate(text, max_new_tokens=512):
            output = self.model.generator.output.save()
        return self.tokenizer.decode(output[0])

    def _setup_widgets(self):
        """Initialize the dashboard widgets"""
        self.text_input = widgets.Textarea(
            placeholder="Enter text to analyze...",
            description="Text:",
            layout=widgets.Layout(
                width="100%",  # Changed from 800px to 100%
                height="auto",
                font_family="sans-serif",
            ),
            style={"description_width": "initial"},
        )

        # Widget for features to compute
        self.feature_input = widgets.Text(
            placeholder="Enter features to compute [1,2,3]",
            description="Features to compute:",
            continuous_update=False,
            style={"description_width": "initial"},
        )

        # New widgets for display control
        self.highlight_feature = widgets.Text(
            placeholder="Enter feature to highlight in red",
            description="Highlight feature:",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="310px"),
        )

        self.tooltip_features = widgets.Text(
            placeholder="Enter features to show in tooltip e.g. 1,2,3",
            description="Tooltip features:",
            continuous_update=False,
            style={"description_width": "initial"},
        )

        self.analyze_button = widgets.Button(
            description="Analyze",
            button_style="primary",
            layout=widgets.Layout(
                min_width="100px",  # Ensure minimum width
                width="auto",  # Allow button to grow if needed
                text_overflow="clip",
                overflow="visible",
            ),
        )

        self.output_area = widgets.Output()
        self.analyze_button.on_click(self._handle_analysis)

        # Replace the chat formatting button with a checkbox
        self.chat_formatting = widgets.Checkbox(
            value=False,
            description="Use Chat Formatting",
            indent=False,
            style={"description_width": "initial"},
        )

        # Add generate response checkbox
        self.generate_response = widgets.Checkbox(
            value=False,
            description="Generate Response",
            indent=False,
            style={"description_width": "initial"},
        )
        if self.model is None:
            print("Model is not set, disabling generate response checkbox")
            self.generate_response.disabled = True

        # Add save button
        self.save_button = widgets.Button(
            description="Save HTML",
            button_style="success",
            disabled=True,  # Initially disabled until analysis is run
        )
        self.save_button.on_click(self._handle_save)

        # Set layout for checkbox widgets to be more compact
        self.chat_formatting.layout = widgets.Layout(
            width="auto", display="inline-flex"
        )
        self.generate_response.layout = widgets.Layout(
            width="auto", display="inline-flex"
        )

    def _create_html_highlight(
        self,
        tokens: list[str],
        activations: th.Tensor,
        all_feature_indicies: list[int],
        highlight_feature_idx: int,
        tooltip_features: list[int],
    ) -> str:
        """Create HTML with highlighted tokens based on activation values"""
        html_parts = []

        # Find highlight feature index in the activation tensor
        highlight_idx = all_feature_indicies.index(highlight_feature_idx)
        # Normalize activations for color intensity (only for highlight feature)
        highlight_acts = activations[:, highlight_idx]
        max_highlight = highlight_acts.max()
        norm_acts = highlight_acts / (max_highlight + 1e-6)

        # Create HTML spans with activation values
        tokens = sanitize_tokens(tokens, non_breaking_space=False)
        for i, token in enumerate(tokens):

            color = f"rgba(255, 0, 0, {norm_acts[i].item():.3f})"

            # Create tooltip content only for requested features
            tok_id = self.tokenizer.convert_tokens_to_ids(tokens[i])
            tooltip_lines = [f"Token {tok_id}: '{token}'"]
            for feat in tooltip_features:
                feat_idx = all_feature_indicies.index(feat)
                act_value = activations[i, feat_idx].item()
                tooltip_lines.append(f"Feature {feat}: {act_value:.3f}")

            tooltip_content = "\n".join(tooltip_lines)
            html_parts.append(create_token_html(token, color, tooltip_content))

        return "".join(html_parts)

    def _handle_analysis(self, _):
        """Handle the analysis button click"""
        try:
            # Parse feature indices for computation
            f_idx_str = self.feature_input.value.strip()
            feature_indicies = parse_list_str(f_idx_str)

            # Parse display control features
            highlight_feature = int(self.highlight_feature.value.strip())
            self.feature_to_highlight = highlight_feature
            tooltip_features = parse_list_str(self.tooltip_features.value.strip())

            # Ensure highlighted feature is included in tooltip features

            for h in tooltip_features:
                if h not in feature_indicies:
                    feature_indicies.append(h)
            if highlight_feature not in feature_indicies:
                feature_indicies.insert(0, highlight_feature)
            if highlight_feature not in tooltip_features:
                tooltip_features.insert(
                    0, highlight_feature
                )  # Add highlight feature as first
            text = self.text_input.value
            if text == "":
                print("No text to analyze")
                return
            if self.chat_formatting.value:
                text = apply_chat(
                    text,
                    self.tokenizer,
                    add_bos=False,
                )
            tokens = self.tokenizer.tokenize(text, add_special_tokens=True)
            if self.generate_response.value:
                # Generate and append model's response
                if text.startswith(self.tokenizer.bos_token):
                    text = text[len(self.tokenizer.bos_token) :]
                full_response = self.generate_model_response(text)
                text = full_response
                tokens = self.tokenizer.tokenize(text, add_special_tokens=True)

            activations = self.get_feature_activation(text, tuple(feature_indicies))
            assert (
                len(tokens) == activations.shape[0]
            ), f"Tokens are not the same length as activations, got {len(tokens)} and {activations.shape[0]}"
            with self.output_area:
                self.output_area.clear_output()

                # Create the HTML content as before
                max_acts_html = []
                for feat in tooltip_features:
                    if feat in feature_indicies:
                        feat_idx = feature_indicies.index(feat)
                        max_act = activations[:, feat_idx].max().item()
                        max_acts_html.append(f"Feature {feat} max: {max_act:.3f}")

                max_acts_display = (
                    "<div style='margin-bottom: 10px'><b>"
                    + "<br>".join(max_acts_html)
                    + "</b></div>"
                )

                html_content = self._create_html_highlight(
                    tokens,
                    activations,
                    feature_indicies,
                    highlight_feature,
                    tooltip_features,
                )
                example_html = create_example_html(
                    max_acts_display, html_content, static=True
                )

                self.current_html = create_base_html(
                    title="Feature Analysis", content=example_html
                )
                # Enable the save button now that we have content
                self.save_button.disabled = False

                # Display the HTML
                display(HTML(self.current_html))

        except ValueError:
            self.current_html = None
            self.save_button.disabled = True
            with self.output_area:
                self.output_area.clear_output()
                print("Please enter a valid feature number")
        except Exception as e:
            self.current_html = None
            self.save_button.disabled = True
            with self.output_area:
                self.output_area.clear_output()
                traceback.print_exc()

    def save_html(self, save_path: Path | None = None, filename: str | None = None):
        if self.current_html is None:
            return

        # Create directory if it doesn't exist
        if save_path is None:
            save_path = Path("results") / "features"
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        if filename is None:
            timestamp = int(time.time())
            filename = save_path / str(self.feature_to_highlight) / f"{timestamp}.html"
        else:
            filename = save_path / filename
        # Write the HTML file
        with open(filename, "w", encoding="utf-8") as f:
            html_content = create_base_html(
                title=f"Feature Analysis",
                content=self.current_html,
            )
            f.write(html_content)
        print(f"Saved analysis to {filename}")

    def _handle_save(self, _):
        """Handle saving the current HTML output"""
        self.save_html()

    def display(self):
        """Display the dashboard"""
        # Create two separate grid layouts - one for inputs, one for buttons
        inputs_layout = widgets.HBox(
            children=[
                self.feature_input,
                self.highlight_feature,
                self.tooltip_features,
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",
                gap="20px",
                width="100%",
                margin="0 0 10px 0",
                align_items="flex-start",
            ),
        )

        buttons_layout = widgets.HBox(
            children=[
                widgets.Box(
                    children=[self.analyze_button],
                    layout=widgets.Layout(margin="0 20px 0 0"),
                ),  # Right margin
                widgets.Box(
                    children=[self.chat_formatting],
                    layout=widgets.Layout(margin="0 20px 0 0"),
                ),  # Right margin
                widgets.Box(
                    children=[self.generate_response],
                    layout=widgets.Layout(margin="0 20px 0 0"),
                ),  # Right margin
                self.save_button,  # No margin needed for the last button
            ],
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",
                justify_content="flex-start",  # Align items to the start
                align_items="center",
                width="auto",  # Changed from 100% to auto
            ),
        )

        # Create the dashboard layout
        dashboard = widgets.VBox(
            [
                self.text_input,
                inputs_layout,
                buttons_layout,
                self.output_area,
            ],
            layout=widgets.Layout(
                width="100%", overflow="visible"
            ),  # Allow overflow to be visible
        )
        display(dashboard)


class OnlineFeatureCentricDashboard(AbstractOnlineFeatureCentricDashboard):
    """Implementation of AbstractOnlineFeatureCentricDashboard using functions
    given as arguments to the constructor"""

    def __init__(
        self,
        get_feature_activation: Callable[[str, tuple[int, ...]], th.Tensor],
        tokenizer: AutoTokenizer,
        generate_model_response: Callable[[str], str] | None = None,
        call_with_self: bool = False,
        model: LanguageModel | None = None,
        window_size: int = 50,
        **kwargs,
    ):
        """
        Args:
            get_feature_activation: Function to compute feature activations
            tokenizer: HuggingFace tokenizer for the model
            generate_model_response: Optional function to generate model's response
            call_with_self: Whether to call the functions with self as the first argument
            model: LanguageModel instance
            window_size: Number of tokens to show before/after the max activation token
        """
        self.call_with_self = call_with_self
        if generate_model_response is not None and model is None:
            model = DummyModel()
            warnings.warn(
                "Model is not set, using DummyModel as a placeholder to allow for response generation using your custom function"
            )
        super().__init__(tokenizer, model, window_size, **kwargs)
        self._get_feature_activation = get_feature_activation
        self._generate_model_response = generate_model_response

    def get_feature_activation(
        self, text: str, feature_indices: tuple[int, ...]
    ) -> th.Tensor:
        if self.call_with_self:
            return self._get_feature_activation(self, text, feature_indices)
        return self._get_feature_activation(text, feature_indices)

    def generate_model_response(self, text: str) -> str:
        if self._generate_model_response is None:
            return super().generate_model_response(text)
        if self.call_with_self:
            return self._generate_model_response(self, text)
        return self._generate_model_response(text)
