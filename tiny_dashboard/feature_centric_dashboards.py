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
from .utils import (
    sanitize_tokens,
    apply_chat,
    parse_list_str,
    DummyModel,
    sanitize_token,
    LazyReadDict,
)
from .html_utils import (
    create_example_html,
    create_base_html,
    create_token_html,
    create_highlighted_tokens_html,
)


class OfflineFeatureCentricDashboard:
    """
    This Dashboard is composed of a feature selector and a feature viewer.
    The feature selector allows you to select a feature and view the max activating examples for that feature.
    The feature viewer displays the max activating examples for the selected feature. Text is highlighted with a gradient based on the activation value of each token.
    An hover text showing the activation value, token id is also shown. When the mouse passes over a token, the token is highlighted in light grey.
    By default, the text sample is not displayed entirely, but only a few tokens before and after the highest activating token. If the user clicks on the text sample, the entire text sample is displayed.
    """

    @classmethod
    def from_db(
        cls,
        db_path: Path | str,
        tokenizer: AutoTokenizer,
        column_name: str = "examples",
        table_name: str = "data_table",
        window_size: int = 50,
        max_examples: int = 30,
    ):
        """
        Create an OfflineFeatureCentricDashboard instance from a database file.
        This is useful to avoid loading the entire max activation examples into memory.

        Args:
            db_path (Path): Path to the database file, which should contain entries in the format:
                key: int -> examples: list of tuples, where each tuple consists of:
                (max_activation_value: float, tokens: list of str, activation_values: list of float).
                The examples are stored as a JSON string in the database.
            tokenizer (AutoTokenizer): A HuggingFace tokenizer used for processing the model's input.
            window_size (int, optional): The number of tokens to display before and after the token with the maximum activation. Defaults to 50.
            max_examples (int, optional): The maximum number of examples to display for each feature. Defaults to 30.
            column_name (str, optional): The name of the column to read from. Defaults to "examples".
            table_name (str, optional): The name of the table to read from. Defaults to "data_table".
        """
        max_activation_examples = LazyReadDict(db_path, column_name, table_name)
        return cls(max_activation_examples, tokenizer, window_size, max_examples)

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
        self._setup_widgets()
        self.use_absolute_max = False
        self.feature_idx = None

    def _setup_widgets(self):
        """Initialize the dashboard widgets"""

        self.available_features = sorted(self.max_activation_examples.keys())
        self.feature_selector = widgets.Text(
            placeholder="Type a feature number...",
            description="Feature:",
            continuous_update=False,  # Only trigger on Enter/loss of focus
            style={"description_width": "initial"},
        )

        # Add checkbox for absolute max
        self.use_absolute_max_checkbox = widgets.Checkbox(
            value=False,
            description="Use Absolute Max",
            indent=False,
            style={"description_width": "initial"},
        )

        self.examples_output = widgets.Output()
        self.feature_selector.observe(self._handle_feature_selection, names="value")
        self.use_absolute_max_checkbox.observe(
            self._handle_absolute_max_change, names="value"
        )

    def _handle_feature_selection(self, change):
        """Handle feature selection, including validation of typed input"""
        try:
            feature_idx = int(change["new"])
            if feature_idx in self.max_activation_examples:
                self.feature_idx = feature_idx
                self._update_examples()
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

    def _handle_absolute_max_change(self, change):
        print(f"use_absolute_max changed to {change}")
        self.use_absolute_max = change["new"]
        self._update_examples()

    def _create_html_highlight(
        self,
        tokens: list[str],
        activations: list[float],
        max_idx: int,
        show_full: bool = False,
        min_max_act: float = None,
    ) -> str:
        act_tensor = th.tensor(activations)

        # Determine window bounds
        if not show_full:
            start_idx = max(0, max_idx - self.window_size)
            end_idx = min(len(tokens), max_idx + self.window_size + 1)
            tokens = tokens[start_idx:end_idx]
            act_tensor = act_tensor[start_idx:end_idx]

        return create_highlighted_tokens_html(
            tokens=tokens,
            activations=act_tensor,
            tokenizer=self.tokenizer,
            highlight_features=0,  # Single feature case
            color1=(255, 0, 0),  # Red color
            activation_names=["Activation"],
            min_max_act=min_max_act,
        )

    def generate_html(self, feature_idx: int, use_absolute_max: bool = False) -> str:
        examples = self.max_activation_examples[feature_idx]

        content_parts = []
        min_max_act = None
        if use_absolute_max:
            min_max_act = examples[0][0]
        for max_act, tokens, token_acts in list(examples)[: self.max_examples]:
            max_idx = np.argmax(token_acts)

            # Create both versions
            collapsed_html = self._create_html_highlight(
                tokens, token_acts, max_idx, False, min_max_act
            )
            full_html = self._create_html_highlight(
                tokens, token_acts, max_idx, True, min_max_act
            )

            content_parts.append(
                create_example_html(max_act, collapsed_html, full_html)
            )

        # Display the HTML content all at once
        html_content = create_base_html(
            title=f"Feature {feature_idx} Examples",
            content=content_parts,
        )
        return html_content

    def _update_examples(self):
        """Update the examples display when a new feature is selected"""
        # Clear the output first
        if self.feature_idx is None:
            print("No feature selected")
            return
        self.examples_output.clear_output(
            wait=True
        )  # wait=True for smoother transition
        with self.examples_output:
            display(HTML(self.generate_html(self.feature_idx, self.use_absolute_max)))

    def display(self):
        """Display the dashboard"""

        dashboard = widgets.VBox(
            [
                widgets.HBox([self.feature_selector, self.use_absolute_max_checkbox]),
                self.examples_output,
            ]
        )
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
        max_acts: dict[int, float] | None = None,
        second_highlight_color: tuple[int, int, int] | None = None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.window_size = window_size
        self.use_chat_formatting = False
        self.current_html = None
        self.max_acts = max_acts
        self.second_highlight_color = second_highlight_color
        self._setup_widgets()

    @abstractmethod
    def get_feature_activation(
        self, text: str, feature_indices: tuple[int, ...]
    ) -> th.Tensor:
        """Get the activation values for given features
        Args:
            text: Input text
            feature_indices: Indices of features to compute
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

        # Update highlight feature input to be more flexible
        self.highlight_feature = widgets.Text(
            placeholder="Enter 1-2 features to highlight (e.g. 1 or 1,2)",
            description="Highlight features:",
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

        # Add new min_max_act widget after existing widgets
        self.min_max_act_input = widgets.Text(
            placeholder="empty, 'auto' or float value",
            description="Max act:",
            value="",
            continuous_update=False,
            style={"description_width": "initial"},
        )

    def _create_html_highlight(
        self,
        tokens: list[str],
        activations: th.Tensor,
        all_feature_indices: list[int],
        highlight_features: list[int],
        tooltip_features: list[int],
        return_max_acts: bool = False,
    ) -> str | tuple[str, str]:
        """Create HTML with highlighted tokens based on activation values"""
        # Map feature indices to their positions in the activations tensor
        highlight_positions = [all_feature_indices.index(f) for f in highlight_features]
        tooltip_positions = [all_feature_indices.index(f) for f in tooltip_features]

        # Create feature names mapping indices to their original feature numbers
        activation_names = [
            f"Feature {all_feature_indices[i]}" for i in range(len(all_feature_indices))
        ]

        # Handle min_max_act value
        min_max_act = None
        min_max_act_value = self.min_max_act_input.value.strip().lower()

        if min_max_act_value == "":
            min_max_act = None
        elif min_max_act_value != "auto":
            try:
                min_max_act = float(min_max_act_value)
            except ValueError:
                raise ValueError("Min-max act must be empty, 'auto' or a float value")
        elif min_max_act_value == "auto" and self.max_acts is not None:
            # Use the first highlight feature's max_act value
            feature = highlight_features[0]
            if feature in self.max_acts:
                min_max_act = self.max_acts[feature]
            else:
                raise ValueError(f"No max activation value found for feature {feature}")
        elif min_max_act_value == "auto":
            raise ValueError(
                "Cannot use 'auto' without max_acts dictionary provided during initialization"
            )

        return create_highlighted_tokens_html(
            tokens=tokens,
            activations=activations,
            tokenizer=self.tokenizer,
            highlight_features=highlight_positions,
            tooltip_features=tooltip_positions,
            color1=(255, 0, 0),
            color2=self.second_highlight_color,
            activation_names=activation_names,
            return_max_acts_str=return_max_acts,
            min_max_act=min_max_act,
        )

    def _handle_analysis(self, _):
        """Handle the analysis button click"""
        try:
            # Parse feature indices for computation
            f_idx_str = self.feature_input.value.strip()
            feature_indices = parse_list_str(f_idx_str)

            # Parse features for highlighting - now accepts 1 or 2 features
            highlight_features = parse_list_str(self.highlight_feature.value.strip())
            self.highlight_features = highlight_features
            if len(highlight_features) not in (1, 2):
                raise ValueError("Please enter one or two features to highlight")

            # Parse display control features
            tooltip_features = parse_list_str(self.tooltip_features.value.strip())

            # Ensure all required features are included in computation
            feature_indices = list(
                dict.fromkeys(highlight_features + tooltip_features + feature_indices)
            )

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

            activations = self.get_feature_activation(text, tuple(feature_indices))
            assert (
                len(tokens) == activations.shape[0]
            ), f"Tokens are not the same length as activations, got {len(tokens)} and {activations.shape[0]}"
            with self.output_area:
                self.output_area.clear_output()

                # Get HTML content and max activations string
                html_content, max_acts_str = self._create_html_highlight(
                    tokens,
                    activations,
                    feature_indices,
                    highlight_features,
                    tooltip_features,
                    return_max_acts=True,
                )

                example_html = create_example_html(
                    max_acts_str, html_content, static=True
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
            filename = (
                save_path
                / str("_".join(map(str, self.highlight_features)))
                / f"{timestamp}.html"
            )
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
                self.min_max_act_input,
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
                "Warning:\nModel is not set, using DummyModel as a placeholder to allow for response generation using your custom function"
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
