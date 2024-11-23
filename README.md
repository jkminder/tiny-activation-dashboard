# Tiny Activation Dashboard
A tiny easily hackable implementation of a feature dashboard.

## Overview

This repository provides a powerful and intuitive tool for visualizing and exploring feature activations in neural language models, with a focus on making complex model interpretability more accessible.

## Motivation

There are some other good feature activations dashboard tools out there, but I found them very hard to hack on when I wanted to add support for Crosscoders. This implementation is not as complete as https://github.com/jbloomAus/SAEDashboard or even the simplier https://github.com/callummcdougall/sae_vis but in my honest non-biased-at-all opinion, this implementation seems easier to hack on?

## Key Features
Both the offline and online dashboards include:
- Token-level activation highlighting
- Hover tooltips showing token details
- Responsive design
- Save HTML reports

### 1. Offline Feature Exploration
- Analyze pre-computed feature activations
- Visualize max activation examples for specific features
- Expandable text views
- Generate interactive HTML reports


```py
from src.feature_centric_dashboards import OfflineFeatureCentricDashboard

# Create dashboard with pre-computed activations
max_activation_examples: dict[int, list[tuple[float, list[str], list[float]]]] = ...
# max_activation_examples is a dictionary where the keys are feature indices and the values are lists of tuples. Each tuple contains a float (max activation value), a list of strings (the text of the example), and a list of floats (the activation values for each token in the example).

dashboard = OfflineFeatureCentricDashboard(max_activation_examples, tokenizer)
dashboard.display()

# Export to HTML for sharing
feature_to_export = 0
dashboard.export_to_html("feature_analysis.html", feature_to_export)
```

### 2. Online Feature Exploration

The online dashboard allows you to analyze the activations of a model in real-time. This is useful for quickly exploring the activations of a model on your custom prompts.

The online dashboard supports `chat_template` formatting: just include `<eot>` in your input text to separate your chat turns. E.g:

```
What is the capital of France?<eot>The capital of France is Paris.<eot>Good bing
```
will be interpreted as:
```json
[
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "Good bing"}
]
```
and formated using the tokenizer's chat template.

Two approaches to build your real-time feature analysis dashboard:

#### A. Class-based Method
Create a class that implements the `AbstractOnlineFeatureCentricDashboard` class and implements the `get_feature_activation` function. This function should take a string and a tuple of feature indices and return a tensor of activation values of shape (seq_len, num_features) containing the activations of the specified features for the input text.

```py
class DummyOnlineFeatureCentricDashboard(AbstractOnlineFeatureCentricDashboard):
    def get_feature_activation(self, text: str, feature_indices: tuple[int, ...]) -> th.Tensor:
        # Custom activation computation logic
        tok_len = len(self.tokenizer.encode(text))
        activations = th.randn((tok_len, len(feature_indices))).exp()
        return activations
    
    # Optional: override generate_model_response to change the model's response generation

online_dashboards = DummyOnlineFeatureCentricDashboard(tokenizer, model)
online_dashboards.display()
```

#### B. Function-based Method
If you hate classes for some reason, you can also use the function-based method:
```py
def get_feature_activation(text, feature_indices):
    return th.randn((len(tokenizer.encode(text)), len(feature_indices))).exp()

online_dashboards = OnlineFeatureCentricDashboard(
    get_feature_activation, 
    tokenizer,
    generate_model_response = None,  # Optional: override the model's response generation function
    model = None,  # Optional: pass in a model to use the model's response generation function
    call_with_self = False,  # Whether to call the functions with self as the first argument, defaults to Falses
)
online_dashboards.display()
```


## Example Workflow

1. Load a pre-trained language model
2. Compute feature activations
3. Create a dashboard
4. Explore and analyze feature behaviors

## Repository Structure

The repository is organized as follows:

- `demo.ipynb`: A Jupyter notebook containing minimal examples demonstrating how to use both offline and online dashboards
- `src/`: Main package directory
  - `feature_centric_dashboards.py`: Core implementation of the dashboard classes (OfflineFeatureCentricDashboard, OnlineFeatureCentricDashboard, and AbstractOnlineFeatureCentricDashboard)
  - `html_utils.py`: Utility functions for generating HTML elements using templates
  - `utils.py`: General utility functions for text processing and HTML sanitization
  - `templates/`: HTML, CSS, and JavaScript templates
    - HTML templates for different components (base layout, feature sections, examples, etc.)
    - `styles.css`: CSS styling for the dashboard
    - `listeners.js`: JavaScript for interactive features (tooltips, expandable text)

## Installation

```bash
pip install git+https://github.com/butanium/tiny-activation-dashboard.git
```

## Contributing

Contributions are welcome! Please feel free to improve the minimal design and add some usage examples.
