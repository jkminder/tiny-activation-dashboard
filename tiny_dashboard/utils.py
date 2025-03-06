import ast
import sqlite3
import json
import torch as th


def parse_list_str(s: str) -> list[int]:
    s = s.strip()
    if not s.startswith("["):
        s = "[" + s
    if not s.endswith("]"):
        s = s + "]"
    return ast.literal_eval(s)


def apply_chat(text: str, tokenizer, add_bos: bool = True) -> str:
    """Apply chat formatting to text using the tokenizer"""
    splitted = text.split("<eot>")
    is_user = True
    chat = []
    for s in splitted[:-1]:
        role = "user" if is_user else "assistant"
        chat.append({"role": role, "content": s})
        is_user = not is_user
    if is_user:
        chat.append({"role": "user", "content": splitted[-1]})
    formated_chat = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )[0 if add_bos else len(tokenizer.bos_token) :]
    if not is_user:
        formated_chat += splitted[-1]
    return formated_chat


def sanitize_html_content(s: str) -> str:
    """
    Sanitize a string to be used as HTML content.
    """
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&apos;")
        .replace('"', "&quot;")
    )


def sanitize_token(
    token: str, non_breaking_space: bool = True, keep_newline: bool = True
) -> str:
    return (
        sanitize_html_content(token)
        .replace("Ċ", "\n")
        .replace("\n", "\\n<br>" if keep_newline else "\\n")
        .replace("▁", " ")
        .replace("Ġ", " ")
        .replace(" ", "&nbsp;" if non_breaking_space else " ")
    )


def sanitize_tokens(
    tokens: list[str], non_breaking_space: bool = True, keep_newline: bool = True
) -> list[str]:
    return [sanitize_token(t, non_breaking_space, keep_newline) for t in tokens]


def update_string(s: str, str_map: dict[str, str]) -> str:
    """Update a string with a mapping from old strings to new strings."""
    for old, new in str_map.items():
        s = s.replace(old, new)
    return s


def update_template_string(s: str, str_map: dict[str, str]) -> str:
    """Update a template string with a mapping from old strings to new strings."""
    return update_string(s, {"{{" + k + "}}": v for k, v in str_map.items()})


class DummyModel:
    def __getattr__(self, name):
        if "__" in name:
            return super().__getattribute__(name)
        raise ValueError(
            f"Attempted to access '{name}' on a DummyModel instance, which is intended solely as a placeholder."
        )

    def __getattribute__(self, name):
        if "__" in name:
            return super().__getattribute__(name)
        raise ValueError(
            f"Attempted to access '{name}' on a DummyModel instance, which is intended solely as a placeholder."
        )

    def __call__(self, *args, **kwargs):
        raise ValueError(
            "Attempted to call a DummyModel instance, which is intended solely as a placeholder."
        )


class LazyReadDict:
    def __init__(self, db_path, column_name: str, table_name="data_table"):
        self.db_path = db_path
        self.column_name = column_name
        self.table_name = table_name
        self._init_keys()

    def _init_keys(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT key FROM {self.table_name}")
            self._keys = [row[0] for row in cursor.fetchall()]

    def __getitem__(self, key):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self.column_name} FROM {self.table_name} WHERE key = ?",
                (key,),
            )
            rows = cursor.fetchall()

            if not rows:
                raise KeyError(key)

            if len(rows) > 1:
                raise ValueError(f"Multiple entries found for key {key}")
            return json.loads(rows[0][0])

    def keys(self):
        return self._keys

    def __contains__(self, key):
        return key in self._keys


def convert_to_latex(
    tokens: list[str],
    activations: th.Tensor,
    feature_indices: list[int],
    max_acts: dict[int] | None = None,
) -> str:
    """Convert tokens and their activations to LaTeX format."""
    if activations.dim() == 1:
        activations = activations.unsqueeze(1)

    # Handle normalization based on max_acts if provided
    if max_acts is not None and len(feature_indices) > 0:
        max_act = max_acts.get(feature_indices[0])
        if max_act is not None:
            norm_acts = activations / max_act
            curr_max = activations.max().item()
        else:
            max_acts = activations.max(dim=0)[0]
            norm_acts = activations / max_acts.unsqueeze(0)
            curr_max = activations.max().item()
    else:
        max_acts = activations.max(dim=0)[0]
        norm_acts = activations / max_acts.unsqueeze(0)
        curr_max = activations.max().item()

    # Convert activations to color specifications
    colors = []
    for i in range(len(tokens)):
        opacity = min(100, max(0, int(norm_acts[i, 0].item() * 100)))
        colors.append(f"red!{opacity}")

    def latex_escape(s):
        """Escape special characters and detect newlines"""
        newline = "\n" in s
        # First handle Unicode characters and special tokens
        if s in [" ", "Ġ", "▁"]:
            s = " "
        else:
            s = s.replace("▁", " ").replace("Ġ", " ")

        # Handle LaTeX special characters
        s = (
            s.replace("\\", r"\textbackslash{}")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("_", r"\_")
            .replace("^", r"\^{}")
            .replace("$", r"\$")
            .replace("#", r"\#")
            .replace("%", r"\%")
            .replace("&", r"\&")
            .replace("~", r"\~{}")
            .replace("\n", r"\textbackslash n")
        )

        return s, newline

    # Generate the preamble
    preamble = [
        "\\newcommand{\\hlbg}[2]{%",
        "  \\setlength{\\fboxsep}{0pt}%    no horizontal/vertical padding",
        "  \\setlength{\\fboxrule}{0pt}%   no rule/border around the box",
        "  \\colorbox{#1}{\\strut #2}%",
        "}",
        "",
        "% 2) Define a listing style:",
        "\\lstdefinestyle{colorchars}{",
        "  basicstyle=\\sffamily\\small,",
        "  columns=fixed,",
        "  escapechar=|,",
        "  keepspaces=true,",
        "  showstringspaces=false,",
        "  breaklines=true,",
        "  breakatwhitespace=false,",
        "}",
    ]

    # Build the content
    content_lines = []

    # Header with feature info
    header = (
        "  \\begin{tabular}{|p{0.95\\columnwidth}|}  % Fixed width relative to column width\n"
        "    \\hline\n"
        "\\cellcolor{gray!20}\\begin{minipage}[t]{0.95\\columnwidth}   % Match width with tabular\n"
        "\\begin{lstlisting}[style=colorchars,aboveskip=-5pt,belowskip=0pt]\n"
        f"|\\textbf{{Feature {feature_indices[0]}}}|\n"
        f"|Max Activation: {curr_max:.3f}|\\end{{lstlisting}}\n"
        "\\end{minipage} \\\\\n"
    )

    # Token content
    token_content = (
        "\\begin{minipage}[t]{0.95\\columnwidth}  % Match width with tabular\n"
        "\\begin{lstlisting}[style=colorchars,aboveskip=-5pt,belowskip=3pt,breaklines=true,breakatwhitespace=false]\n"
    )

    # Add tokens with their colors
    current_line = []
    for token, color in zip(tokens, colors):
        escaped_token, newline = latex_escape(token)
        token_str = f"|\\allowbreak\\hlbg{{{color}}}{{{escaped_token}}}|"
        current_line.append(token_str)
        if newline:
            current_line.append("\n")
    token_content += "".join(current_line) + "\\end{lstlisting}\n"
    token_content += "\\end{minipage} \\\\ \\hline\n"
    token_content += "\\end{tabular}"

    # Combine all parts
    return {"preamble": "\n".join(preamble), "content": header + token_content}
