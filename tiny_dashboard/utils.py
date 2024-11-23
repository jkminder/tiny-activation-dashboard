import ast


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
    for s in splitted:
        role = "user" if is_user else "assistant"
        chat.append({"role": role, "content": s})
        is_user = not is_user
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=not is_user
    )[0 if add_bos else len(tokenizer.bos_token) :]


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


def sanitize_token(token: str, non_breaking_space: bool = True) -> str:
    return (
        sanitize_html_content(token)
        .replace("\n", "\\n\n")
        .replace("▁", " ")
        .replace("Ġ", " ")
        .replace(" ", "&nbsp;" if non_breaking_space else " ")
    )


def sanitize_tokens(tokens: list[str], non_breaking_space: bool = True) -> list[str]:
    return [sanitize_token(t, non_breaking_space) for t in tokens]


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
