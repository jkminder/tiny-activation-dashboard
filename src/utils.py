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
        chat, tokenize=False, add_generation_prompt=True
    )[0 if add_bos else 5 :]


def sanitize_tokens(tokens: list[str]) -> list[str]:
    return [
        t.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&apos;")
        .replace('"', "&quot;")
        .replace("\n", "\\n\n")
        .replace("▁", " ")
        .replace("Ġ", " ")
        for t in tokens
    ]
