"""Citation manipulation helpers for chunked synthesis."""
import re


def offset_citations(text: str, offset: int) -> str:
    """Remap [N] -> [N+offset] in LLM response text.

    Uses temp markers to avoid double-replacement when [1] appears before [12].
    """
    if offset == 0:
        return text

    # Handle comma-separated [1, 5, 12] first
    def remap_comma(match):
        nums = [int(n.strip()) for n in match.group(1).split(',')]
        return '[' + ', '.join(f'\x00{n + offset}\x00' for n in nums) + ']'

    text = re.sub(r'\[(\d+(?:\s*,\s*\d+)+)\]', remap_comma, text)

    # Handle single [N] — largest first to avoid [1] eating [12]
    cited = sorted(set(int(m) for m in re.findall(r'\[(\d+)\]', text)), reverse=True)
    for n in cited:
        text = text.replace(f'[{n}]', f'[\x00{n + offset}\x00]')

    return text.replace('\x00', '')


def validate_citations(text: str, max_source: int) -> str:
    """Strip citations [N] where N is outside 1..max_source.

    Normalizes whitespace left by removed citations.
    """
    def check_comma(match):
        nums = [int(n.strip()) for n in match.group(1).split(',')]
        valid = [n for n in nums if 1 <= n <= max_source]
        return f'[{", ".join(str(n) for n in valid)}]' if valid else ''

    def check_single(match):
        n = int(match.group(1))
        return match.group(0) if 1 <= n <= max_source else ''

    text = re.sub(r'\[(\d+(?:\s*,\s*\d+)+)\]', check_comma, text)
    text = re.sub(r'\[(\d+)\]', check_single, text)

    # Whitespace cleanup
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r' ([.,;:!?])', r'\1', text)
    text = re.sub(r'^ +', '', text, flags=re.MULTILINE)  # leading spaces per line
    text = re.sub(r' +$', '', text, flags=re.MULTILINE)  # trailing spaces per line
    return text
