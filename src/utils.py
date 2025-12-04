"""
Utility functions for preserving markdown elements during translation.
"""
import re
from typing import Dict, Tuple


# Regex patterns for elements to preserve
PATTERNS = {
    'code_block': r'```[\s\S]*?```',
    'math_block': r'\$\$[\s\S]*?\$\$',
    'inline_math': r'\$[^\$\n]+?\$',
    'inline_code': r'`[^`\n]+?`',
    'link': r'\[([^\]]+)\]\(([^)]+)\)',
    'image': r'!\[([^\]]*)\]\(([^)]+)\)'
}


def extract_preserved_elements(content: str) -> Tuple[str, Dict[str, list]]:
    """
    Extract code blocks, math, links, and other elements to preserve.

    Args:
        content: Original markdown content

    Returns:
        Tuple of (content_with_placeholders, preserved_elements_dict)
    """
    preserved = {key: [] for key in PATTERNS.keys()}
    modified_content = content

    # Extract in order of precedence (largest to smallest)
    # Code blocks first (to avoid conflicts with inline code)
    for element_type in ['code_block', 'math_block', 'image', 'link', 'inline_math', 'inline_code']:
        pattern = PATTERNS[element_type]
        matches = list(re.finditer(pattern, modified_content))

        # Replace from end to start to maintain positions
        for match in reversed(matches):
            element = match.group(0)
            index = len(preserved[element_type])
            preserved[element_type].append(element)

            placeholder = f"{{{{{element_type.upper()}_{index}}}}}"
            modified_content = (
                modified_content[:match.start()] +
                placeholder +
                modified_content[match.end():]
            )

    return modified_content, preserved


def restore_preserved_elements(content: str, preserved: Dict[str, list]) -> str:
    """
    Restore preserved elements from placeholders.

    Args:
        content: Content with placeholders
        preserved: Dictionary of preserved elements

    Returns:
        Content with restored original elements
    """
    restored_content = content

    # Restore in reverse order (smallest to largest)
    for element_type in ['inline_code', 'inline_math', 'link', 'image', 'math_block', 'code_block']:
        elements = preserved.get(element_type, [])
        for idx, element in enumerate(elements):
            placeholder = f"{{{{{element_type.upper()}_{idx}}}}}"
            restored_content = restored_content.replace(placeholder, element)

    return restored_content


def verify_placeholders(original_preserved: Dict[str, list], translated_content: str) -> bool:
    """
    Verify that all placeholders are present in translated content.

    Args:
        original_preserved: Original preserved elements dictionary
        translated_content: Translated content with placeholders

    Returns:
        True if all placeholders are present, False otherwise
    """
    for element_type, elements in original_preserved.items():
        for idx in range(len(elements)):
            placeholder = f"{{{{{element_type.upper()}_{idx}}}}}"
            if placeholder not in translated_content:
                print(f"⚠️  Warning: Missing placeholder {placeholder}")
                return False
    return True


def count_chars(text: str) -> int:
    """
    Count characters in text.

    Args:
        text: Input text

    Returns:
        Character count
    """
    return len(text)


def format_size(size: int) -> str:
    """
    Format size in human-readable format.

    Args:
        size: Size in characters

    Returns:
        Formatted size string (e.g., "1.2K chars", "10.5K chars")
    """
    if size < 1000:
        return f"{size} chars"
    else:
        return f"{size/1000:.1f}K chars"
