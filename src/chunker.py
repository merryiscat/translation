"""
Smart chunking of markdown content for translation.
"""
from typing import List
from .utils import count_chars


def chunk_markdown(content: str, chunk_size: int = 10000, tolerance: int = 1000) -> List[str]:
    """
    Split markdown content into chunks at paragraph boundaries.

    Args:
        content: Markdown content (with placeholders already substituted)
        chunk_size: Target chunk size in characters (default: 10000)
        tolerance: Acceptable deviation from chunk_size (default: 1000)

    Returns:
        List of content chunks
    """
    if count_chars(content) <= chunk_size:
        return [content]

    chunks = []
    current_pos = 0
    content_length = len(content)

    while current_pos < content_length:
        # Calculate target end position
        target_end = min(current_pos + chunk_size, content_length)

        # If this is the last chunk, take everything
        if target_end >= content_length:
            chunks.append(content[current_pos:])
            break

        # Find the best split point (paragraph boundary)
        split_point = find_split_point(
            content,
            current_pos,
            target_end,
            chunk_size,
            tolerance
        )

        # Extract chunk
        chunk = content[current_pos:split_point].strip()
        if chunk:
            chunks.append(chunk)

        current_pos = split_point

    return chunks


def find_split_point(
    content: str,
    start: int,
    target: int,
    chunk_size: int,
    tolerance: int
) -> int:
    """
    Find the best split point near target position.

    Strategy:
    1. Look for double newlines (paragraph boundaries) within tolerance range
    2. If not found, look for single newlines
    3. If still not found, split at target position

    Args:
        content: Full content
        start: Start position of current chunk
        target: Target end position
        chunk_size: Desired chunk size
        tolerance: Acceptable deviation

    Returns:
        Best split position
    """
    min_pos = max(start, target - tolerance)
    max_pos = min(len(content), target + tolerance)

    # Search range
    search_range = content[min_pos:max_pos]

    # Strategy 1: Find double newline (paragraph boundary)
    paragraph_breaks = [m.end() for m in re.finditer(r'\n\n+', search_range)]
    if paragraph_breaks:
        # Find the one closest to target
        best_break = min(paragraph_breaks, key=lambda x: abs((min_pos + x) - target))
        return min_pos + best_break

    # Strategy 2: Find single newline
    line_breaks = [m.end() for m in re.finditer(r'\n', search_range)]
    if line_breaks:
        best_break = min(line_breaks, key=lambda x: abs((min_pos + x) - target))
        return min_pos + best_break

    # Strategy 3: Split at target (last resort)
    return target


import re
