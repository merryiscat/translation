"""
Merge translated chunks back into a single document.
"""
from typing import List, Dict
from .utils import restore_preserved_elements, verify_placeholders


def merge_chunks(
    translated_chunks: List[str],
    preserved_elements: Dict[str, list],
    verify: bool = True
) -> str:
    """
    Merge translated chunks and restore preserved elements.

    Args:
        translated_chunks: List of translated text chunks
        preserved_elements: Dictionary of preserved elements
        verify: Whether to verify placeholders before restoration

    Returns:
        Final merged and restored markdown content
    """
    # Merge chunks with double newline separator
    merged_content = "\n\n".join(translated_chunks)

    # Optionally verify placeholders
    if verify:
        all_valid = verify_placeholders(preserved_elements, merged_content)
        if not all_valid:
            print("⚠️  Warning: Some placeholders are missing in translation.")
            print("   This may result in incomplete restoration.")

    # Restore preserved elements
    final_content = restore_preserved_elements(merged_content, preserved_elements)

    return final_content


def merge_with_original_on_error(
    chunks: List[str],
    translated_chunks: List[str],
    preserved_elements: Dict[str, list]
) -> str:
    """
    Merge with fallback to original chunks on translation errors.

    This function is useful when some chunks failed to translate,
    and you want to keep original chunks for failed sections.

    Args:
        chunks: Original chunks
        translated_chunks: Translated chunks (may contain originals for failed ones)
        preserved_elements: Dictionary of preserved elements

    Returns:
        Final merged content with restored elements
    """
    # Ensure same length
    if len(chunks) != len(translated_chunks):
        raise ValueError(
            f"Chunk count mismatch: {len(chunks)} original vs "
            f"{len(translated_chunks)} translated"
        )

    # Merge chunks
    merged_content = "\n\n".join(translated_chunks)

    # Restore preserved elements
    final_content = restore_preserved_elements(merged_content, preserved_elements)

    return final_content
