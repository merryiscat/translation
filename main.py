"""
Main CLI interface for MD translation.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path
import yaml
from tqdm import tqdm
from dotenv import load_dotenv

from src.utils import extract_preserved_elements, format_size
from src.chunker import chunk_markdown
from src.translator import create_translator_from_config
from src.merger import merge_chunks


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Replace environment variables in config
    api_key = config.get('api', {}).get('api_key', '')
    if api_key.startswith('${') and api_key.endswith('}'):
        env_var = api_key[2:-1]
        config['api']['api_key'] = os.environ.get(env_var, '')

    return config


async def translate_file(input_path: str, output_path: str, config: dict):
    """
    Main translation workflow.

    Args:
        input_path: Path to input markdown file
        output_path: Path to output translated file
        config: Configuration dictionary
    """
    print(f"\n📄 Reading file: {input_path}")

    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"   File size: {format_size(len(content))}")

    # Extract preserved elements
    print("\n🔍 Extracting preserved elements (code blocks, math, links)...")
    content_with_placeholders, preserved_elements = extract_preserved_elements(content)

    total_preserved = sum(len(v) for v in preserved_elements.values())
    print(f"   Preserved {total_preserved} elements")
    for elem_type, elems in preserved_elements.items():
        if elems:
            print(f"   - {elem_type}: {len(elems)}")

    # Chunk content
    chunk_size = config.get('translation', {}).get('chunk_size', 10000)
    print(f"\n✂️  Chunking content (target size: {chunk_size} chars)...")
    chunks = chunk_markdown(content_with_placeholders, chunk_size=chunk_size)
    print(f"   Created {len(chunks)} chunks")

    # Create translator
    translator = create_translator_from_config(config)

    # Translate chunks with progress bar
    print(f"\n🌐 Translating {len(chunks)} chunks...")
    print(f"   Model: {translator.model}")

    translated_chunks = []
    with tqdm(total=len(chunks), desc="Translation progress", unit="chunk") as pbar:
        def update_progress(current, total, info):
            pbar.update(1)

        translated_chunks = await translator.translate_chunks(
            chunks,
            progress_callback=update_progress
        )

    # Merge chunks
    print("\n🔗 Merging translated chunks...")
    final_content = merge_chunks(translated_chunks, preserved_elements)

    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output file
    print(f"\n💾 Writing output: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f"\n✅ Translation complete!")
    print(f"   Input:  {input_path} ({format_size(len(content))})")
    print(f"   Output: {output_path} ({format_size(len(final_content))})")


def main():
    """CLI entry point."""
    # Load .env file if it exists
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print("✓ Loaded environment variables from .env")

    parser = argparse.ArgumentParser(
        description="Translate markdown files using LLM API"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input markdown file path'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output file path (default: translated/<input_filename>)'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"❌ Error: Configuration file not found: {args.config}")
        print(f"   Please create {args.config} or specify --config path")
        sys.exit(1)

    config = load_config(args.config)

    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        output_dir = config.get('translation', {}).get('output_dir', 'translated')
        output_path = input_path.parent / output_dir / input_path.name
    else:
        output_path = Path(args.output)

    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    # Check API key (from env or config)
    api_key = os.environ.get('LLM_API_KEY', config.get('api', {}).get('api_key', ''))
    if not api_key:
        print("❌ Error: API key not found")
        print("   Set LLM_API_KEY in .env file or environment variable")
        sys.exit(1)

    # Check endpoint
    endpoint = os.environ.get('LLM_API_ENDPOINT', config.get('api', {}).get('endpoint', ''))
    if not endpoint:
        print("❌ Error: API endpoint not found")
        print("   Set LLM_API_ENDPOINT in .env file or config.yaml")
        sys.exit(1)

    # Run translation
    try:
        asyncio.run(translate_file(args.input, str(output_path), config))
    except KeyboardInterrupt:
        print("\n\n⚠️  Translation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during translation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
