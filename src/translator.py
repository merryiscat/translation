"""
Translation engine with API calls and retry logic.
"""
import aiohttp
import asyncio
import os
import json
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential


class Translator:
    """Translator class for handling API calls."""

    def __init__(self, endpoint: str, api_key: str, model: str, max_retries: int = 3, request_template: Optional[str] = None):
        """
        Initialize translator.

        Args:
            endpoint: API endpoint URL
            api_key: API key for authentication
            model: Model name to use
            max_retries: Maximum number of retry attempts
            request_template: Optional custom request body template (JSON string)
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.request_template = request_template

        # System prompt for translation
        self.system_prompt = (
            "You are a professional academic translator. "
            "Translate the following English text to Korean. "
            "IMPORTANT: Preserve all placeholders like {{CODE_BLOCK_0}}, {{MATH_0}}, etc. exactly as they appear. "
            "Only translate the actual text content, not the placeholders."
        )

    def _build_request_body(self, chunk: str) -> Dict[str, Any]:
        """
        Build request body from template or default format.

        Args:
            chunk: Text chunk to translate

        Returns:
            Request body dictionary
        """
        user_content = f"Translate to Korean:\n\n{chunk}"

        if self.request_template:
            # Use custom template from environment variable
            try:
                # Replace template variables
                template_str = self.request_template
                template_str = template_str.replace("{model}", self.model)
                template_str = template_str.replace("{system_prompt}", self.system_prompt)
                template_str = template_str.replace("{user_content}", user_content)
                template_str = template_str.replace("{temperature}", "0.3")

                return json.loads(template_str)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Invalid request template JSON, using default format: {e}")

        # Default: OpenAI-compatible format
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "temperature": 0.3
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def translate_chunk(self, chunk: str) -> str:
        """
        Translate a single chunk with retry logic.

        Args:
            chunk: Text chunk to translate (may contain placeholders)

        Returns:
            Translated text

        Raises:
            Exception: If translation fails after all retries
        """
        # Build request body
        request_body = self._build_request_body(chunk)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                json=request_body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120)  # 2 minutes timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")

                result = await response.json()

                # Extract translated text from response
                # Try OpenAI-compatible format first
                try:
                    translated = result['choices'][0]['message']['content']
                    return translated.strip()
                except (KeyError, IndexError):
                    # Try alternative formats
                    if 'content' in result:
                        return result['content'].strip()
                    elif 'text' in result:
                        return result['text'].strip()
                    else:
                        raise Exception(f"Failed to parse API response. Response: {result}")

    async def translate_chunks(self, chunks: list[str], progress_callback=None) -> list[str]:
        """
        Translate multiple chunks sequentially.

        Args:
            chunks: List of text chunks to translate
            progress_callback: Optional callback function(current, total, chunk_info)

        Returns:
            List of translated chunks
        """
        translated_chunks = []

        for idx, chunk in enumerate(chunks, 1):
            if progress_callback:
                chunk_info = f"chunk {idx}/{len(chunks)}"
                progress_callback(idx - 1, len(chunks), chunk_info)

            try:
                translated = await self.translate_chunk(chunk)
                translated_chunks.append(translated)

                # Small delay to avoid overwhelming the API
                if idx < len(chunks):
                    await asyncio.sleep(0.5)

            except Exception as e:
                print(f"\n❌ Error translating chunk {idx}: {e}")
                print(f"Keeping original chunk for this section.")
                translated_chunks.append(chunk)

        return translated_chunks


def create_translator_from_config(config: dict) -> Translator:
    """
    Create translator instance from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Translator instance
    """
    api_config = config.get('api', {})
    retry_config = config.get('retry', {})

    # Get configuration from environment variables (priority) or config file
    api_key = os.environ.get('LLM_API_KEY', api_config.get('api_key', ''))
    endpoint = os.environ.get('LLM_API_ENDPOINT', api_config.get('endpoint', ''))
    model = os.environ.get('LLM_MODEL_NAME', api_config.get('model', ''))
    request_template = os.environ.get('LLM_REQUEST_TEMPLATE', None)

    return Translator(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        max_retries=retry_config.get('max_attempts', 3),
        request_template=request_template
    )
