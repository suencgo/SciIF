#!/usr/bin/env python3
"""
SciIF API Client

API client for SciIF framework supporting multiple API schemes (boyue/ailab).
Uses AsyncOpenAI responses.create() interface with improved error handling.
"""
from typing import List, Dict, Any, Optional, Union
from openai import AsyncOpenAI, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError
import os
import time
import asyncio
import aiohttp
import ssl

from .config import MODEL_MAP, API_URL, API_SCHEMES, get_model_config


_CLIENT_CACHE: Dict[str, Union[AsyncOpenAI, aiohttp.ClientSession]] = {}

# Global concurrency control: configurable via environment variable (default: 32)
_API_SEMAPHORE: Optional[asyncio.Semaphore] = None

def _get_semaphore() -> asyncio.Semaphore:
    """Get or create global Semaphore (lazy initialization)"""
    global _API_SEMAPHORE
    if _API_SEMAPHORE is None:
        max_concurrent = int(os.getenv("MAX_CONCURRENT", "32"))
        _API_SEMAPHORE = asyncio.Semaphore(max_concurrent)
    return _API_SEMAPHORE


def get_client(model_name: str = "gpt-5", timeout: float = 120.0) -> Union[AsyncOpenAI, aiohttp.ClientSession]:
    """
    Get client (returns different clients based on api_scheme field in model config)
    
    Args:
        model_name: Model name
        timeout: Timeout duration (seconds)
    
    Returns:
        Client instance (OpenAI SDK or aiohttp Session)
    """
    cfg = get_model_config(model_name)
    
    # Get API scheme used by model (prefer model config, otherwise use environment variable, finally default boyue)
    model_api_scheme = cfg.get("api_scheme")
    if not model_api_scheme:
        # If model does not specify scheme, use environment variable or default boyue
        model_api_scheme = os.getenv("API_SCHEME", "boyue")
    
    scheme_config = API_SCHEMES[model_api_scheme]
    # Prefer using model-specific api_url (if configured), otherwise use scheme default
    api_url = cfg.get("api_url", scheme_config["api_url"])
    
    if scheme_config["api_type"] == "openai_sdk":
        # boyue and other schemes: use AsyncOpenAI SDK
        if not cfg.get("api_key"):
            raise ValueError(f"{model_name} missing api_key")
        
        # Cache a client for each (model, timeout) combination (timeout set at client level)
        cache_key = f"{model_name}_{timeout}"
        if cache_key not in _CLIENT_CACHE:
            client = AsyncOpenAI(
                base_url=api_url,
                api_key=cfg["api_key"],
                timeout=timeout,
            )
            _CLIENT_CACHE[cache_key] = client
        return _CLIENT_CACHE[cache_key]
    
    elif scheme_config["api_type"] == "direct_http":
        # ailab scheme: use aiohttp
        cache_key = f"ailab_session_{model_name}"
        if cache_key not in _CLIENT_CACHE:
            # Create SSL context (if needed)
            ssl_context = None
            if not scheme_config.get("verify_ssl", True):
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            session = aiohttp.ClientSession(
                timeout=timeout_obj,
                connector=aiohttp.TCPConnector(ssl=ssl_context)
            )
            _CLIENT_CACHE[cache_key] = session
        return _CLIENT_CACHE[cache_key]
    
    else:
        raise ValueError(f"Unsupported API type: {scheme_config['api_type']}")


def _run(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # In case future extensions call from async context.
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return asyncio.run(coro)


def run_responses_api(
    model: str = "gpt-5",
    input_text: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    timeout: int = 6000,
    max_retries: Optional[int] = None
) -> str:
    """
    Use API interface (selects different calling methods based on current scheme)
    Supports two interfaces: responses.create() and chat.completions.create()
    Supports two schemes: boyue (OpenAI SDK) and ailab (direct HTTP)
    """
    # Ensure input_text is string type
    if input_text is None:
        input_text = ""
    elif not isinstance(input_text, str):
        input_text = str(input_text)
    
    # Validate input_text is not empty (API requires input parameter)
    if not input_text or not input_text.strip():
        raise ValueError("input_text cannot be empty, API requires a valid input parameter")
    
    if max_retries is None:
        max_retries = int(os.getenv("API_MAX_RETRIES", "3"))

    client = get_client(model, timeout=float(timeout))

    # Use model name configured in MODEL_MAP (may differ from passed model parameter)
    cfg = get_model_config(model)
    api_model_name = cfg.get("model", model)  # If model field exists in config, use it; otherwise use passed name
    use_chat_completions = cfg.get("use_chat_completions", False)  # Whether to use chat.completions interface

    async def _call_with_retry():
        last_exception = None
        # Get API scheme used by model
        cfg = get_model_config(model)
        model_api_scheme = cfg.get("api_scheme", os.getenv("API_SCHEME", "boyue"))
        scheme_config = API_SCHEMES[model_api_scheme]
        
        # Detailed API call logs removed, only keep error messages to view overall progress
        
        # Choose different call method according to API scheme type
        if scheme_config["api_type"] == "direct_http":
            # ailab scheme: use aiohttp direct HTTP request (refer to method 2 provided by user)
            # Prefer using model-specific api_url
            api_url = cfg.get("api_url", scheme_config["api_url"])
            api_key = scheme_config.get("api_key", "")
            
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": api_model_name,
                "messages": [{"role": "system", "content": input_text}],  # Note: ailab uses system instead of user
                "max_tokens": 4096,
                "stream": False
            }
            
            # Create SSL context (disable verification)
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            for attempt in range(max_retries):
                start_time = time.perf_counter()
                try:
                    if attempt > 0:
                        print(f"  Retry attempt...")
                        await asyncio.sleep(min(2 ** attempt, 10))  # Exponential backoff
                    
                    # Detailed API request logs removed
                    async with client.post(api_url, json=payload, headers=headers, ssl=ssl_context) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"HTTP error: {error_text[:200]}")
                            raise aiohttp.ClientError(f"{response.status}: {error_text[:200]}")
                        result = await response.json()
                        latency = time.perf_counter() - start_time
                        output_text = (result['choices'][0]['message']['content'] or "").strip()
                        # Detailed success logs removed, only keep error messages
                        return output_text
                except aiohttp.ClientError as e:
                    latency = time.perf_counter() - start_time
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 10)
                        print(f"Request error, retrying after seconds ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise Exception(f"API call failed (aiohttp.ClientError): {str(e)}")
                except asyncio.TimeoutError as e:
                    latency = time.perf_counter() - start_time
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 10)
                        print(f"  Timeout, retrying after seconds ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise Exception(f"API call timeout: {str(e)}")
                except Exception as e:
                    latency = time.perf_counter() - start_time
                    last_exception = e
                    error_type = type(e).__name__
                    print(f"API call failed: {error_type}: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 10)
                        print(f"  {wait_time}seconds before retry ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise Exception(f"API call failed ({error_type}): {str(e)}")
            
            if last_exception:
                raise Exception(f"API call failed, still failed after retries: {str(last_exception)}")
        
        elif scheme_config["api_type"] == "openai_sdk":
            # boyue scheme: use OpenAI SDK
            # Get api_url for error reporting
            api_url = cfg.get("api_url", scheme_config["api_url"])
            for attempt in range(max_retries):
                start_time = time.perf_counter()
                try:
                    if attempt > 0:
                        print(f"  Retry attempt...")
                    if use_chat_completions:
                        # Use chat.completions.create() interface (for Gemini and other models)
                        messages = [{"role": "user", "content": input_text}]
                        params = {"model": api_model_name, "messages": messages}
                        if tools:
                            params["tools"] = tools

                        # Detailed API request logs removed
                        response = await asyncio.wait_for(
                            client.chat.completions.create(**params),
                            timeout=timeout
                        )
                        latency = time.perf_counter() - start_time
                        if response.choices and len(response.choices) > 0:
                            content = response.choices[0].message.content
                            if content is None:
                                print(f"  API returned empty content (None), Model: {api_model_name}")
                                if attempt < max_retries - 1:
                                    wait_time = (attempt + 1) * 2
                                    print(f"  {wait_time}seconds before retry ({attempt + 1}/{max_retries})...")
                                    await asyncio.sleep(wait_time)
                                    continue
                                raise Exception(f"API returned empty content (None), still failed after retries")
                            output_text = content.strip()
                            if not output_text:
                                print(f"  API returned empty string, Model: {api_model_name}")
                                if attempt < max_retries - 1:
                                    wait_time = (attempt + 1) * 2
                                    print(f"  {wait_time}seconds before retry ({attempt + 1}/{max_retries})...")
                                    await asyncio.sleep(wait_time)
                                    continue
                                raise Exception(f"API returned empty string, still failed after retries")
                            # Detailed success logs removed
                            return output_text
                        raise Exception("No content in response")
                    else:
                        # Use responses.create() interface (for GPT and other models)
                        params = {"model": api_model_name, "input": input_text}
                        if tools:
                            params["tools"] = tools

                        # Detailed API request logs removed
                        response = await asyncio.wait_for(
                            client.responses.create(**params),
                            timeout=timeout
                        )
                        latency = time.perf_counter() - start_time
                        output_text = response.output_text.strip()
                        # Detailed success logs removed
                        return output_text
                except AttributeError as e:
                    # OpenAI SDK version does not support current call method
                    if use_chat_completions:
                        error_msg = "API client does not support chat.completions.create(), please check API version"
                    else:
                        error_msg = "API client does not support responses.create(), please check API version"
                    print("Error type: AttributeError")
                    print(f"Error details: {str(e)}")
                    print(f"API URL: {api_url}")
                    print(f"Model: {model}")
                    print(f"Interface used: {'chat.completions' if use_chat_completions else 'responses'}")
                    raise Exception(f"{error_msg}: {e}")
                except RateLimitError as e:
                    latency = time.perf_counter() - start_time
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"Rate limit error, retrying after seconds ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                    print(f"Still encountering rate limit after retries")
                    raise Exception(f"API call failed (RateLimitError): {str(e)}")
                except asyncio.TimeoutError as e:
                    latency = time.perf_counter() - start_time
                    last_exception = e
                    print(f"  Async timeout error (elapsed: timeout limit:)")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"  {wait_time}seconds before retry ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                    print(f"  Still timeout after retries")
                    raise Exception(f"API call timeout (asyncio.TimeoutError): {str(e)}")
                except (APITimeoutError, TimeoutError) as e:
                    latency = time.perf_counter() - start_time
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"Request timeout (latency:), retrying after seconds ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                    print(f"❌ Still timeout after retries")
                    raise Exception(f"API call failed (TimeoutError): {str(e)}")
                except APIConnectionError as e:
                    latency = time.perf_counter() - start_time
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"Connection error (latency: {latency:.2f}s), retrying after seconds ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                    print("Still connection error after retries")
                    raise Exception(f"API call failed (APIConnectionError): {str(e)}")
                except APIStatusError as e:
                    latency = time.perf_counter() - start_time
                    last_exception = e
                    error_type = f"HTTP_{e.status_code}"
                    print("API call failure details:")
                    print(f"   - Error type: {error_type}")
                    print(f"   - HTTP status code: {e.status_code}")
                    print(f"   - Error message: {str(e)}")
                    print(f"   - API URL: {api_url}")
                    print(f"   - Model: {model}")
                    print(f"   - Request latency: {latency:.2f}s")
                    print(f"   - Input length: characters")
                    if hasattr(e, "response"):
                        print(f"   - Response content: {getattr(e.response, 'text', 'N/A')[:200]}...")
                    if e.status_code >= 500 and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"Server error, retrying after seconds ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise Exception(f"API call failed ({error_type}): {str(e)}")
                except Exception as e:
                    latency = time.perf_counter() - start_time
                    last_exception = e
                    error_type = type(e).__name__
                    error_msg = str(e)

                    print("API call failure details:")
                    print(f"   - Error type: {error_type}")
                    print(f"   - Error message: {error_msg}")
                    print(f"   - API URL: {api_url}")
                    print(f"   - Model: {model}")
                    print(f"   - Request latency: {latency:.2f}s")
                    print(f"   - Request parameters: model={model}, tools={tools}, timeout={timeout}")
                    print(f"   - Input length: characters")
                    print(f"   - Attempt count: {attempt + 1}/{max_retries}")

                    if hasattr(e, "response"):
                        print(f"   - Response status: {getattr(e.response, 'status_code', 'N/A')}")
                        print(f"   - Response content: {getattr(e.response, 'text', 'N/A')[:200]}...")

                    if ("timeout" in error_msg.lower() or "timed out" in error_msg.lower()) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"Detected timeout error, retrying after seconds ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue

                    raise Exception(f"API call failed ({error_type}): {error_msg}")
            
            if last_exception:
                raise Exception(f"API call failed, still failed after retries: {str(last_exception)}")

        if last_exception:
            raise Exception(f"API call failed, still failed after retries: {str(last_exception)}")
        raise Exception("API call failed: unknown error")

    return _run(_call_with_retry())
