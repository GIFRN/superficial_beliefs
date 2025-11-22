#!/usr/bin/env python3
"""
Test script to experiment with GPT-5 Responses API response extraction.
This helps us figure out the correct way to get model outputs before modifying the actual code.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package is not installed")
    sys.exit(1)


def print_separator(title=""):
    print("\n" + "="*80)
    if title:
        print(f" {title}")
        print("="*80)


def inspect_object(obj, name="object", max_depth=3, current_depth=0, visited=None):
    """Recursively inspect an object and print its structure."""
    if visited is None:
        visited = set()
    
    indent = "  " * current_depth
    obj_id = id(obj)
    
    # Avoid infinite recursion
    if obj_id in visited or current_depth >= max_depth:
        return
    visited.add(obj_id)
    
    print(f"{indent}{name} (type: {type(obj).__name__})")
    
    # Print object attributes
    if hasattr(obj, "__dict__"):
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            try:
                attr_value = getattr(obj, attr_name)
                # Skip methods
                if callable(attr_value):
                    continue
                
                # For simple types, print the value
                if isinstance(attr_value, (str, int, float, bool, type(None))):
                    print(f"{indent}  {attr_name}: {repr(attr_value)}")
                # For lists, show length and inspect first item
                elif isinstance(attr_value, list):
                    print(f"{indent}  {attr_name}: list[{len(attr_value)}]")
                    if attr_value and current_depth < max_depth - 1:
                        inspect_object(attr_value[0], f"{attr_name}[0]", max_depth, current_depth + 1, visited)
                # For dicts, show keys
                elif isinstance(attr_value, dict):
                    print(f"{indent}  {attr_name}: dict with keys: {list(attr_value.keys())}")
                # For other objects, recurse
                else:
                    if current_depth < max_depth - 1:
                        inspect_object(attr_value, attr_name, max_depth, current_depth + 1, visited)
            except Exception as e:
                print(f"{indent}  {attr_name}: <error accessing: {e}>")


def test_extraction_method(response, method_name, extractor_func):
    """Test a specific method of extracting text from the response."""
    print(f"\n  Method: {method_name}")
    try:
        result = extractor_func(response)
        if result:
            print(f"    ✓ SUCCESS: Got {len(result)} characters")
            print(f"    First 100 chars: {repr(result[:100])}")
            return result
        else:
            print(f"    ✗ FAILED: Got empty or None result: {repr(result)}")
            return None
    except Exception as e:
        print(f"    ✗ ERROR: {type(e).__name__}: {e}")
        return None


def main():
    print_separator("GPT-5 RESPONSES API TEST")
    
    # Initialize client
    client = OpenAI()
    
    # Check if responses API is available
    if not hasattr(client, "responses"):
        print("ERROR: Client does not have 'responses' attribute!")
        print("Your OpenAI library version may not support the Responses API.")
        print("\nAvailable attributes:")
        for attr in dir(client):
            if not attr.startswith("_"):
                print(f"  - {attr}")
        sys.exit(1)
    
    print("✓ Client has 'responses' attribute")
    
    # Test with a simple prompt
    print_separator("MAKING TEST REQUEST")
    
    test_messages = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Say 'Hello, World!' and nothing else."}],
        }
    ]
    
    print("Request parameters:")
    print(f"  Model: gpt-5-mini")
    print(f"  Input: {test_messages}")
    
    try:
        # Try with different token limits
        print("\nAttempt 1: With temperature and max_output_tokens=50")
        try:
            response = client.responses.create(
                model="gpt-5-mini",
                input=test_messages,
                temperature=0.7,
                max_output_tokens=50,
            )
            print("  ✓ Success with all parameters")
        except Exception as e1:
            print(f"  ✗ Failed: {e1}")
            print("\nAttempt 2: Without temperature, max_output_tokens=256")
            try:
                response = client.responses.create(
                    model="gpt-5-mini",
                    input=test_messages,
                    max_output_tokens=256,
                )
                print("  ✓ Success without temperature, higher token limit")
            except Exception as e2:
                print(f"  ✗ Failed: {e2}")
                print("\nAttempt 3: Without temperature, max_output_tokens=1000")
                try:
                    response = client.responses.create(
                        model="gpt-5-mini",
                        input=test_messages,
                        max_output_tokens=1000,
                    )
                    print("  ✓ Success with higher token limit")
                except Exception as e3:
                    print(f"  ✗ Failed: {e3}")
                    print("\nAttempt 4: Only model and input (no token limit)")
                    response = client.responses.create(
                        model="gpt-5-mini",
                        input=test_messages,
                    )
                    print("  ✓ Success with minimal parameters")
        
        print("\n✓ API call successful!")
        
        # Check response status
        if hasattr(response, 'status'):
            print(f"\n⚠️  Response status: {response.status}")
            if response.status == 'incomplete':
                print(f"   Incomplete reason: {getattr(response.incomplete_details, 'reason', 'unknown')}")
        
    except Exception as e:
        print(f"\n✗ All API call attempts failed!")
        print(f"Final error: {type(e).__name__}: {e}")
        sys.exit(1)
    
    # Inspect the response object
    print_separator("RESPONSE OBJECT STRUCTURE")
    print("\nFull object inspection:")
    inspect_object(response, "response", max_depth=4)
    
    # Try different extraction methods
    print_separator("TESTING EXTRACTION METHODS")
    
    results = {}
    
    # Method 1: Direct output_text attribute
    results['output_text'] = test_extraction_method(
        response,
        "response.output_text",
        lambda r: getattr(r, "output_text", None)
    )
    
    # Method 2: output attribute with iteration
    results['output_iteration'] = test_extraction_method(
        response,
        "response.output (iterate and extract text)",
        lambda r: (
            "".join(
                content.text
                for item in getattr(r, "output", [])
                if hasattr(item, "content")
                for content in item.content
                if hasattr(content, "text")
            ) if hasattr(r, "output") else None
        )
    )
    
    # Method 3: choices format
    results['choices'] = test_extraction_method(
        response,
        "response.choices[0].message.content",
        lambda r: (
            r.choices[0].message.content
            if hasattr(r, "choices") and r.choices and 
               hasattr(r.choices[0], "message") and 
               hasattr(r.choices[0].message, "content")
            else None
        )
    )
    
    # Method 4: Direct text attribute
    results['text'] = test_extraction_method(
        response,
        "response.text",
        lambda r: getattr(r, "text", None)
    )
    
    # Method 5: content attribute
    results['content'] = test_extraction_method(
        response,
        "response.content",
        lambda r: getattr(r, "content", None)
    )
    
    # Method 6: output[0].content[0].text
    results['output_nested'] = test_extraction_method(
        response,
        "response.output[0].content[0].text",
        lambda r: (
            r.output[0].content[0].text
            if hasattr(r, "output") and r.output and
               hasattr(r.output[0], "content") and r.output[0].content and
               hasattr(r.output[0].content[0], "text")
            else None
        )
    )
    
    # Summary
    print_separator("SUMMARY")
    
    print("\nSUPPORTED API PARAMETERS:")
    print("  The API call succeeded. Check the 'MAKING TEST REQUEST' section above")
    print("  to see which parameter combination worked.")
    
    successful_methods = [name for name, result in results.items() if result]
    
    if successful_methods:
        print(f"\n✓ {len(successful_methods)} method(s) successfully extracted text:")
        for method in successful_methods:
            print(f"  - {method}")
        
        print("\n" + "="*80)
        print("RECOMMENDED IMPLEMENTATION:")
        print("="*80)
        
        # Show the first successful method's implementation
        first_success = successful_methods[0]
        if first_success == 'output_text':
            print("""
result = getattr(response, "output_text", None)
if not result:
    raise RuntimeError("Response returned empty or no output_text")
return result
""")
        elif first_success == 'output_iteration':
            print("""
if hasattr(response, "output") and response.output:
    result = ""
    for item in response.output:
        if hasattr(item, "content"):
            for content in item.content:
                if hasattr(content, "text"):
                    result += content.text
    if not result:
        raise RuntimeError("Response output exists but contains no text")
    return result
raise RuntimeError("Response has no output attribute")
""")
        elif first_success == 'output_nested':
            print("""
if hasattr(response, "output") and response.output:
    if hasattr(response.output[0], "content") and response.output[0].content:
        text = response.output[0].content[0].text
        if not text:
            raise RuntimeError("Response returned empty text")
        return text
raise RuntimeError("Response structure unexpected")
""")
    else:
        print("\n✗ No methods successfully extracted text!")
        print("\nThis suggests the response format is different than expected.")
        print("Check the 'RESPONSE OBJECT STRUCTURE' section above for details.")


if __name__ == "__main__":
    main()

