import re
from typing import List, Dict, Tuple

def extract_numbers(text: str) -> List[str]:
    """
    Extracts all numeric sequences from a string.
    Extracted as strings to handle formats like 123,456 or 12.34
    and to allow simple string matching.
    """
    if not text:
        return []
    # Regex for finding numbers (integers, floats, with or without commas)
    # Matches patterns like: 123, 1,234.56, 12.34, etc.
    raw_numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
    return [num.replace(',', '') for num in raw_numbers]

def verify_grounding(final_answer: str, steps: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Checks every number in the final answer against the raw tool results.
    Returns (is_grounded, list_of_unsupported_numbers).
    """
    # 1. Extract all numbers from the final answer
    answer_numbers = extract_numbers(final_answer)
    if not answer_numbers:
        return True, [] # Nothing to ground
        
    # 2. Extract all numbers from all tool results
    tool_text = ""
    for step in steps:
        if step.get("type") == "tool_result":
            tool_text += str(step.get("result", "")) + " "
            
    tool_numbers = extract_numbers(tool_text)
    
    # 3. Check for presence
    # Note: This is a strict check. If the LLM does math (e.g., calculates a percentage),
    # the guard will flag it unless that exact percentage appeared in the tool output.
    # This is intended behavior for a strict grounding guard.
    unsupported = []
    tool_numbers_set = set(tool_numbers)
    
    for num in answer_numbers:
        # Check if the number or its exact representation exists in the tool output
        # e.g. the answer has 1234 but tool output has 1234.0, so a basic float
        # conversion check catches that instead of a strict string match.
        found = False
        if num in tool_numbers_set:
            found = True
        else:
            # Try float comparison for flexibility (e.g. 10.0 vs 10)
            try:
                num_float = float(num)
                for tn in tool_numbers:
                    try:
                        if float(tn) == num_float:
                            found = True
                            break
                    except ValueError:
                        pass
            except ValueError:
                pass
                
        if not found:
            unsupported.append(num)
            
    is_grounded = len(unsupported) == 0
    return is_grounded, unsupported

if __name__ == "__main__":
    # Simple test
    ans = "There are 99,441 orders in the dataset, and 5% are canceled."
    steps = [{"type": "tool_result", "result": "total_orders\n99441"}]
    is_grounded, unsupp = verify_grounding(ans, steps)
    print(f"Grounded: {is_grounded}")
    print(f"Unsupported numbers: {unsupp}")
