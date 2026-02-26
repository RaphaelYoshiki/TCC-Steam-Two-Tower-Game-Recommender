import json
import pickle
from pathlib import Path
from collections import defaultdict

REQUIRED_FIELDS = {
    'name',
    'developers',
    'publishers',
    'price_overview',
    'platforms',
    'categories',
    'genres',
    'release_date',
    'review_score',
    'user_tags'
}

def is_valid_entry(entry):
    # Check if entry has all required fields
    if not all(field in entry for field in REQUIRED_FIELDS):
        return False
    
    # Check if content_descriptors.ids exists and contains 1
    content_descriptors = entry.get('content_descriptors', {})
    if isinstance(content_descriptors, dict):
        ids = content_descriptors.get('ids', [])
        if 1 in ids:
            return False
        
    if not entry['name'] or not isinstance(entry['name'], str):
        return False
        
    if not entry['developers'] or not isinstance(entry['developers'], list):
        return False
        
    if not entry['publishers'] or not isinstance(entry['publishers'], list):
        return False
        
    if not entry['price_overview'].get('final') or not isinstance(entry['price_overview']['final'], int):
        return False
        
    if not any(entry['platforms'].values()):  # At least one platform must be True
        return False
        
    if not entry['categories'] or not isinstance(entry['categories'], list):
        return False
        
    if not entry['genres'] or not isinstance(entry['genres'], list):
        return False
        
    if not entry['release_date'].get('date') or not isinstance(entry['release_date']['date'], str):
        return False
        
    if not entry['review_score'] or not isinstance(entry['review_score'], str):
        return False
        
    if not entry['user_tags'] or not isinstance(entry['user_tags'], dict) or not entry['user_tags']:
        return False
    
    return True

def filter_and_save_dataset(input_path):

    # Filter out entries where content_descriptors.ids contains 1
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    fixed_dataset = fix_unicode_escapes(dataset)
    
    filtered_data = {
        key: value for key, value in fixed_dataset.items() 
        if isinstance(value, dict) and is_valid_entry(value)
    }    
    
    with open('filtered_datasets/filtered_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

def fix_unicode_escapes(obj):
    if isinstance(obj, str):
        # Handle both `\uXXXX` and raw Unicode characters safely
        try:
            # Try decoding as raw Unicode first (for already correct strings)
            return obj.encode('raw_unicode_escape').decode('unicode-escape')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return obj  # Fallback if encoding fails
    elif isinstance(obj, dict):
        return {key: fix_unicode_escapes(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [fix_unicode_escapes(item) for item in obj]
    else:
        return obj

if __name__ == "__main__":
    filter_and_save_dataset("json_converted_dataset/dataset_with_reviews.json")