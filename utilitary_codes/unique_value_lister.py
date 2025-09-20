import json
from pathlib import Path
from collections import defaultdict

def extract_unique_values():
    # Initialize storage
    unique_data = {
        'developers': set(),
        'publishers': set(),
        'categories': set(),
        'genres': set(),
        'user_tags': set()
    }

    # Path setup
    input_dir = Path('filtered_datasets')
    output_dir = Path('gamedata_value_lists')
    output_dir.mkdir(exist_ok=True)

    # Process JSON file
    with open(input_dir / 'normalized_filtered_dataset.json', 'r', encoding='utf-8') as f:
        games_data = json.load(f)
        
        games_with_tags = 0  # Debug counter
        
        for game_id, game in games_data.items():
            # Extract developers
            if 'developers' in game:
                unique_data['developers'].update(game['developers'])
            
            # Extract publishers
            if 'publishers' in game:
                unique_data['publishers'].update(game['publishers'])
            
            # Extract categories
            if 'categories' in game:
                for category in game['categories']:
                    if isinstance(category, dict) and 'description' in category:
                        unique_data['categories'].add(category['description'])
            
            # Extract genres
            if 'genres' in game:
                for genre in game['genres']:
                    if isinstance(genre, dict) and 'description' in genre:
                        unique_data['genres'].add(genre['description'])                            
            
            # Extract user tags
            if 'user_tags' in game:
                games_with_tags += 1
                if isinstance(game['user_tags'], dict):
                    unique_data['user_tags'].update(game['user_tags'].keys())
                else:
                    print(f"Unexpected user_tags format in game {game_id}: {type(game['user_tags'])}")

        print(f"Found {games_with_tags} games with user_tags")

    # Convert sets to sorted lists and save
    for key, values in unique_data.items():
        sorted_values = sorted(list(values))
        output_file = output_dir / f'{key}_list.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_values, f, indent=2, ensure_ascii=False)
        
        print(f'Saved {len(sorted_values)} {key} to {output_file}')

if __name__ == '__main__':
    extract_unique_values()