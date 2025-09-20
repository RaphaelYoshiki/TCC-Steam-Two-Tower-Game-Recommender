import pickle
import json
import os

def json_to_pickle(json_filepath, pickle_filepath=None):
    """
    Converts a pickle file to a JSON file.
    
    Args:
        pickle_filepath (str): The path to the input pickle file.
        json_filepath (str, optional): The path to the output JSON file. 
            If None, it defaults to the pickle filename with a .json extension.
    """
    if pickle_filepath is None:
      # Replace the extension with .json
      pickle_filepath = os.path.splitext(json_filepath)[0] + '.p'
    
    try:
        with open(json_filepath, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: json file not found at '{json_filepath}'")
        return
    except Exception as e:
         print(f"An error occurred while reading the json file: {e}")
         return

    try:
        with open(pickle_filepath, 'wb') as file:
            pickle.dump(data, file)
        print(f"Successfully converted '{json_filepath}' to '{pickle_filepath}'")
    except Exception as e:
        print(f"An error occurred while writing the pickle file: {e}")
        return
    
def pickle_to_json(pickle_filepath, json_filepath=None):
    """
    Converts a pickle file to a JSON file.
    
    Args:
        pickle_filepath (str): The path to the input pickle file.
        json_filepath (str, optional): The path to the output JSON file. 
            If None, it defaults to the pickle filename with a .json extension.
    """
    if json_filepath is None:
      # Replace the extension with .json
      json_filepath = os.path.splitext(pickle_filepath)[0] + '.json'
    
    try:
        with open(pickle_filepath, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Pickle file not found at '{pickle_filepath}'")
        return
    except Exception as e:
         print(f"An error occurred while reading the pickle file: {e}")
         return

    try:
        with open(json_filepath, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Successfully converted '{pickle_filepath}' to '{json_filepath}'")
    except Exception as e:
        print(f"An error occurred while writing the JSON file: {e}")
        return

# Example usage:
pickle_file = 'checkpoints/apps_dict-ckpt-fin.p'
json_file = 'json_converted_dataset/dataset.json'
pickle_to_json(pickle_file, json_file) # You can omit json_file to generate it automatically
#json_to_pickle(json_file, pickle_file)
#pickle_to_json('filtered_games/final_dataset.p', 'json_converted_dataset/final_dataset.json')