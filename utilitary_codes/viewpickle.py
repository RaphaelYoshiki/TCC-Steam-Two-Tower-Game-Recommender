import pickle
from pprint import pformat  # pformat returns the formatted string instead of printing

#filepath = 'normalized_filtered_games/overwhelmingly_positive_normalized.p'
filepath = 'filtered_games/overwhelmingly_positive.p'

with open(filepath, 'rb') as f:
    data = pickle.load(f)

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(pformat(data))  # Better formatting for nested structures