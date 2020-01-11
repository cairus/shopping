# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Shopping Optimizer, project in Advanced Algorithmics (MTAT.03.238), 2019/2020 Autumn

# ## Code for loading data:

# +
from collections import namedtuple
import numpy as np

# Define the Shop namedtuple outside the loading function to make it available for later use, if necessary.
Shop = namedtuple("Shop", "id x y items prices")

# Loads shops with items from text files.
# Input: a) path to shops file (assumes each row is a shop with X and Y coordinates, separated by whitespace),
# b) path to items file (assumes each row is items for a shop, items separated by whitespace).
# Output: A collection of Shops (e.g. Shop(id=0, x=869, y=696, items=['A', 'J']))
def load_shops(path_to_shops, path_to_items):
    shops = []
    with open(path_to_shops) as file:
        for i,line in enumerate(file.readlines()):
            line = line.strip().split(" ")
            shops.append([i, int(line[0]), int(line[1])])
    with open(path_to_items) as file:
        for i,line in enumerate(file.readlines()):
            line = line.strip().split(" ")
            
            # Add items
            items = [line[j] for j in range(0, len(line), 2)]
            shops[i].append(items)
            # Add prices
            prices = [line[j+1] for j in range(0, len(line), 2)]
            shops[i].append(prices)
                
    shops = [Shop(e[0],e[1],e[2],e[3], e[4]) for e in shops]
    
    return shops

def distance_matrix(shops):
    distances = np.zeros((len(shops), len(shops)))
    for i in range(len(shops)):
        for j in range(len(shops)):
            distances[i][j] = 0 if i==j else euclidean_distance((shops[i].x, shops[i].y), (shops[j].x, shops[j].y))
    return distances


# +
# Load and save the shops and all required items to variables
shops = load_shops("tsp_10.txt", "shops_10_priced.txt")
all_items = get_all_items(shops)

# Print for debugging
print("> Loaded", len(shops), "shops:")
for shop in shops:
    print(">>",shop)
print("> All items:", all_items)

# Create distance matrix
distances = distance_matrix(shops)
#print(distances)

# +
# Generating test data
import random
import sys
import numpy as np

# Create shops with items. Requires list of tuples of positions (x,y).
def create_shops(positions, shop_count=10, item_types=10, max_item_count=30, min_items=1, max_items=6, 
                 min_price=1, max_price=4):
    possible_item_types = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:item_types]
    print("Generating shops. Using", len(possible_item_types), "item types.")
    items = []
    item_prices = []
    
    # First, create some items to shops, randomly between given min and max parameters.
    for i in range(shop_count):
        shop_item_count = random.randint(min_items, max_items)
        shop_items = list(np.random.choice(list(possible_item_types), shop_item_count, replace=False))
        items.append(shop_items)
    
    # Make sure not more than item_count items exist in the shops
    item_count = sum(len(e) for e in items)
    while item_count > max_item_count:
        # Pick one shop randomly and delete an item from its list
        i = random.randint(0, len(items) - 1)
        if len(items[i]) > 1:
            items[i].pop()
        item_count = sum(len(e) for e in items)
    
    # Give prices to items. Currently selects them randomly.
    for i in range(len(items)):
        item_prices.append([random.randint(min_price, max_price) for e in range(len(items[i]))])
    return list(Shop(i, positions[i][0], positions[i][1], items[i], item_prices[i]) for i in range(len(positions)))

# Saves the given shops to a text file. 
def save_shops(shops, fileName=""):
    if fileName == "":
        fileName = "shops_"+str(len(shops))+"_priced.txt"
    with open(fileName, "w+") as file:
        for i, shop in enumerate(shops):
            for i in range(len(shop.items)):
                file.write(str(shop.items[i]) + " " + str(shop.prices[i]) + " ")
            file.write("\n")

# Example usage:
positions = [(e[1], e[2]) for e in shops]
shops_random = create_shops(positions)
print("Created shops are:")
for shop in shops_random:
    print(">", shop)
save_shops(shops_random)
# -

# ## Visualization code

# +
from matplotlib import pyplot as plt
import networkx as nx

# Visualizes a given ordered collection of Shops using matplotlib and networkx
# Install networkx if have not yet done so: pip install networkx
def visualize(shops):
    G = nx.DiGraph()
    for i,shop in enumerate(shops):
        G.add_node(shop.id, pos=(shop.x, shop.y))
        if i < len(shops) - 1:
            G.add_edge(i, i+1)
    pos=nx.get_node_attributes(G,'pos')
    label = {e.id:e.id for e in shops}
    nx.draw(G, pos, labels=label)
    
    legend = []
    for e in shops:
        lgn = "Shop " + str(e.id) + ": " + "".join([str(e.items[i]) + " " + str(e.prices[i]) + ", " for i in range(len(e.items))])
        #lgn = "Shop " + str(e.id) + ": " + "".join([e.items[i]] + "" for i in range(len(e.items)))
        legend.append(lgn)

    plt.legend(legend, bbox_to_anchor=(1, 1))
    plt.show()

# Example usage:
visualize(shops_random)

# -

# ## Helper code

# +
import math
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# Returns all the unique items from the given list of shops.
def get_all_items(shops):
    items = set()
    for shop in shops:
        items.update(shop.items)
    return items
# -

# ## Algorithm




