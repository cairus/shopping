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

# Define the Shop namedtuple outside the loading function to make it available for later use, if necessary.
Shop = namedtuple("Shop", "id x y items")

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
            shops[i].append([e for e in line])
    shops = [Shop(e[0],e[1],e[2],e[3]) for e in shops]
    return tuple(shops)



# -

# Load and save the shops to a variable, print them
shops = load_shops("tsp_10.txt", "shops_10.txt")
print("> Loaded", len(shops), "shops:")
for shop in shops:
    print(">>",shop)


# ## Visualization code

# +
from matplotlib import pyplot as plt
import networkx as nx

# Visualizes a given ordered collection of Shops using matplotlib and networkx
def visualize(shops):
    G = nx.DiGraph()
    for i,shop in enumerate(shops):
        G.add_node(shop.id, pos=(shop.x, shop.y))
        if i<len(shops) - 1:
            G.add_edge(i, i+1)
    pos=nx.get_node_attributes(G,'pos')
    lbl = {e.id:e.id for e in shops}
    nx.draw(G, pos, labels=lbl)
    plt.show()

# Example usage:
visualize(shops)

# -


