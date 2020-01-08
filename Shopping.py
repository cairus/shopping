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

# ### Shopping Optimizer, project in Advanced Algorithmics (MTAT.03.238), 2019/2020 Autumn

# +
from collections import namedtuple

# Input: path to shops file (assumes each row is a shop with X and Y coordinates, separated by whitespace),
# path to items file (assumes each row is items for a shop, items separated by whitespace).
# Output: A collection of Shops (e.g. Shop(id=0, x=869, y=696, items=['A', 'J']))
def load_shops(path_to_shops, path_to_items):
    Shop = namedtuple("Shop", "id x y items")
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

shops = load_shops("tsp_10.txt", "shops_10.txt")
print("> Loaded", len(shops), "shops:")
for shop in shops:
    print(">>",shop)

    
#Test for writing a comment
def function():
    print("The new fun works pew pew")

# +
#Also a test for new cell.
