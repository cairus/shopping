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
    items = set()
    for shop in shops:
        items.update(shop.items)
    return tuple(shops), items

def load_distances(shops):
    distances = np.zeros((len(shops), len(shops)))
    for i in range(len(shops)):
        for j in range(len(shops)):
            distances[i][j] = 0 if i==j else euclidean_distance((shops[i].x, shops[i].y), (shops[j].x, shops[j].y))
    return distances


# +
# Load and save the shops and all required items to variables, print them
shops, all_items = load_shops("tsp_10.txt", "shops_10.txt")
print("> Loaded", len(shops), "shops:")
for shop in shops:
    print(">>",shop)
print("> All items:", all_items)

# Distance matrix
distances = load_distances(shops)
#print(distances)
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
    lbl = {e.id:e.id for e in shops}
    nx.draw(G, pos, labels=lbl)
    plt.show()

# Example usage:
visualize(shops)

# -

# ## Helper code

import math
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# ## Algorithm (initial idea)

# +
from scipy.spatial import distance
import sys
#input:
    #items - list of items that need to be purchased
    #shops - list of shops with shop ID, x,y-coordinates and list of items it has
        #eg Shop(id=0, x=869, y=696, items=['A', 'J'])
    #start - x and y coordinates for the starting point of the shopping trip
    #distance matrix - how far shops are from each other

#output:
    #optimal list of shops to visit
    
def shopping(item_list, shop_list, start, dists): 
    opt_shops = [] #list of shops to visit
    purchase = item_list[:] #[e for e in item_list] #items that yet need to be purchased
    remaining_shops = [a for a in shop_list] #shops that we haven't visited yet
    current = start #the location we are currently at (start [x,y] or a current shop (Shop))
    
    while len(purchase) != 0: #some items still need to be purchased
        
        best = sys.maxsize #just a very large number to start comparing shops
        best_shop = ""
        for shop in remaining_shops: #find optimal next shop
            #if current is list type, we must find the distance from there to a shop
            #if current is not of list type, we are already at some shop
            if type(current) == list: #we are not at any specific shop yet
                dist = distance.euclidean(current, [shop.x, shop.y]) #distance from current
            else: #to move between shops, use precalculated distances
                dist = distances[current.id][shop.id]
            common = len(set(shop.items).intersection(set(purchase))) #how many needed items it has
            if common == 0: #the shop has no necessary items
                remaining_shops.remove(shop) #so we don't need the shop
            else:  
                efficiency = dist / common
                if efficiency < best: #if we found a more efficient shop, use that as comparison
                    best = efficiency
                    best_shop = shop
                    
        #update visited shops, needed items and current location
        opt_shops.append(best_shop) #add the shop we chose to the final list
        purchase = list(set(purchase).difference(set(best_shop.items))) #remove the bought items from list
        remaining_shops.remove(best_shop) #no need to revisit the shop
        #current = (best_shop.x, best_shop.y) #move to the chosen shop
        current = best_shop
        
    return opt_shops
 
#test it with some set of items, starting at point 0,0
shopping(["A","E","G","H"], shops, [0,0], distances)
# -

# ## Algorithm - new approach phase 1

# +
from scipy.spatial import distance
import sys

#input:
    #items - list of items that need to be purchased
    #shops - list of shops with shop ID, x,y-coordinates and list of items it has
        #eg Shop(id=0, x=869, y=696, items=['A', 'J'])
    #start - x and y coordinates for the starting point of the shopping trip
    #distance matrix - how far shops are from each other
    #distance cost - how much it costs to walk/drive/etc a distance

#phase 1 output:
    #possible list of shops to visit with repeating items and prices
        #if 2 different shops have item A, we still need to choose which shop to buy it from

#phase 2 output
    #list of shops, where shop.items includes only the items that we bought there

#Shop(id, x, y, items, price)
    
def shopping(item_list, shop_list, start, dists, dist_cost): 
    ###
    #PHASE 1
    ###
    opt_shops = [] #list of shops to visit
    purchase = item_list[:] #[e for e in item_list] #items that yet need to be purchased
    remaining_shops = [a for a in shop_list] #shops that we haven't visited yet
    current = start #the location we are currently at (start [x,y] or a current shop (Shop))
    
    while len(purchase) != 0: #some items still need to be purchased
        
        best = sys.maxsize #just a very large number to start comparing shops
        best_shop = ""
        for shop in remaining_shops: #find optimal next shop
            #if current is list type, we must find the distance from there to a shop
            #if current is not of list type, we are already at some shop
            if type(current) == list: #we are not at any specific shop yet
                dist = distance.euclidean(current, [shop.x, shop.y]) #distance from current
            else: #to move between shops, use precalculated distances
                dist = distances[current.id][shop.id]
            
            common = len(set(shop.items).intersection(set(purchase))) #how many needed items it has
            #TODO: might not be the most elegant solution
            #finds indexes of elements chosen to "common" and finds the corresponding prices
            prices = [shop.prices[ix] for ix in [shop.items.index(el) for el in common]]
            if common == 0: #the shop has no necessary items
                remaining_shops.remove(shop) #so we don't need the shop
            else:  
                #efficiency = dist / common 
                efficiency = dist_cost * dist * (sum(prices)/common)
                if efficiency < best: #if we found a more efficient shop, use that as comparison
                    best = efficiency
                    best_shop = shop
                    
        #update visited shops, needed items and current location
        opt_shops.append(best_shop) #add the shop we chose to the final list
        purchase = list(set(purchase).difference(set(best_shop.items))) #remove the bought items from list
        remaining_shops.remove(best_shop) #no need to revisit the shop
        current = best_shop
        
    ###
    #PHASE 2
    ###
    #we have an initial list of shops, where we can get all items we need
    #we don't want to buy multiple copies of the same item
    #and items are sold at different prices in different shops
    #in phase 2 we find the cheapest option for each item
    #in the end we get a list of shops, where only those items are listed, that we bought there 
    #first, find all repeating elements
    all_items = [] #all items we selected from shops
    for shop in opt_shops:
        all_items.extend(shop.items)
        
    bought = opt_shops[:] #shows which items we buy from each shop
        
    #find repeating items
    for item in all_items:
        if all_items.count(item) > 1: #we could but that item from several shops
            cheapest = sys.maxsize
            cheapest_shop = "" #where item was cheapest
            #find which shop sells it cheapest
            for shop in opt_shops:
                item_price = shop.prices[shop.items.index(item)]
                if item_price < cheapest:
                    cheapest = item_price 
                    
        for shop in bought: #we don't buy that item from other shops
            if shop.id != cheapest_shop.id:
                shop.items.remove(item) #remove the item
                shop.prices.remove(shop.prices[shop.items.index(item)]) #remove its price

    
    #returns list of shops we need to visit to get all items 
    return opt_shops, bought
 
#test it with some set of items, starting at point 0,0 and distance cost 1
shopping([["A","D","E","G","H"], [1,2,3,4,5], shops, [0,0], distances, 1)
# -


