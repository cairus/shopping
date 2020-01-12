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
def visualize(shops, start):
    G = nx.DiGraph()
    
    # Create nodes:
    G.add_node("Start", pos=start)
    for i,shop in enumerate(shops):
        G.add_node(shop.id, pos=(shop.x, shop.y))
        
    # Create edges:
    last_node = "Start"
    for i in range(len(shops)):
        if len(shops[i].items) != 0:
            G.add_edge(last_node, shops[i].id)
            last_node = shops[i].id
    pos=nx.get_node_attributes(G,'pos')
    label = {e.id:e.id for e in shops}
    label["Start"] = "S"
    nx.draw(G, pos, labels=label)
    
    legend = []
    for e in shops:
        if len(e.items) == 0:
            continue
        lgn = "Shop " + str(e.id) + ": " + "".join([str(e.items[i]) + " " + str(e.prices[i]) + ", " for i in range(len(e.items))])
        legend.append(lgn)

    plt.legend(legend, bbox_to_anchor=(1, 1))
    plt.show()

# Example usage:
visualize(shops_random, [0,0])

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
    return list(items)


# -

# ## Algorithm (initial idea) 
# - [Taavi]: buggy but since not used, not going to update

# +
#from scipy.spatial import distance
import sys
#input:
    #items - list of items that need to be purchased
    #shops - list of shops with shop ID, x,y-coordinates and list of items it has
        #eg Shop(id=0, x=869, y=696, items=['A', 'J'])
    #start - x and y coordinates for the starting point of the shopping trip
    #distance matrix - how far shops are from each other

#output:
    #optimal list of shops to visit
    
def shopping_initial(item_list, shop_list, start, dists): 
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
                dist = euclidean_distance(current, [shop.x, shop.y]) #distance from current
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
shopping_initial(["A","E","G","H"], shops, [0,0], distances)
# -

# ## Algorithm - new approach

# +
#from scipy.spatial import distance
import sys
import copy

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
    current = start #the location we are currently at (start [x,y] or a current shop (Shop))
    blacklist = [] #Shops to ignore 
    
    while len(purchase) != 0: #some items still need to be purchased
        best = sys.maxsize #just a very large number to start comparing shops
        best_shop = None
        for shop in shop_list: #find optimal next shop
            if shop in blacklist: #skip if its already visited or has no still needed items
                continue
            #if current is list type, we must find the distance from there to a shop
            #if current is not of list type, we are already at some shop
            if current is start: #we are not at any specific shop yet
                dist = euclidean_distance(current, [shop.x, shop.y]) #distance from current
                current = shop
            else: #to move between shops, use precalculated distances
                dist = distances[current.id][shop.id]
            
            common = set(shop.items).intersection(set(purchase)) #needed items it has
            #TODO: might not be the most elegant solution
            #finds indexes of elements chosen to "common" and finds the corresponding prices
            prices = [shop.prices[ix] for ix in [shop.items.index(el) for el in common]]

            if len(common) == 0: #the shop has no necessary items
                blacklist.append(shop) #so we don't need the shop
            else:  
                #efficiency = dist / common 
                efficiency = dist_cost * dist * (sum(prices)/len(common))
                if efficiency < best: #if we found a more efficient shop, use that as comparison
                    best = efficiency
                    best_shop = shop
                    
        #update visited shops, needed items and current location
        opt_shops.append(best_shop) #add the shop we chose to the final list
        purchase = list(set(purchase).difference(set(best_shop.items))) #remove the bought items from list
        blacklist.append(best_shop)
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
    #bought = opt_shops[:] #shows which items we buy from each shop
    # Need a deep copy for bought:
    bought = copy.deepcopy(opt_shops)
    
    #find repeating items
    for item in all_items:
        cheapest_shop = None
        if all_items.count(item) > 1: #we could buy that item from several shops
            cheapest = sys.maxsize
            #where item was cheapest
            #find which shop sells it cheapest
            for shop in opt_shops:
                if item not in shop.items:
                    continue
                item_price = shop.prices[shop.items.index(item)]
                if item_price < cheapest:
                    cheapest = item_price 
                    cheapest_shop = shop
        for shop in bought: #we don't buy that item from other shops
            if cheapest_shop is not None and shop.id != cheapest_shop.id:
                if item not in shop.items:
                    continue
                shop.prices.remove(shop.prices[shop.items.index(item)]) #remove its price
                shop.items.remove(item) #remove the item

    
    ###
    #PHASE 3
    ###
    #remove the shops we didn't but anything from and perform TSP on them
    #currently TSP is done by simple nearest neighbour
    #distances come from the previously calculated matrix
    final_shops = []
    for shop in bought:
        if len(shop.items) != 0: #if we bought at least one item from that shop, we include it
            final_shops.append(shop)
            
    #all shops are list final_shops. we start adding them to ordered_shops, based on their nearest unvisited shop
    ordered_shops = []
    #find the closest shop to the starting point
    closest = sys.maxsize #big number to start comparing
    first_shop = ""
    for shop in final_shops:
        dist = euclidean_distance(start, [shop.x, shop.y])
        if dist < closest:
            closest = dist
            first_shop = shop
    ordered_shops.append(first_shop)
    i = 0 #the shop we are looking at
    while len(ordered_shops) < len(final_shops): #we mustn't leave out any shops so lists have same length
        #find the next closest unvisited shop. if the closest one is visited, choose the second closest etc
        curr_shop = ordered_shops[i]
        #found = False #we haven't yet found the next closest shop
        next_shop = None #where we want to go to next
        best_dist = sys.maxsize
        for shop in final_shops:
            if shop not in ordered_shops: #we haven't been to that shop yet
                distance = dists[curr_shop.id][shop.id] #dist from current to that shop
                if distance < best_dist:
                    best_dist = distance
                    next_shop = shop
                    
        ordered_shops.append(next_shop)      
    
    #returns list of shops we need to visit to get all items 
    return opt_shops, bought, ordered_shops

#test it with some set of items, starting at point 0,0 and distance cost 0.005
shopping(get_all_items(shops_random), shops_random, [0,0], distances, 0.005)
# +
# Testing with random data. The shopping(..) function returns two lists: shops after Phase 1 and after Phase 2
# If there is a shop with no items in Phase 2, then that shop is actually redundant and Phase 2 is useful.
shops_random = create_shops(positions)
print("-----------")
print("Trying to buy:", get_all_items(shops_random))
start = [0, 0]
start_shops, shops, ordered = shopping(get_all_items(shops_random), shops_random, start, distances, 0.005)

# Print results of Phase 1
print("Results of Phase 1:")
for shop in start_shops:
    print(shop)
    
# Print results of Phase 2
print("-----------")
print("Results of Phase 2:")
for shop in shops:
    print(shop)
    
# Print results of Phase 3
print("-----------")
print("Results of Phase 3:")
for shop in ordered:
    print(shop)

# Visualize
visualize(shops, start)
# -

