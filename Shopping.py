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

# ## Main variables
# - *shops* - collection of shops (Shop(id, x, y, items, prices))
# - *all_items* - collection of all items required from the shops
# - *distances* - distance matrix of *shops*

# ## General functions

# +
import math
from collections import namedtuple


# Define the Shop namedtuple that is used to represent shops in the code.
Shop = namedtuple("Shop", "id x y items prices")

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# Calculated and returns the distance matrix of given shops (as a 2d numpy array)
def distance_matrix(shops):
    distances = np.zeros((len(shops), len(shops)))
    for i in range(len(shops)):
        for j in range(len(shops)):
            distances[i][j] = 0 if i==j else euclidean_distance((shops[i].x, shops[i].y), (shops[j].x, shops[j].y))
    return distances


# Returns all the unique items from the given collection of shops
def get_all_items(shops):
    items = set()
    for shop in shops:
        items.update(shop.items)
    return list(items)


# Returns the total distance travelled between the shops in given order. Also uses start and end points, if either is given. 
def total_distance(shops, distance_matrix, start=[0,0], end=None):
    dist = sum(distance_matrix[shops[i].id][shops[i+1].id] for i in range(len(shops)-1))
    if start is not None:
        start_dist = euclidean_distance(start, (shops[0].x, shops[0].y))
        dist += start_dist
    if end is not None:
        last_index = len(shops) - 1
        end_dist = euclidean_distance(end, (shops[last_index].x, shops[last_index].y))
        dist += end_dist
    return dist


# Returns the total cost of items in the given shop. If there are duplicate items, then pick_first must be True, 
# otherwise duplicate items' prices are also added.
def total_item_cost(shops, pick_first=False):
    if not pick_first:
        return sum(sum(shop.prices) for shop in shops)
    else:
        items = get_all_items(shops)
        total_sum = 0
        for item in items:
            for shop in shops:
                if item in shop.items:
                    total_sum += shop.prices[shop.items.index(item)]
                    break
        return total_sum



# -

# ## Code for loading, saving data

# +
import numpy as np


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

# Saves the given shops to a text file. 
def save_shops(shops, fileName=""):
    if fileName == "":
        fileName = "shops_"+str(len(shops))+"_priced.txt"
    with open(fileName, "w+") as file:
        for i, shop in enumerate(shops):
            for i in range(len(shop.items)):
                file.write(str(shop.items[i]) + " " + str(shop.prices[i]) + " ")
            file.write("\n")


# +
# Load shops from files
shops = load_shops("tsp_10.txt", "shops_10_priced.txt")

# Find the set of all required items
all_items = get_all_items(shops)

# Create distance matrix
distances = distance_matrix(shops)

# Print for debugging
print("> Loaded", len(shops), "shops:")
for shop in shops:
    print(">>",shop)
print("> All items:", all_items)

# -

# ## Generating random test data

# +
import random
import sys
import numpy as np

# Creates given amount of shops. Assigns them random positions if None are provided, else uses those.
def create_shops(shop_count=10, positions=None):
    if positions is None:
        #print("Note: Creating random positions as none were provided.")
        positions = [(random.randint(0, 1000), random.randint(0, 1000)) for i in range(shop_count)]
    shops = [Shop(i, positions[i][0], positions[i][1], [], []) for i in range(shop_count)]
    return shops
    
# Creates collections of items under given parameters and assigns them to shops
def create_and_assign_items(shops, total_item_types=10, min_items_per_shop=1, max_items_per_shop=10, min_price=1, max_price=5):
    base_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if total_item_types > len(base_letters):
        raise Exception("Can only support {} item types, but was asked {}.".format(len(base_letters), total_item_types))
    item_types = base_letters[:total_item_types]
    
    for i, shop in enumerate(shops):
        amount = random.randint(min_items_per_shop, max_items_per_shop)
        items = list(np.random.choice(list(item_types), amount, replace=False))
        shop.items.extend(items)
        shop.prices.extend([random.randint(min_price, max_price) for e in range(len(items))])
    
### Example usage:
# shops = create_shops(shop_count=5)
# create_and_assign_items(shops, total_item_types=10, min_items_per_shop=1, max_items_per_shop=5, min_price=1, max_price=6)
# items = get_all_items(shops)
# distances = distance_matrix(shops)



# -

# ## Visualization code

# +
from matplotlib import pyplot as plt
import networkx as nx

# Visualizes a given ordered collection of Shops using matplotlib and networkx
def visualize(shops, start=[0, 0], header="", x="X position", y="Y position"):
    G = nx.DiGraph()
    #print("Visualizing order:", [s.id for s in shops])
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
    pos = nx.get_node_attributes(G,'pos')
    # Create edge labels (inside node)
    label = {e.id:e.id for e in shops}
    label["Start"] = "X"
    nx.draw(G, pos, labels=label)

    # Create legend of items in the style: "Shop 0: A 1, B 2"
    legend = []
    for e in shops:
        if len(e.items) == 0:
            continue
        lgn = "Shop " + str(e.id) + ": " + "".join([str(e.items[i]) + " " + str(e.prices[i]) + ", " for i in range(len(e.items))])
        legend.append(lgn)
    plt.legend(legend, bbox_to_anchor=(1, 1))
    
    # Set header, axis labels if provided
    if x is not "":
        plt.xlabel(x)
        plt.axis("on")
    if y is not "":
        plt.ylabel(y)
        plt.axis("on")
    if header is not "":
        plt.title(header)
    plt.show()

### Example usage:
# visualize(shops)



# -

# ## Algorithm 
# **Three phased algorithm:**
# 1. Set cover - pick shops greedily until a shop for all items have been found
# 2. Price optimization - buy each item only from the cheapest shop, discarding shops where no items are bought
# 3. TSP - reorder the shop visit order

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
    #remove the shops we didn't buy anything from and perform TSP on them
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
        i += 1
    
    # Phase 1 modification: Remove bought items from subsequent shops
    items_already_bought = set()
    for shop in opt_shops:
        # Find indices of items in this shop that are already bought (those are to be removed)
        indices = [shop.items.index(e) for e in items_already_bought if e in shop.items]
        # Update our list of bought items
        items_already_bought.update(shop.items)
        # Create new list of items, clear the old one and add items there
        new_items = [shop.items[i] for i in range(len(shop.items)) if i not in indices]
        shop.items.clear()
        shop.items.extend(new_items)
        # Do same for list of prices
        new_prices = [shop.prices[i]  for i in range(len(shop.prices)) if i not in indices]
        shop.prices.clear()
        shop.prices.extend(new_prices)
    
    #returns list of shops we need to visit to get all items 
    return opt_shops, bought, ordered_shops

#test it with some set of items, starting at point 0,0 and distance cost 0.005
#shopping(get_all_items(shops_random), shops_random, [0,0], distances, 0.005)
# -
# ## Testing and visualizing

# +
# Testing and visualizing the algorithm with random data.

shops = create_shops(shop_count=10)
create_and_assign_items(shops, total_item_types=10, min_items_per_shop=1, max_items_per_shop=5, min_price=1, max_price=6)

print("Initial list of shops:")
for shop in shops:
    print(shop)
items = get_all_items(shops)
print("Items to be bought:", items)

distances = distance_matrix(shops)
distance_cost = 0.005
start = [0, 0]

phase_1, phase_2, phase_3 = shopping(items, shops, start, distances, distance_cost)
# Find total distance
td_1 = total_distance(phase_1, distances)
td_2 = total_distance(phase_2, distances)
td_3 = total_distance(phase_3, distances)
# Find total item cost
tic_1 = total_item_cost(phase_1, True)
tic_2 = total_item_cost(phase_2)
tic_3 = total_item_cost(phase_3)
# Calculate total cost
tc_1 = tic_1 + td_1 * distance_cost
tc_2 = tic_2 + td_2 * distance_cost
tc_3 = tic_3 + td_3 * distance_cost

# Print results of Phase 1
print("-----------")
print("Results of Phase 1:")
for shop in phase_1:
    print(shop)
print("Total distance", td_1 , "| item cost:", tic_1, "| total cost:", tc_1)

# Print results of Phase 2
print("-----------")
print("Results of Phase 2:")
for shop in phase_2:
    print(shop)
print("Total distance:", td_2, "| item cost:", tic_2, "| total cost:", tc_2)
    
# Print results of Phase 3
print("-----------")
print("Results of Phase 3:")
for shop in phase_3:
    print(shop)
print("Total distance:", td_3, "| item cost:", tic_3, "| total cost:", tc_3)

# Create headers for visualization and visualize shop visit order for each phase.
header_1 = "Phase 1 result. Distance %.2f, item cost %d, total cost %.2f." % (td_1, tic_1, tc_1)
header_2 = "Phase 2 result. Distance %.2f, item cost %d, total cost %.2f." % (td_2, tic_2, tc_2)
header_3 = "Phase 3 result. Distance %.2f, item cost %d, total cost %.2f." % (td_3, tic_3, tc_3)
visualize(phase_1, start=start, header=header_1)
visualize(phase_2, start=start, header=header_2)
visualize(phase_3, start=start, header=header_3)
# -
# ## Running tests
# Run the algorithm N times, see the avg gains for each phase, find an example of data where the algorithm works well, where it doesn't.

# +
test_count = 2500

best_shops = None
best_metrics = (0, 0, 0)
worst_shops = None
worst_metrics = (0, 0, 0)
all_metrics_delta = []

for i in range(test_count):
    shops = create_shops(shop_count=10)
    create_and_assign_items(shops, total_item_types=12, min_items_per_shop=2, max_items_per_shop=6, 
                            min_price=1, max_price=6)
    items = get_all_items(shops)
    distances = distance_matrix(shops)
    start = [0, 0]
    
    p1, p2, p3 = shopping(items, shops, start, distances, distance_cost)
    phases = (p1, p2, p3)
    td = [total_distance(phase, distances) for phase in phases]
    tic_1 = total_item_cost(phase_1, True)
    tic_2 = total_item_cost(phase_2)
    tic_3 = total_item_cost(phase_3)
    tic = [tic_1, tic_2, tic_3]
    tc = [tic[i] + td[i] * distance_cost for i in range(3)]
    
    metrics_delta = (td[2] - td[0], tic[2] - tic[0], tc[2] - tc[0])
    all_metrics_delta.append(metrics_delta)
    
    #print("metrics:", td, tic, tc)
    #print("tc2-tc0:", tc[2]-tc[0])
    #best_metrics[2] is total TOTAL COST
    #print("Current result:", tc[2] - tc[0])
    #print("current best", best_metrics[2])
    if best_shops is None or best_metrics[2] > tc[2] - tc[0]:
        #print("Comparison:", best_metrics[2], tc[2] - tc[0])
        best_shops = copy.deepcopy(shops)
        best_metrics = metrics_delta
        #print("updated best to", best_metrics)
    if worst_shops is None or worst_metrics[2] < tc[2] - tc[0]:
        worst_shops = copy.deepcopy(shops)
        worst_metrics = metrics_delta
        #print("updated worst to", worst_metrics)

print("Tests finished.")
print("Worst:", worst_metrics)
print("Best:", best_metrics)

print("best shops list:")
for shop in best_shops:
    print(shop)

distances = distance_matrix(best_shops)
items = get_all_items(best_shops)
for i,phase in enumerate(shopping(items, best_shops, start, distances, distance_cost)):
    td = total_distance(phase, distances)
    tic = total_item_cost(phase, True) if i==0 else total_item_cost(phase)
    tc = tic + td * distance_cost
    header = "Distance %.2f, item cost %d, total cost %.2f." % (td, tic, tc)
    visualize(phase, start=start, header=header)


print("Worst shops list:")
for shop in worst_shops:
    print(shop)

distances = distance_matrix(worst_shops)
items = get_all_items(worst_shops)
for i,phase in enumerate(shopping(items, worst_shops, start, distances, distance_cost)):
    td = total_distance(phase, distances)
    tic = total_item_cost(phase, True) if i==0 else total_item_cost(phase)
    tc = tic + td * distance_cost
    header = "Distance %.2f, item cost %d, total cost %.2f." % (td, tic, tc)
    visualize(phase, start=start, header=header)

    
all_delta_distances = [e[0] for e in all_metrics_delta]
all_delta_item_prices = [e[1] for e in all_metrics_delta]
all_delta_cost = [e[2] for e in all_metrics_delta]
print("Avg distance between p1 and p3:", sum(all_delta_distances)/len(all_delta_distances))
print("Avg better item total cost change:", sum(all_delta_item_prices)/len(all_delta_item_prices))
print("Avg total cost change:", sum(all_delta_cost)/len(all_delta_cost))
# -



