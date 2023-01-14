import numpy as np

##############
# Greenhouse #
##############
id_camvid_to_greenhouse = np.array(
    [
        3,  # Sky
        1,  # Building
        1,  # Pole
        2,  # Road
        2,  # Pavement
        0,  # Tree
        1,  # SignSymbol
        1,  # Fence
        1,  # Car
        3,  # Pedestrian
        3,  # Bicyclist
        1,  # Road_marking(?)
        3,  # Unlabeled
    ]
)

id_cityscapes_to_greenhouse = np.array(
    [
        2,  # Road
        2,  # Sidewalk
        1,  # Building
        1,  # Wall
        1,  # Fence
        1,  # Pole
        1,  # Traffic light
        1,  # Traffic sign
        0,  # Vegetation
        2,  # Terrain
        3,  # Sky
        3,  # Person
        3,  # Rider
        1,  # Car
        1,  # Truck
        1,  # Bus
        1,  # Train
        1,  # Motorcycle
        1,  # Bicycle
        3,  # Background
    ]
)

id_forest_to_greenhouse = np.array(
    [2, 0, 0, 1, 1], dtype=np.uint8  # road  # grass  # tree  # sky  # obstacle
)

###############
# OxfordRobot #
###############
id_camvid_to_oxford = np.array(
    [
        1,  # Sky -> Sky
        7,  # Building -> Building
        16, # Pole -> Other obstacle
        11, # Road ->  Road
        10, # Pavement -> Side walk
        8,  # Tree -> Grass
        5,  # SignSymbol -> Traffic sign
        16, # Fence -> Other obstacle
        4,  # Car -> Automobile
        2,  # Pedestrian -> Person
        2,  # Person
        14, # Road_marking(?)-> Other road marking
        0,  # Unlabeled -> Background
    ], dtype=np.uint8 
)

id_cityscapes_to_oxford = np.array(
    [
        11,  # Road -> ROad
        10,  # Sidewalk -> side walk
        7,  # Building -> building
        16,  # Wall -> Other obstacle
        16,  # Fence -> Other obstacle
        16,  # Pole -> Other obstacle
        6,  # Traffic light -> Traffic light
        5,  # Traffic sign -> Traffic sign
        8,  # Vegetation -> grass
        0,  # Terrain -> Background
        1,  # Sky -> sky
        2,  # Person -> person
        2,  # Rider -> person
        4,  # Car -> automobile
        4,  # Truck -> automobile
        4,  # Bus -> automobile
        4,  # Train -> automobile
        3,  # Motorcycle -> two-wheel vehicle
        3,  # Bicycle -> two-wheel vehicle
        0,  # Background
    ], dtype=np.uint8 
)

id_forest_to_oxford = np.array(
    [11, 8, 8, 1, 16], dtype=np.uint8  # road  # grass  # tree  # sky  # obstacle
)