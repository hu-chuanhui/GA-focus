import numpy as np
import time
import cv2
import random

from settings import *

class QuadTree():
    def __init__(self, range_, collision=-1, depth=0):
        self.collision = collision  # -1 for undecided, 0 for safe, 1 for obstacle
        self.depth = depth
        self.range = range_  # [x0, y0, x1, y1], where [x0, y0] is top-left corner, [x1, y1] is bottom_right corner, 
        self.parent = None
        self.children = []
        self.neighbors = []
        self.area = 0
        

def build_quad_tree(img):
    root = QuadTree(range_=[0, 0, img.shape[1], img.shape[0]], collision=-1, depth=0)
    queue = [root]
    while queue:
        block = queue.pop(0)
        # print(f"depth={block.depth}, range = {block.range}, collision = {block.collision}")
        if block.collision == -1:
            x0, y0, x1, y1 = block.range
            xm = int((x0 + x1) // 2)
            ym = int((y0 + y1) // 2)
            new_depth = block.depth + 1 
            
            children_ranges = [[x0, y0, xm, ym],   # top-left child
                              [xm, y0, x1, ym],   # top-right child
                              [x0, ym, xm, y1],   # bottom-left child
                              [xm, ym, x1, y1]]   # bottom-right child
            for range_ in children_ranges:
                child = QuadTree(range_=range_, depth=new_depth)
                obstacle_percent = np.sum(img[range_[1]:range_[3]+1, range_[0]:range_[2]+1]) / (255 * (range_[2]-range_[0]+1) * (range_[3]-range_[1]+1))
                if new_depth == QUAD_TREE_MAX_LEVEL:
                    child.collision = 1 if obstacle_percent > 0 else 0
                elif obstacle_percent >= OBSTACLE_PERCENT:
                    child.collision = 1
                elif obstacle_percent == 0:
                    child.collision = 0
                else:
                    child.collision = -1
                    
                if child.collision == 1:
                    x0_c, y0_c, x1_c, y1_c = child.range
                    if x0_c == 0 or y0_c == 0 or x1_c == MAP_SIZE or y1_c == MAP_SIZE:
                        child.area = MAP_SIZE * MAP_SIZE
                    else:
                        child.area = (x1_c - x0_c) * (y1_c - y0_c)
                elif child.collision == 0:
                    x0_c, y0_c, x1_c, y1_c = child.range
                    child.area = (x1_c - x0_c) * (y1_c - y0_c)
                
                child.parent = block
                queue.append(child)
                    # print(f"append depth {new_depth}, range {range_}")
                block.children.append(child)
                            
            
    return root

def is_collision_free(quadtree, A):
    x, y = A
    
    # when both vertices of a segment lies on the boundary of the map,
    # the area calculation will be zero.     
    if x == 0 or y == 0:
        return False
    elif quadtree.collision == 1:
        return False
    elif quadtree.collision == 0:
        return True
    else:
        for child in quadtree.children:
            x0, y0, x1, y1 = child.range
            if x0 <= x <= x1 and y0 <= y <= y1:
                return is_collision_free(child, A)
        return False

def euclidean(x0, y0, x1, y1):
    return np.sqrt((x0-x1)**2 + (y0-y1)**2)

def pick_random_wp(free_blocks):
    weights = [block.area for block in free_blocks]
    weights = np.array(weights)
    weights = weights / weights.sum()
    block = np.random.choice(free_blocks, p=weights, size=1)[0]
    x = np.random.randint(block.range[0], block.range[2])
    y = np.random.randint(block.range[1], block.range[3])
    return (x, y)

# def pick_random_corner_wp(free_blocks):
# #     block = random.choice(free_blocks)
#     block = random.choices(free_blocks, weights=[block.area for block in free_blocks], k=1)[0]
#     x0, y0, x1, y1 = block.range
#     x = random.choice([x0+5, x1-5])
#     y = random.choice([y0+5, y1-5])
#     return (x, y)

def point_in_map(x0, y0, row, col):
    if 0 <= x0 < col and 0 <= y0 < row:
        return True
    else:
        return False

def ccw(A, B, C):  # return True if crossproduct(AB, AC) > 0
    # https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect_point(A, B, C, D):  # return the intersection point of AB and CD
    # represent the two lines by a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0 and 
    xa, ya = A
    xb, yb = B
    xc, yc = C
    xd, yd = D
    
    a1 = yb - ya
    b1 = xa - xb
    c1 = ya*(xb-xa) - xa*(yb-ya)
    a2 = yd - yc
    b2 = xc - xd
    c2 = yc*(xd-xc) - xc*(yd-yc)    
    
    if (a1*b2 - a2*b1) != 0:
        x0 = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
        y0 = (a2*c1 - a1*c2) / (a1*b2 - a2*b1)
    else:
        x0 = None
        y0 = None
    
    return (x0, y0)

def point_on_segment(M, A, B):
    xm, ym = M 
    return (xm-A[0])*(xm-B[0]) <= 0 and (ym-A[1])*(ym-B[1]) <= 0

def intersect(A, B, C, D):  # Return true if line segments AB and CD intersect
    xm, ym = intersect_point(A, B, C, D)
    if xm == None:
        return False
    return point_on_segment((xm, ym), A, B) and point_on_segment((xm, ym), C, D)

def intersect_square(range_, A, B):
    x0, y0, x1, y1 = range_
    
    # one of the point is in the square
    if (x0 < A[0] < x1 and y0 < A[1] < y1) or (x0 < B[0] < x1 and y0 < B[1] < y1):
#         print("in square")
        return True
    
    # if the segment intersects with one side of the square
    C, D = [x0, y0], [x0, y1]
    if intersect(A, B, C, D):
        return True
    
    C, D = [x0, y0], [x1, y0]
    if intersect(A, B, C, D):
        return True
    
    C, D = [x0, y1], [x1, y1]
    if intersect(A, B, C, D):
        return True

    C, D = [x1, y0], [x1, y1]
    if intersect(A, B, C, D):
        return True

    return False

def triangle_area(A, B, C):
    x1, y1, x2, y2, x3, y3 = A[0], A[1], B[0], B[1], C[0], C[1]
    return 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

def calculate_ccw_area_proportion(square_range, A, B):
    # calculate the area on the right hand side of the line segment AB. The line and the square has to intersect
    xa, ya = A
    xb, yb = B
    x0, y0, x1, y1 = square_range
    
    xm = (x0 + x1) / 2
    ym = (y0 + y1) / 2
    
    # if the line segment AB aligns with one of the side, 
    if (xa == xb and xb == x0) or (xa == xb and xb == x1) or (ya == yb and yb == y0) or (ya == yb and yb == y1):
#         print("align")
        if ccw(A, B, (xm, ym)):
            return 1
        else:
            return 0
    
    points = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
    sides = [[(x0, y0), (x0, y1)], [(x0, y1), (x1, y1)], [(x1, y1), (x1, y0)], [(x1, y0), (x0, y0)]]
    
    # find the intersection points between the line AB and the four sides of the square
    intersections = []
    for side in sides:
#         print(f"intersect{(A, B, side[0], side[1])}: {intersect(A, B, side[0], side[1])}")
        if intersect(A, B, side[0], side[1]):
            p = intersect_point(A, B, side[0], side[1])
            if p not in intersections:
                intersections.append(p)
    if len(intersections) == 1:
#         print(intersections)
        if ccw(A, B, (xm, ym)):
            return 1
        else:
            return 0
    
    # find the vertices of squares on the counter-clockwise side of AB
    cw_points = []
    ccw_points = []
    for point in points:
#         print(point)
        if ccw(A, B, point):
            ccw_points.append(point)
        else:
            cw_points.append(point)

#     print(f"intersections: {intersections}")
#     print(f"ccw_points: {ccw_points}")
#     print(f"cw_points: {cw_points}")
    
    # calculate the proportion of area on the counter-clockwise side of AB
    if len(ccw_points) == 0:
        return 0
    elif len(ccw_points) == 1:
        return triangle_area(intersections[0], intersections[1], ccw_points[0]) / abs((x1-x0) * (y1-y0))
    elif len(ccw_points) == 2:
        polygon = [intersections[0], intersections[1], ccw_points[0], ccw_points[1]]
        if ccw(intersections[0], ccw_points[0], intersections[1]) != ccw(intersections[0], ccw_points[0], ccw_points[1]):
            area = triangle_area(intersections[0], ccw_points[0], intersections[1]) + triangle_area(intersections[0], ccw_points[0], ccw_points[1])
        else:
            area = triangle_area(intersections[0], ccw_points[1], intersections[1]) + triangle_area(intersections[0], ccw_points[1], ccw_points[0])
        return area / abs((x1-x0) * (y1-y0))
    elif len(ccw_points) == 3:
        return 1 - triangle_area(intersections[0], intersections[1], cw_points[0]) / abs((x1-x0) * (y1-y0))

def is_neighbor(quad_tree_A, quad_tree_B):
    x0_A, y0_A, x1_A, y1_A = quad_tree_A.range
    x0_B, y0_B, x1_B, y1_B = quad_tree_B.range
    
    # the following two cases are that the two blocks are connected on one side
    if abs(x1_A - x0_B) == 0 or abs(x0_A - x1_B) == 0:
        if (y0_A - y0_B) * (y1_A - y1_B) <= 0:
            return True
    
    if abs(y1_A - y0_B) == 0 or abs(y0_A - y1_B) == 0:
        if (x0_A - x0_B) * (x1_A - x1_B) <= 0:
            return True
    
    # the following four cases are that the two blocks are diagonally connected at a corner
    if abs(x1_A - x0_B) == 0 and abs(y1_A - y0_B) == 0:
        return True
    
    if abs(x0_A - x1_B) == 0 and abs(y1_A - y0_B) == 0:
        return True
    
    if abs(x1_A - x0_B) == 0 and abs(y0_A - y1_B) == 0:
        return True
    
    if abs(x0_A - x1_B) == 0 and abs(y0_A - y1_B) == 0:
        return True
    
    
    return False
       
def find_all_neighbors(obs_blocks):
    for target_block in obs_blocks:
        queue = [target_block]
        closed = set()
        while queue:
            block = queue.pop(0)
            if block in closed:
                continue
            closed.add(block)
            if is_neighbor(block, target_block):
                if block.collision == 1:
                    if block not in target_block.neighbors:
                        target_block.neighbors.append(block)
                    if target_block not in block.neighbors:
                        block.neighbors.append(target_block)
                elif block.collision == 0:
                    continue
                else:  # if block.collision == -1:
                    for child in block.children:
                        if child in closed or child in queue:
                            continue
                        if is_neighbor(target_block, child):
                            queue.append(child)
            
            if not block.parent is None:
                queue.append(block.parent)
                for child in block.parent.children:
                    if child == block:
                        continue
                    if child in closed or child in queue:
                        continue
                    if is_neighbor(target_block, child):
                        queue.append(child)
            

def find_all_islands(obs_blocks, island_test_img=None):
    '''
    obs_blocks: list of [block: QuadTree()], each block has block.collision == 1
    return: islands: list of [island]. Each island is a list of [block: QuadTree()] in the same island 
    '''
    t1 = time.time()

    islands = []
    visited = set()

    find_all_neighbors(obs_blocks)
    
    # island_test_img = np.zeros([MAP_SIZE, MAP_SIZE], np.uint8)

    for obs_block in obs_blocks:
        if obs_block in visited:
            continue

        in_island = []

        queue = [obs_block]
        while queue:
        #     print(len(to_do))
            block = queue.pop(0)
            in_island.append(block)
            visited.add(block)
            
            for neighbor in block.neighbors:
                if (neighbor not in visited) and (neighbor not in queue):
                    queue.append(neighbor)
            
            
        islands.append(in_island)
    
    # print(f"time consumed to find all islands: {time.time() - t1}")
    
    islands_ranges = []
    islands_areas = []

    for island in islands[:]:
        x0_min, y0_min, x1_max, y1_max = MAP_SIZE, MAP_SIZE, 0, 0
        area = 0
        for block in island:
            x0, y0, x1, y1 = block.range
            x0_min, y0_min, x1_max, y1_max = min(x0_min, x0), min(y0_min, y0), max(x1_max, x1), max(y1_max, y1)
            area += block.area
            if not island_test_img is None:
                island_test_img = cv2.rectangle(island_test_img, (x0, y0), (x1-1, y1-1), [255], 5)
        islands_ranges.append([x0_min, y0_min, x1_max, y1_max])
        islands_areas.append(area)

    return islands, islands_ranges, islands_areas


            
# def find_all_islands(obs_blocks, island_test_img=None):
#     '''
#     obs_blocks: list of [block: QuadTree()], each block has block.collision == 1
#     return: islands: list of [island]. Each island is a list of [block: QuadTree()] in the same island 
#     '''
#     t1 = time.time()

#     islands = []
#     visited = set()

#     # island_test_img = np.zeros([MAP_SIZE, MAP_SIZE], np.uint8)

#     for obs_block in obs_blocks:
#         if obs_block in visited:
#             continue

#         in_island = []

#         to_search_neighbor = [obs_block]
#         while to_search_neighbor:
#         #     print(len(to_do))
#             target_block = to_search_neighbor.pop(0)
#             visited.add(target_block)
#             x0_t, y0_t, x1_t, y1_t = target_block.range

#             queue = [target_block]

#             closed = []

#             while queue:
#                 block = queue.pop(0)
#                 x0, y0, x1, y1 = block.range
#                 closed.append(block)

#                 # draw block range
#                 if block.collision == 1:
#                     if block not in in_island:
#                         in_island.append(block)
#                         if block not in to_search_neighbor:
#                             to_search_neighbor.append(block)
#                         if block not in target_block.neighbors:
#                             target_block.neighbors.append(block)
#                         if target_block not in block.neighbors:
#                             block.neighbors.append(target_block)
#                         # island_test_img = cv2.rectangle(island_test_img, (x0, y0), (x1, y1), [255], -1)
#                 elif block.collision == 0:
#                     pass
#                     # island_test_img = cv2.rectangle(island_test_img, (x0, y0), (x1, y1), [255], 5)
#                 else:  # if block.collision == -1:
#                     for child in block.children:
#                         if child in closed or child in queue:
#                             continue
#                         if is_neighbor(target_block, child):
#                             queue.append(child)

#                 if block.parent in closed or not block.parent:
#                     continue
#         #         if x0 < x0_t and x1 > x1_t and y0 < y0_t and y1 > y1_t:
#         #             continue

#                 queue.append(block.parent)
#                 for child in block.parent.children:
#                     if child == block:
#                         continue
#                     if child in in_island or child in closed or child in queue:
#                         continue
#                     if is_neighbor(target_block, child):
#                         queue.append(child)

#         islands.append(in_island)
    
#     print(f"time consumed to find all islands: {time.time() - t1}")
    
#     islands_ranges = []
#     islands_areas = []

#     for island in islands[:]:
#         x0_min, y0_min, x1_max, y1_max = MAP_SIZE, MAP_SIZE, 0, 0
#         area = 0
#         for block in island:
#             x0, y0, x1, y1 = block.range
#             x0_min, y0_min, x1_max, y1_max = min(x0_min, x0), min(y0_min, y0), max(x1_max, x1), max(y1_max, y1)
#             area += block.area
#             if not island_test_img is None:
#                 island_test_img = cv2.rectangle(island_test_img, (x0, y0), (x1, y1), [255], 5)
#         islands_ranges.append([x0_min, y0_min, x1_max, y1_max])
#         islands_areas.append(area)

#     return islands, islands_ranges, islands_areas

def draw_island(img, island):
    '''
    img: [height, width] array
    island: list of [block: QuadTree], each block has block.collision == 1
    '''
    for block in island:
        x0, y0, x1, y1 = block.range
        img = cv2.rectangle(img, (x0, y0), (x1-1, y1-1), [255], 5)
    return img

def intersected_islands(islands, islands_ranges, A, B):
    intersected_islands_idx = []

    # find possibly intersected islands
    for i in range(len(islands)):
        island_range = islands_ranges[i]
        if intersect_square(island_range, A, B):
            intersected_islands_idx.append(i)
    return intersected_islands_idx

def calculate_ccw_area_of_island(island, A, B, ccw_area_test_img=None):
    ccw_area = 0
    closed = set()
    ccw_neighbors = []
    
    # find all intersected blocks, add their ccw neighbors to ccw_neighbors
    intersected_blocks = []
    for block in island:
        x0, y0, x1, y1 = block.range
        if intersect_square(block.range, A, B):
            intersected_blocks.append(block)
            
    for block in intersected_blocks:
        x0, y0, x1, y1 = block.range
        closed.add(block)
        if not ccw_area_test_img is None:
            ccw_area_test_img = cv2.rectangle(ccw_area_test_img, (x0, y0), (x1-1, y1-1), [127], -1)     
            ccw_area_test_img = cv2.rectangle(ccw_area_test_img, (x0, y0), (x1-1, y1-1), [255], 5)     
        ccw_area += calculate_ccw_area_proportion(block.range, A, B) * block.area
        for neighbor in block.neighbors:
            if neighbor in closed or neighbor in ccw_neighbors or neighbor in intersected_blocks:
                continue
            x0, y0, x1, y1 = neighbor.range
            xm, ym = (x0+x1)/2, (y0+y1)/2
            if ccw(A, B, (xm, ym)):
                ccw_neighbors.append(neighbor)

    # search all the connected neighbors of ccw_neighbors
    while ccw_neighbors:
        block = ccw_neighbors.pop(0)
        ccw_area += block.area
        closed.add(block)
        x0, y0, x1, y1 = block.range
        if not ccw_area_test_img is None:
            ccw_area_test_img = cv2.rectangle(ccw_area_test_img, (x0, y0), (x1-1, y1-1), [255], 10)     
        for neighbor in block.neighbors:
            if neighbor in closed or neighbor in ccw_neighbors:
                continue
            ccw_neighbors.append(neighbor)
    return ccw_area

def calculate_collision_area_of_island(island, A, B, collision_area_test_img=None, island_area=None):
    # find all intersected blocks, calculate the collision area by finding cw neighbor and ccw neighbor s
    intersected_blocks = set()
    for block in island:
        x0, y0, x1, y1 = block.range
        if intersect_square(block.range, A, B):
            intersected_blocks.add(block)
            
    connected_intersected_blocks = []
    closed = set()
    for intersected_block in intersected_blocks:
        if intersected_block in closed:
            continue
        connected_blocks = set()
        queue = [intersected_block]
        while queue:
            block = queue.pop(0)
            if block in closed:
                continue
            connected_blocks.add(block)
            closed.add(block)
            for neighbor in block.neighbors:
                if (neighbor not in queue) and (neighbor not in closed) and (neighbor in intersected_blocks):
                    queue.append(neighbor)
        connected_intersected_blocks.append(connected_blocks)
    
#     print(f"number of independent intersection segments: {len(connected_intersected_blocks)}")
    
    col_area = 0
    
    for intersected_blocks in connected_intersected_blocks:
        cw_area = 0
        ccw_area = 0
        cw_neighbors = []
        ccw_neighbors = []
        closed = set()
        for block in intersected_blocks:
            x0, y0, x1, y1 = block.range
            closed.add(block)
#             if not collision_area_test_img is None:
#                 collision_area_test_img = cv2.rectangle(collision_area_test_img, (x0, y0), (x1-1, y1-1), [127], -1)     
#                 collision_area_test_img = cv2.rectangle(collision_area_test_img, (x0, y0), (x1-1, y1-1), [255], 5)     
            block_ccw_prop = calculate_ccw_area_proportion(block.range, A, B)
            ccw_area += block_ccw_prop * block.area
            cw_area += (1 - block_ccw_prop) * block.area
            for neighbor in block.neighbors:
                if (neighbor in closed) or (neighbor in cw_neighbors) or (neighbor in ccw_neighbors) or (neighbor in intersected_blocks):
                    continue
                x0, y0, x1, y1 = neighbor.range
                xm, ym = (x0+x1)/2, (y0+y1)/2
                if ccw(A, B, (xm, ym)):
                    ccw_neighbors.append(neighbor)
                else:
                    cw_neighbors.append(neighbor)

        # search all the connected neighbors of cw_neighbors
        if (island_area is None) or (not collision_area_test_img is None):
            all_cw_neighbors = []
            while cw_neighbors:
                block = cw_neighbors.pop(0)
                all_cw_neighbors.append(block)
                cw_area += block.area
                closed.add(block)
    #             x0, y0, x1, y1 = block.range
                for neighbor in block.neighbors:
                    if (neighbor in closed) or (neighbor in cw_neighbors) or (neighbor in ccw_neighbors):
                        continue
                    cw_neighbors.append(neighbor)

            # search all the connected neighbors of ccw_neighbors
            all_ccw_neighbors = []
            while ccw_neighbors:
                block = ccw_neighbors.pop(0)
                all_ccw_neighbors.append(block)
                ccw_area += block.area
                closed.add(block)
    #             x0, y0, x1, y1 = block.range
                for neighbor in block.neighbors:
                    if (neighbor in closed) or (neighbor in cw_neighbors) or (neighbor in ccw_neighbors):
                        continue
                    ccw_neighbors.append(neighbor)
        else:  # if island_area is given and no need to draw collision_area_test_img
            while cw_neighbors or ccw_neighbors:
                if cw_neighbors and ccw_neighbors:
                    # pop the next cw neighbor
                    block = cw_neighbors.pop(0)
                    cw_area += block.area
                    closed.add(block)
                    for neighbor in block.neighbors:
                        if (neighbor in closed) or (neighbor in cw_neighbors) or (neighbor in ccw_neighbors):
                            continue
                        cw_neighbors.append(neighbor)
                        
                    # pop the next ccw neighbor
                    block = ccw_neighbors.pop(0)
                    ccw_area += block.area
                    closed.add(block)
                    for neighbor in block.neighbors:
                        if (neighbor in closed) or (neighbor in cw_neighbors) or (neighbor in ccw_neighbors):
                            continue
                        ccw_neighbors.append(neighbor)
                
                elif not cw_neighbors:
                    ccw_area = island_area - cw_area
                    break
                else:  # if not ccw_neighbors:
                    cw_area = island_area - ccw_area
                    break
                
            
        # draw collision area
        if not collision_area_test_img is None:
            if cw_area < ccw_area:
                to_draw = all_cw_neighbors
            else:
                to_draw = all_ccw_neighbors
#             to_draw = all_ccw_neighbors
            for block in to_draw:
                x0, y0, x1, y1 = block.range
                collision_area_test_img = cv2.rectangle(collision_area_test_img, (x0, y0), (x1-1, y1-1), [255], -1) 
            
            
            for block in intersected_blocks:
                vertices = []
                x0, y0, x1, y1 = block.range
                
                for x in range(x0, x1+1):
                    for y in range(y0, y1+1):
                        if cw_area < ccw_area and not ccw(A, B, (x, y)):
                            collision_area_test_img[y, x] = 127
                        elif cw_area >= ccw_area and ccw(A, B, (x, y)):
                            collision_area_test_img[y, x] = 127
        
        col_area += min(cw_area, ccw_area)
        
    return col_area