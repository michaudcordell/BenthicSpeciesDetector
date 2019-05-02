#!/u#!/usr/bin/env python3
import numpy as np

"""
This module contains methods for certain geometric calculations needed by the Benthic Species Detector.
"""

def point_enclosed(point, rect):
    """
    Test whether a point is enclosed by a rectangle.

    :param point: a point (x, y) to test
    :type point: tuple
    :param rect: a rectangle (x, y, w, h) to test, where
        x, y is the top-left corner of the rectangle,
        w is width, and
        h is height
    :type rect: tuple
    :return: a boolean that is True if the point is enclosed by the rectangle, False otherwise
    """

    (x, y, w, h) = rect
    (px, py) = point
    enclosed = False

    if (px >= x) and (py >= y) and (px <= x + w) and (py <= y + h):
        enclosed = True

    return enclosed

def scale_point(point, width, height, width_scaling, height_scaling):
    """
    Scale a point from a reference rectangle to a scaled version of the reference rectangle.

    :param point: the point (x, y) to be scaled
    :type point: tuple
    :param width: the width of the point's reference rectangle
    :type width: int
    :param height: the height of the point's reference rectangle
    :type height: int
    :param width_scaling: the factor by which to scale the width
    :type width_scaling: float
    :param height_scaling: the factor by which to scale the height
    :type height_scaling: float
    :return: the point (x, y) scaled within the scaled rectangle
    """

    (x, y) = point
    scaled_width = int(np.round(width * width_scaling))
    scaled_height = int(np.round(height * height_scaling))
    scaled_x = int(np.round((x / width) * scaled_width))
    scaled_y = int(np.round((y / height) * scaled_height))
    scaled_point = (scaled_x, scaled_y)

    return scaled_point

def recontextualize_rect(rect, context_width, context_height, width_scaling, height_scaling):
    """
    Scale a rectangle proportionally from one rectangular context to a scaled version of the original.

    :param rect: a rectangle of the form (x_top_left, y_top_left, width, height)
    :type rect: tuple
    :param context_width: the width of the original context
    :type context_width: int
    :param context_height: the height of the original context
    :type context_height: int
    :param width_scaling: the factor by which to scale the width of the context
    :type width_scaling: float
    :param height_scaling: the factor by which to scale the height of the context
    :type height_scaling: float
    :return: a rectangle (x_top_left, y_top_left, width, height) proportionally scaled and positioned in the new
             context
    """

    (x, y, rect_width, rect_height) = rect
    scaled_context_width = int(np.round(context_width * width_scaling))
    scaled_context_height = int(np.round(context_height * height_scaling))
    (scaled_x, scaled_y) = scale_point((x, y), context_width, context_height, width_scaling, height_scaling)
    scaled_rect_width = int(np.round((rect_width / context_width) * scaled_context_width))
    scaled_rect_height = int(np.round((rect_height / context_height) * scaled_context_height))

    scaled_rect = (scaled_x, scaled_y, scaled_rect_width, scaled_rect_height)
    return scaled_rect

def filter_similar_points(point_list, distance_threshold=10):
    """
    Filter a list of points such that no point is within distance_threshold pixels from another.

    :param point_list: a list of points to filter through based on similarity
    :type point_list: list
    :param distance_threshold: the minimum distance two points must be from each other to be considered dissimilar
                               (10 pixels by default)
    :type distance_threshold: int
    :return: a list of points filtered by distance to each other
    """

    points_to_process = point_list.copy()
    pivot = 0
    while pivot + 1 < len(points_to_process):
        scan = len(points_to_process) - 1
        while scan > pivot:
            distance = euclideanDistance(points_to_process[pivot], points_to_process[scan])
            if distance < distance_threshold:
                points_to_process[scan] = midpoint(points_to_process[pivot], points_to_process[scan])
                # now remove pivot
                if pivot == 0:
                    points_to_process = points_to_process[pivot + 1:]
                    scan -= 1
                else:
                    points_to_process_new = list(points_to_process[:pivot])
                    points_to_process_right = list(points_to_process[pivot + 1:])
                    points_to_process_new.extend(points_to_process_right)
                    points_to_process = points_to_process_new
                    scan -= 1
            else:
                scan -= 1
        pivot += 1

    return points_to_process

def euclidean_distance(point1, point2):
    """
    Calculate the euclidean distance between point1 and point2.

    :param point1: point (x1, y1)
    :type point1: tuple
    :param point2: point (x2, y2)
    :type point2: tuple
    :return: the euclidean distance between point1 and point2
    """

    difference = (point2[0] - point1[0], point2[1] - point1[1])
    distance = np.sqrt(difference[0]*difference[0] + difference[1]*difference[1])

    return distance

def midpoint(point1, point2):
    """
    Calculate the midpoint between point1 and point2.

    :param point1: point (x1, y1)
    :type point1: tuple
    :param point2: point (x2, y2)
    :type point2: tuple
    :return: the midpoint between point1 and point2
    """

    (p1x, p1y) = point1
    (p2x, p2y) = point2

    mp = (int((p1x + p2x) / 2.0), int((p1y + p2y) / 2.0))

    return mp

def angle_between_vectors(vec1, vec2):
    """
    Calculate the angle between two vectors.

    :param vec1: a length n vector
    :type vec1: np.ndarray
    :param vec2: a length n vector
    :type vec2: np.ndarray
    :raise ZeroDivisionError: raises ZeroDivisionError if parameter vec1 or vec2 is a zero vector.
    :return: the angle between the vectors vec1 and vec2
    """

    vec1_mag = np.linalg.norm(vec1)
    vec2_mag = np.linalg.norm(vec2)

    if (vec1_mag < 0.0005) or (vec2_mag < 0.0005):
        raise ZeroDivisionError("Parameter vec1 or vec2 is a zero vector. The angle between the vectors is undefined.")

    dotprod = np.dot(vec1, vec2)
    angle = np.abs(np.arccos(dotprod / (vec1_mag * vec2_mag)))
    return angle

def bound_rect(rect, x_min, y_min, x_max, y_max):
    """
    Bound a rectangle within a range of x and y values.

    :param rect: a rectangle (x, y, w, h) to bound, where
        x, y is the top-left corner of the rectangle,
        w is width, and
        h is height
    :type rect: tuple
    :param x_min: the minimum x value to bound above, inclusive
    :type x_min: int
    :param y_min: the minimum y value to bound above, inclusive
    :type y_min: int
    :param x_max: the maximum x value to bound below, inclusive
    :type x_max: int
    :param y_max: the maximum y value to bound below, inclusive
    :type y_max: int
    :return: the original rectangle bounded above x_min, y_min and bounded below x_max, y_max
    """

    (x, y, w, h) = rect

    if x < x_min:
        x = x_min
    if x + w > x_max:
        w = x_max - x

    if y < y_min:
        y = y_min
    if y + h > y_max:
        h = y_max - y

    bounded_rect = (x, y, w, h)
    return bounded_rect

def rect_compare(rect1, rect2):
    """
    Compare how similar two rectangles are and how contained one is by the other.

    :param rect1: a rectangle tuple (x, y, w, h)
    :type rect1: tuple
    :param rect2: a rectangle tuple (x, y, w, h)
    :type rect2: tuple
    :return: a tuple (containment, similarity) with a containment score (a percentage of the area of the contained
              rectangle inside the containing rectangle; negative if rect1 is contained by rect2) and a similarity score
              (a percentage of how coinciding rect1 and rect2 are)
    """

    (x1, y1, w1, h1) = rect1
    (x2, y2, w2, h2) = rect2

    rect1_area = w1 * h1
    rect2_area = w2 * h2
    intersection_corners = (max(x1, x2), max(y1, y2), min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2))
    intersection_area = 0
    similarity = 0.0
    if intersection_corners[0] < intersection_corners[2] and intersection_corners[1] < intersection_corners[3]:
        intersection_area = (intersection_corners[2] - intersection_corners[0]) \
                            * (intersection_corners[3] - intersection_corners[1])
        similarity = (2 * intersection_area) / (rect1_area + rect2_area)

    encloses = intersection_area / float(rect2_area)
    enclosed = intersection_area / float(rect1_area)

    if encloses > enclosed:
        containment = encloses
    else:
        containment = -enclosed


    comparison = (containment, similarity)
    return comparison

def filter_similar_rects(rects_list, cap, similarity_alpha=0.1, enclosure_alpha=0.1):
    """
    Filter a list of rects such that none of the rects are too similar within a percent difference similarity_alpha
        or unenclosed by less than a percentage enclosure_alpha

    :param rects_list: a list of rects to filter by similarity and containment
    :type rects_list: list
    :param cap: the maximum number of rects to process
    :type cap: int
    :param similarity_alpha: the minimum percent difference required for the rectangles to be considered dissimilar
                             (0.1 by default)
    :type similarity_alpha: float
    :param enclosure_alpha: the minimum percent unenclosed required for a rectangle to be considered unenclosed
                              by another rectangle (0.1 by default)
    :type enclosure_alpha: float
    :return: a list of rects filtered such that no rects are too similar and none are contained by another
    """

    if cap < len(rects_list):
        rects_to_process = rects_list[:cap]
    else:
        rects_to_process = rects_list[:]

    pivot = 0

    while pivot + 1 < len(rects_to_process):
        scan = len(rects_to_process) - 1
        while scan > pivot:
            rect_comparison = rectCompare(rects_to_process[pivot], rects_to_process[scan])
            contains = True if rect_comparison[0] > (1 - enclosure_alpha) else False
            contained = True if rect_comparison[0] < -(1 - enclosure_alpha) else False
            rect_sim = rect_comparison[1] > (1 - similarity_alpha)
            if contained:
                if pivot > 0:
                    rects_to_process = np.vstack((rects_to_process[:pivot], rects_to_process[pivot + 1:]))
                else:
                    rects_to_process = rects_to_process[1:]
                scan = len(rects_to_process) - 1
            elif contains:
                if scan + 1 < len(rects_to_process):
                    rects_to_process = np.vstack((rects_to_process[:scan], rects_to_process[scan + 1:]))
                else:
                    rects_to_process = rects_to_process[:scan]
                scan -= 1
            else:
                if rect_sim >= 1 - similarity_alpha:
                    if pivot > 0:
                        rects_to_process = np.vstack((rects_to_process[:pivot], rects_to_process[pivot + 1:]))
                    else:
                        rects_to_process = rects_to_process[1:]
                    scan = len(rects_to_process) - 1
                else:
                    scan -= 1

        pivot += 1

    return rects_to_process

def find_adjacent_corners(corner, other_corners):
    """
    Find two adjacent corners assuming a closed shape.

    :param corner: a point (x, y) to which found corners are adjacent
    :type corner: tuple
    :param other_corners: a list of points of the form (x, y) through which to search for adjacent corners
    :type other_corners: list
    :return: the nearest two points, which are thus adjacent assuming all points in other_corners are vertices
              in the same closed shape
    """

    sorted_corners = other_corners.copy()
    sorted_corners.sort(key=lambda p2: euclideanDistance(corner, p2))
    if sorted_corners is not None:
        if len(sorted_corners) > 1:
            adjacent_corners = [sorted_corners[0], sorted_corners[1]]
        else:
            adjacent_corners = None
    else:
        adjacent_corners = None
    return adjacent_corners

def order_vertices(corners):
    """
    Order the given list of vertices by adjacency, assuming a closed shape.

    :param corners: a list of points (x, y)
    :type corners: list
    :return: a list of points (x, y) ordered by adjacency assuming a closed shape
    """

    if len(corners) < 3:
        ordered_vertices = corners
    else: # 3 or more corners
        unordered_vertices = corners.copy()
        first_vertex = unordered_vertices.pop(0)
        adjacent_corners = find_adjacent_corners(first_vertex, unordered_vertices)
        ordered_vertices = [adjacent_corners[0], first_vertex, adjacent_corners[1]]
        if len(unordered_vertices) == 2:
            unordered_vertices = []
        else:
            unordered_vertices.remove(adjacent_corners[0])
            unordered_vertices.remove(adjacent_corners[1])
        while len(unordered_vertices) > 1:
            distances = list(map(lambda vertex: euclideanDistance(ordered_vertices[len(ordered_vertices) - 1], vertex),
                                 unordered_vertices))
            min_distance = min(distances)
            next_vertex_index = distances.index(min_distance)
            ordered_vertices.append(unordered_vertices.pop(next_vertex_index))
        if len(unordered_vertices) > 0:
            ordered_vertices.append(unordered_vertices.pop(0))

    return ordered_vertices

def calculate_central_angles(corners):
    """
    Calculate the angles bounded between the lines from the centroid to each pair of corners.

    :param corners: a list of corners specifying a shape
    :type corners: list
    :return: a list of the angles bounded between the lines from the centroid to each pair of corners
    """

    angle_list = []
    if len(corners) < 3: # shape has no internal angles
        return angle_list
    else: # shape has 3 or more corners
        # shape has 3 or more corners, so calculate the centroid of the corners, then pivot on each corner,
        # calculate the angle
        connected_vertices = order_vertices(corners)
        centroid = calculate_centroid(connected_vertices)
        for pivot in range(0, len(connected_vertices) - 1):
            center_vec1 = [connected_vertices[pivot][0] - centroid[0], connected_vertices[pivot][1] - centroid[1]]
            center_vec2 = [connected_vertices[pivot + 1][0] - centroid[0],
                           connected_vertices[pivot + 1][1] - centroid[1]]
            angle = angle_between_vectors(center_vec1, center_vec2)
            angle_list.append(angle)

        center_vec1 = [connected_vertices[len(connected_vertices) - 1][0] - centroid[0],
                       connected_vertices[len(connected_vertices) - 1][1] - centroid[1]]
        center_vec2 = [connected_vertices[0][0] - centroid[0], connected_vertices[0][1] - centroid[1]]
        angle = angle_between_vectors(center_vec1, center_vec2)
        angle_list.append(angle)
        return angle_list

def calculate_interior_angles(corners):
    """
    Calculate the angles bounded between the lines from each corner to its adjacent corners.

    :param corners: a list of corners specifying a shape
    :type corners: list
    :return: a list of the angles bounded between the lines from each corner to its adjacent corners
    """

    angle_list = []
    if len(corners) < 3: # shape has no external angles
        return angle_list
    else: # shape has 3 or more corners
        # shape has 3 or more corners, so pivot on each corner, find the two nearest corners,
        # then calculate the angle with them and add it to the list of angles,
        # then after iterating through all pivot corners, return the angle list
        connected_vertices = order_vertices(corners)
        external_vec1 = [connected_vertices[len(connected_vertices) - 1][0] - connected_vertices[0][0],
                         connected_vertices[len(connected_vertices) - 1][1] - connected_vertices[0][1]]
        external_vec2 = [connected_vertices[1][0] - connected_vertices[0][0],
                         connected_vertices[1][1] - connected_vertices[0][1]]
        angle = angle_between_vectors(external_vec1, external_vec2)
        angle_list.append(angle)
        for pivot in range(1, len(connected_vertices) - 1):
            external_vec1 = [connected_vertices[pivot - 1][0] - connected_vertices[pivot][0],
                             connected_vertices[pivot - 1][1] - connected_vertices[pivot][1]]
            external_vec2 = [connected_vertices[pivot + 1][0] - connected_vertices[pivot][0],
                             connected_vertices[pivot + 1][1] - connected_vertices[pivot][1]]
            angle = angle_between_vectors(external_vec1, external_vec2)
            angle_list.append(angle)

        external_vec1 = [connected_vertices[0][0] - connected_vertices[len(connected_vertices) - 1][0],
                         connected_vertices[0][1] - connected_vertices[len(connected_vertices) - 1][1]]
        external_vec2 = [connected_vertices[len(connected_vertices) - 2][0] - connected_vertices[len(connected_vertices) - 1][0],
                         connected_vertices[len(connected_vertices) - 2][1] - connected_vertices[len(connected_vertices) - 1][1]]
        angle = angle_between_vectors(external_vec1, external_vec2)
        angle_list.append(angle)

    return angle_list
