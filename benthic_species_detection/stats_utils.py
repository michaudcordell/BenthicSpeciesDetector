#!/usr/bin/env python3

"""
This module contains methods for certain statistical calculations needed by the Benthic Species Detector.
"""

def find_2_mins(number_list):
    """
    Determine the two smallest values in number_list, as well as their indices.

    :param number_list: a list of numbers from which to determine the two minimums
    :type number_list: list
    :return: the two smallest values and their indices in the form ((min1, min1_index),(min2, min2_index))
    """

    min1 = float('inf')
    min1_index = -1
    min2 = float('inf')
    min2_index = -1

    for index in range(0, len(number_list)):
        if number_list[index] <= min1:
            min2 = min1
            min2_index = min1_index
            min1 = number_list[index]
            min1_index = index
        elif number_list[index] < min2:
            min2 = number_list[index]
            min2_index = index

    if min1_index == -1:
        min1 = None
    if min2_index == -1:
        min2 = None

    return ((min1, min1_index),(min2, min2_index))

def mean_abs_deviation(mean, observations):
    """
    Calculate the mean absolute deviation of a set of observations with respect to a mean.

    :param mean: an expected mean
    :type mean: float
    :param observations: a list of observations from which to calculate deviation from the mean
    :type observations: list
    :return: the mean absolute deviation of observations with respect to the mean
    """

    mean_abs_dev = sum(list(map(lambda obs: abs(obs - mean), observations))) / len(observations)
    return mean_abs_dev

def centroid(points_list):
    """
    Calculate the centroid of the points in points_list.

    :param points_list: a list of points of the form (x, y)
    :type points_list: list
    :returns: the centroid of the list of points
    """

    calculated_centroid = np.array([0 * i for i in range(0, len(points_list[0]))])

    for axis in range(0, len(points_list[0])):
        compsum = 0
        for point_index in range(0, len(points_list)):
            compsum += points_list[point_index][axis]
        comp = float(compsum) / len(points_list)
        calculated_centroid[axis] = comp

    return calculated_centroid