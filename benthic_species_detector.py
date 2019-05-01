#!/usr/bin/env python3
from enum import enum
import cv2 as cv
import numpy as np
import stats_utils as stutils
import geometry_utils as geomutils

class BenthicSpeciesDetector:
    """
    This is a class to detect particular shapes in an image corresponding to the Benthic Species outlined in the
        MATE 2019 mission manual.
    """

    class ShapeType(Enum):
        """
        This is an Enum class for specifying the types of shapes the BenthicSpeciesDetector can detect.
        """
        triangle = 1
        rectangle = 2
        square = 3
        circle = 4

    @classmethod
    def _filter_similar_points(cls, point_list, distance_threshold=5):
        """
        Filter a list of points such that no point is within distance_threshold pixels from another.

        :param point_list: a list of points to filter through based on similarity
        :type point_list: list
        :param distance_threshold: the minimum distance two points must be from each other to be considered dissimilar
                                   (5 pixels by default)
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

    @classmethod
    # filter out bounding boxes that are too close to each other or are mostly contained within another bounding box
    def _filter_similar_rects(cls, rects_list, cap, similarity_alpha=0.10, containment_alpha=0.10):
        """
        Filter a list of rects such that none of the rects are too similar within a percent difference similarity_alpha
            or unenclosed by less than a percentage containment_alpha

        :param rects_list: a list of rects to filter by similarity and containment
        :type rects_list: list
        :param cap: the maximum number of rects to process
        :type cap: int
        :param similarity_alpha: (0.1 by default)
        :type similarity_alpha: float
        :param containment_alpha: (0.1 by default)
        :type containment_alpha: float
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
                contains = True if rect_comparison[0] > (1 - containment_alpha) else False
                contained = True if rect_comparison[0] < -(1 - containment_alpha) else False
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

    @classmethod
    def _ellipse_score(cls, shapeimg, corners):
        """
        Test how likely it is that the shape to test is an ellipse.

        :param corners: the corners specifying the shape to test
        :type corners: list
        :return: the confidence that the shape specified by the parameter corners is an ellipse
        """

        height = shapeimg.shape[0]
        width = shapeimg.shape[1]
        if len(corners) < 3:
            blurred = cv.cvtColor(cv.GaussianBlur(shapeimg, (5, 5), 0), cv.COLOR_BGR2GRAY)
            ret, thresholded = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            contours, hierarchy = cv.findContours(thresholded, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
            if contours is not None:
                contour_areas = list(map(lambda cont: np.pi * cv.contourArea(cont), contours))
                if len(contour_areas) > 1:
                    largest_contour_area = max(contour_areas)  # calculate contour area of the largest contour
                else:
                    largest_contour_area = contour_areas[0]
                expected_area = np.pi * width * height  # calculate using height and width of rect as if they are for an ellipse just inside it
                score = 1.0 - (np.abs(expected_area - largest_contour_area) / expected_area)
            else:
                score = 0.0
        else:
            score = 0.0

        return score

    @classmethod
    def _triangle_score(cls, corners):
        """
        Test how likely it is that the shape to test is a triangle.

        :param corners: the corners specifying the shape to test
        :type corners: list
        :return: the confidence that the shape specified by the parameter corners is a triangle
        """

        if len(corners) == 3:
            interior_angles = geomutils.calculate_interior_angles(corners)
            expected_mean = (np.pi) / 3.0
            mean_abs_dev = stutils.mean_abs_deviation(expected_mean, interior_angles)
            score = 1.0 - (mean_abs_dev / expected_mean)
        else:
            score = 0.0

        return score

    @classmethod
    def _rectangle_score(cls, corners):
        """
        Test how likely it is that the shape to test is a rectangle.

        :param corners: the corners specifying the shape to test
        :type corners: list
        :return: the confidence that the shape specified by the parameter corners is a rectangle
        """

        if len(corners) == 4:
            interior_angles = geomutils.calculate_interior_angles(corners)
            expected_mean = np.pi / 2.0
            mean_abs_dev = stutils.mean_abs_deviation(expected_mean, interior_angles)
            score = 1.0 - (np.abs(expected_mean - np.mean(interior_angles)) / expected_mean)
        else:
            score = 0.0

        return score

    @classmethod
    def _square_score(cls, corners):
        """
        Test how likely it is that the shape to test is a square.

        :param corners: the corners specifying the shape to test
        :type corners: list
        :return: the confidence that the shape specified by the parameter corners is a square
        """

        if len(corners) == 4:
            expected_mean_interior = expected_mean_central = np.pi / 2.0
            interior_angles = geomutils.calculate_interior_angles(corners)
            central_angles = geomutils.calculate_central_angles(corners)
            mean_abs_dev_interior = stutils.mean_abs_deviation(expected_mean_interior, interior_angles)
            mean_abs_dev_central = stutils.mean_abs_deviation(expected_mean_central, central_angles)
            score = ((1.0 - (mean_abs_dev_interior / expected_mean_interior)) + (
                        1.0 - (mean_abs_dev_central / expected_mean_central))) / 2.0
        else:
            score = 0.0

        return score

    @classmethod
    def _classify_shape(cls, shapeimg, corners, alpha=0.1):
        """
        Classify the shape bounded in shapeimg and specified by the corners in the list parameter corners.

        :param shapeimg: a matrix of the portion of the image just bounding a particular shape of interest
        :type shapeimg: numpy.ndarray
        :param corners: a list of the corners bounded within the portion of the image
        :type corners: list
        :param alpha: the maximum percent difference from ideal shape for the shape tests (0.1 by default)
        :type alpha: float
        :return: the most probable shape for the provided shape, of the form (shape_type, score) where
                  shape_type is the detected ShapeType and
                  score is the percent confidence that the detected shape is of this type,
                  ((None, 0.0) if no probable shape)
        """

        scores = [(ShapeType.circle, _ellipse_score(shapeimg, corners)),
                  (shapeType.triangle, _triangle_score(corners)),
                  (ShapeType.rectangle, _rectangle_score(corners)),
                  (ShapeType.square, _square_score(corners))]

        filtered_scores = list(filter(lambda score_info: score_info[1] >= 1 - alpha, scores))
        if len(filtered_scores) != 0:
            filtered_scores.sort(key=lambda score_info: score_info[1])

            if (filtered_scores[0][0] == ShapeType.rectangle) and (filtered_scores[0][1]):
                square_sufficient = False
                square_index = -1
                for index in range(0, len(filtered_scores)):
                    if (filtered_scores[index][0] == ShapeType.square) and (filtered_scores[index][1] >= 1 - alpha):
                        square_sufficient = True
                        square_index = index
                        break
                if square_sufficient:
                    most_probable = filtered_scores[square_index]
                else:
                    most_probable = filtered_scores[0]
            else:
                most_probable = filtered_scores[0]
        else:
            most_probable = None

        return most_probable

    @classmethod
    def process(cls, image, image_scaling_factor=0.5, debug=False, alphas=(0.10, 0.10)):
        """
        Process an image to detect shapes present in it.

        :param image: a matrix image to be processed for shape detection
        :type image: numpy.ndarray
        :param image_scaling_factor: a positive floating point number which scales the height and width of the image
                                     before processing (0.5 by default)
        :type image_scaling_factor: float
        :param debug: a boolean which determines whether debug information should be displayed (intermediate region
                      proposals, corners, and shapes) (False by default)
        :type debug: bool
        :param alphas: a tuple of the form (rect_alpha, shape_alpha), where rect_alpha is the maximum percent dissimilarity
                       for rectangle similarity and shape_alpha is the maximum percent difference from ideal shape for the
                       shape tests ((0.1, 0.1) by default)
        :type alphas: tuple
        :return: a list of detected shapes of the form ((x, y, w, h), (shape_type, confidence)) where
                  (x, y, w, h) is the bounding rectangle of the shape,
                  shape_type is the ShapeType of the shape,
                  and confidence is the percent confidence of for the detected shape
        """

        cv.setUseOptimized(True)
        cv.setNumThreads(4)

        # resize the image proportionally
        resized_height = image_scaling_factor * image.shape[0]
        resized_width = int(resized_height * (image.shape[1] / image.shape[0]))
        image = cv.resize(image, (resized_width, resized_height))

        # create the SelectiveSearchSegmentation object, set the image to be segmented, and set to use the
        # lower-recall-faster-runtime implementation of Selective Search
        selective_search = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        selective_search.setBaseImage(image)
        selective_search.switchToSelectiveSearchFast()

        # segment the image into potential object regions with bounding boxes
        rects = selective_search.process()
        # remove rects that include too much of the image
        rects = np.array(list(filter(lambda current_rect: ((current_rect[2] < 0.90 * resized_width)
                                                   and
                                                   (current_rect[3] < 0.90 * resized_height)),
                                     rects)
                              )
                         )

        if debug:
            print('Total number of region proposals: {0}'.format(len(rects)))
            visible_rects_cap = 700

        # filter the bounding boxes by similarity and enclosure
        rects_cap = 150
        rect_alpha = alphas[0]  # filter by similarity >= 90%

        # filter rects
        filtered_rects = _filter_similar_rects(rects, rects_cap, rect_alpha)
        if debug:
            print("Number of rects after similarity filtering: {0}".format(len(filtered_rects)))
            print("Filtered rects: {0}".format(filtered_rects))

        # find Harris corners
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        dst = cv.dilate(dst, None)
        ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
        corners = _filter_similar_points(corners, distance_threshold=10)

        corners_by_rect = []
        for rect_index in range(0, len(filtered_rects)):
            corners_by_rect.append([])
            for corner_index in range(0, len(res)):
                if pointEnclosed(res[corner_index, 0:], filtered_rects[rect_index]):
                    corners_by_rect[rect_index].append(res[corner_index, 0:])

        # filter out rects that have too many corners to avoid unnecessary processing
        filtered_rects = list(filtered_rects)
        rect_index = 0
        while rect_index < len(filtered_rects):
            if len(corners_by_rect[rect_index]) > 4:
                filtered_rects.pop(rect_index)
                corners_by_rect.pop(rect_index)
            else:
                rect_index += 1

        # classify and score detected shapes
        shapes_list = []
        shape_alpha = alphas[1]
        for rect_index in range(0, len(filtered_rects)):
            (x, y, w, h) = filtered_rects[rect_index]
            shape_score = _classify_shape(image[y:y + h, x:x + w], corners_by_rect[rect_index], alpha=shape_alpha)
            if debug:
                print("Rect Index: {0}, Location: {1}\nCorners: {2}\nShape score: {3}\n\n".format(
                    str(rect_index),
                    str((x, y)),
                    str(corners_by_rect[rect_index]), str(shape_score)))
            shapes_list.append((filtered_rects[rect_index], shape_score))

        if debug:
            while True:
                prelim_out_image = image.copy()
                filtered_out_image = image.copy()
                corners_out_image = image.copy()

                for i, rect in enumerate(rects):
                    if i < visible_rects_cap:
                        x, y, w, h = rect
                        cv.rectangle(prelim_out_image, (x, y), (x + w, y + h), (0, 255, 0), 1, cv.LINE_AA)
                    else:
                        break

                for i, filtered_rect in enumerate(filtered_rects):
                    if i < visible_rects_cap:
                        x, y, w, h = filtered_rect
                        cv.rectangle(filtered_out_image, (x, y), (x + w, y + h),
                                     (int(x % 255), int(y % 255), int(w % 255)),
                                     1, cv.LINE_AA)
                    else:
                        break

                # Prepare to draw the corners
                res = corners
                res = np.int0(res)

                corners_out_image[res[:, 1], res[:, 0]] = [0, 0, 255]
                corners_out_image[res[:, 1] - 1, res[:, 0]] = [0, 0, 255]
                corners_out_image[res[:, 1] + 1, res[:, 0]] = [0, 0, 255]
                corners_out_image[res[:, 1], res[:, 0] - 1] = [0, 0, 255]
                corners_out_image[res[:, 1], res[:, 0] + 1] = [0, 0, 255]

                print("Corners grouped by bounding rectangle: {0}".format(corners_by_rect))

                # draw and display detected shapes with type and confidence score
                final_image = image.copy()
                for shape in shapes_list:
                    if shape[1] is not None:
                        shape_type = shape[1][0]
                        if shape_type == shapeType.triangle:
                            text = "Triangle: {0}%".format(np.round(shape[1][1] * 100, 1))
                        elif shape_type == shapeType.square:
                            text = "Square: {0}%".format(np.round(shape[1][1] * 100, 1))
                        elif shape_type == shapeType.rectangle:
                            text = "Rectangle: {0}%".format(np.round(shape[1][1] * 100, 1))
                        elif shape_type == shapeType.circle:
                            text = "Circle: {0}%".format(np.round(shape[1][1] * 100, 1))
                        cv.putText(final_image, text, (shape[0][0], shape[0][1] + shape[0][3] + 10),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 0, 0))

                cv.imshow("Preliminary Rects", prelim_out_image)
                cv.imshow("Filtered Rects", filtered_out_image)
                cv.imshow("Corners", corners_out_image)
                cv.imshow("Labeled Shapes", final_image)

                key = cv.waitKey(0) & 0xFF
                if key == 113:  # if key = 'q'
                    break

            cv.destroyAllWindows()

