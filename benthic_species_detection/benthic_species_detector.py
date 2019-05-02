#!/usr/bin/env python3
from enum import enum
import cv2 as cv
import numpy as np
import stats_utils as stutils
import geometry_utils as geomutils

class BenthicSpeciesDetector:
    """
    This is a class to detect particular shapes in an image corresponding to the Benthic Species outlined in the
        MATE 2019 ROV competition mission manual.
    """

    def __init__(self, rect_similarity_alpha=0.1, rect_enclosure_alpha=0.1, shape_alpha=0.1, corner_min_distance=10):
        """
        Construct an instance of the BenthicSpeciesDetector.

        :param rect_similarity_alpha: the minimum percent difference required for the rectangles to be considered
                                      dissimilar (0.1 by default)
        :type rect_similarity_alpha: float
        :param rect_enclosure_alpha: the minimum percent unenclosed required for a rectangle to be considered
                                       unenclosed by another rectangle (0.1 by default)
        :type rect_enclosure_alpha: float
        :param shape_alpha: the maximum percent difference from ideal shape characteristics for the shape tests
                            (0.1 by default)
        :type shape_alpha: float
        :param corner_min_distance: the minimum distance two points must be from each other to be considered dissimilar
                                   (10 pixels by default)
        :type corner_min_distance: int
        """

        self.rect_similarity_alpha = rect_similarity_alpha
        self.rect_enclosure_alpha = rect_enclosure_alpha
        self.shape_alpha = shape_alpha
        self.corner_min_distance = corner_min_distance

    class ShapeType(Enum):
        """
        This is an Enum class for specifying the types of shapes the BenthicSpeciesDetector can detect.
        """
        triangle = 1
        rectangle = 2
        square = 3
        circle = 4

    @staticmethod
    def _ellipse_score(shapeimg, corners):
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

    @staticmethod
    def _triangle_score(corners):
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

    @staticmethod
    def _rectangle_score(corners):
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

    @staticmethod
    def _square_score(corners):
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

    @staticmethod
    def _classify_shape(shapeimg, corners, shape_alpha=0.1):
        """
        Classify the shape bounded in shapeimg and specified by the corners in the list parameter corners.

        :param shapeimg: a matrix of the portion of the image just bounding a particular shape of interest
        :type shapeimg: numpy.ndarray
        :param corners: a list of the corners bounded within the portion of the image
        :type corners: list
        :param shape_alpha: the maximum percent difference from ideal shape characteristics for the shape tests
                            (0.1 by default)
        :type shape_alpha: float
        :return: the most probable shape for the provided shape, of the form (shape_type, score) where
                  shape_type is the detected ShapeType and
                  score is the percent confidence that the detected shape is of this type,
                  ((None, 0.0) if no probable shape)
        """

        scores = [(ShapeType.circle, _ellipse_score(shapeimg, corners)),
                  (shapeType.triangle, _triangle_score(corners)),
                  (ShapeType.rectangle, _rectangle_score(corners)),
                  (ShapeType.square, _square_score(corners))]

        filtered_scores = list(filter(lambda score_info: score_info[1] >= 1 - shape_alpha, scores))
        if len(filtered_scores) != 0:
            filtered_scores.sort(key=lambda score_info: score_info[1])

            if (filtered_scores[0][0] == ShapeType.rectangle) and (filtered_scores[0][1]):
                square_sufficient = False
                square_index = -1
                for index in range(0, len(filtered_scores)):
                    if (filtered_scores[index][0] == ShapeType.square) and (filtered_scores[index][1] >= 1 - shape_alpha):
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

    def process(self, image, image_scaling_factor=0.5, debug=False):
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

        # filter rects
        filtered_rects = geomutils.filter_similar_rects(rects, rects_cap, similarity_alpha=self.rect_similarity_alpha,
                                               enclosure_alpha=self.rect_enclosure_alpha)
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
        corners = geomutils.filter_similar_points(corners, distance_threshold=self.corner_min_distance)

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
        for rect_index in range(0, len(filtered_rects)):
            (x, y, w, h) = filtered_rects[rect_index]
            shape_score = _classify_shape(image[y:y + h, x:x + w], corners_by_rect[rect_index],
                                          shape_alpha=self.shape_alpha)
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

        return list(map(lambda shape_info: (geomutils.recontextualize_rect(shape_info[0], resized_width, resized_height,
                                                                           1 / image_scaling_factor,
                                                                           1 / image_scaling_factor), shape_info[1]),
                                                                           shapes_list))

