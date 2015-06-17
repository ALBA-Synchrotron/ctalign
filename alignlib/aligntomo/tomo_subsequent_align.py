#!/usr/bin/python

"""
(C) Copyright 2014 Marc Rosanes
The program is distributed under the terms of the
GNU General Public License (or the Lesser GPL).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import cv2
import numpy as np
from alignlib.align import Alignment
from alignlib.utils import Utils


class TomoSubsequentAlign(Alignment):

    def __init__(self, inputfile, roi_select, spec, firstimg, printmv,
                 num_roi_horizontal, num_roi_vertical, width, height):
        super(TomoSubsequentAlign, self).__init__(inputfile,
                                                  roi_select, spec,
                                                  firstimg, printmv,
                                                  num_roi_horizontal,
                                                  num_roi_vertical,
                                                  width,
                                                  height)

    def find_mv_vector_from_many_rois(self, move_vectors_matrix):
        """Find move vector from many ROIs.
        Input argument: Matrix of move vectors
        Return variable: Average move vector of relevant move vectors.
        A move vector is considered to be relevant if it is similar to
        the other move vectors in a certain degree. It it is to distant from
        other move vectors it is not taken into account for being considered
        in the calculation of the returned average move vector."""

        # An error has been observed, if the thresholds are to high and
        # no vectors are stored after filtering, a problem exist. It
        # should be tried to lower the thresholds if no vectors are stored,
        # In order to at least find one of the vectors.

        # Useful for sorting list of lists based on second list element.
        # Our list of lists, is a list of vectors.

        from operator import itemgetter

        # Threshold in pixels for knowing if we increment or not the counter
        # indicating how many move vectors are similar to the processed move
        # vector currently being processed.
        pixels_threshold = 5

        # Threshold for deciding if we keep or not a given move vector,
        # according in how many similar vectors to itself exists.
        # Expressed in tant_per_one.
        threshold_similarity = 0.5

        mv_list = list_of_mv_vectors_for_processing = []

        for vert in range(self.numroivert):
            for horiz in range(self.numroihoriz):
                #print(move_vectors_matrix[vert][horiz])

                mv_list.append(list(move_vectors_matrix[vert][horiz]))

        # We sort the mv_vectors list by its first element
        sorted_mv_list = sorted(mv_list)

        print("\nSorted full vector list")
        print(sorted_mv_list)

        len_mv_list = len(mv_list)
        threshold_similar_vectors = threshold_similarity * len_mv_list

        # Empty list that will contain the number of vectors similar between
        # them, based by the similarity between first vector elements.
        # We will use a given tolerance (in number of pixels to be moved),
        # in order to decided if the vectors are similar or not, based on the
        # first element.
        counts_of_similar_vectors = len_mv_list*[0]

        for i in range(len(sorted_mv_list)):
            vector = sorted_mv_list[i]
            counts = 0
            for j in range(len(sorted_mv_list)):
                if i != j:
                    # Vector to compare: vector_to_cmp
                    vector_to_cmp = sorted_mv_list[j]
                    if abs(vector_to_cmp[0] - vector[0]) <= pixels_threshold:
                        counts += 1
            counts_of_similar_vectors[i] = counts

        # move vectors after having being filtered by the similarity of the
        # first elements.
        first_elem_filtered_mv_vectors = []
        for i in range(len(sorted_mv_list)):
            counts = counts_of_similar_vectors[i]
            if counts >= threshold_similar_vectors:
                first_elem_filtered_mv_vectors.append(sorted_mv_list[i])

        # If there are not at least 2 vectors in the list repeat the procdure
        # for a lower threshold of number of similar vectors.
        for i in range(8):
            if len(first_elem_filtered_mv_vectors) < 2:
                first_elem_filtered_mv_vectors = []
                threshold_similarity = threshold_similarity - i*0.05
                threshold_similar_vectors = threshold_similarity * len_mv_list
                for i in range(len(sorted_mv_list)):
                    counts = counts_of_similar_vectors[i]
                    if counts >= threshold_similar_vectors:
                        first_elem_filtered_mv_vectors.append(sorted_mv_list[i])
            else:
                break

        print("\nSorted first filtering vector list")
        print(first_elem_filtered_mv_vectors)

        # After filtering the vectors by its first element, a new filter will
        # be applied, filtering the move vectors by its second element.
        sorted_mv_list = sorted(first_elem_filtered_mv_vectors,
                                key=itemgetter(1))

        len_filtered_vectors = len(sorted_mv_list)
        threshold_similarity = 0.5
        threshold_similar_vectors = threshold_similarity * len_filtered_vectors
        counts_of_similar_vectors = len_filtered_vectors*[0]

        for i in range(len(sorted_mv_list)):
            vector = sorted_mv_list[i]
            counts = 0
            for j in range(len(sorted_mv_list)):
                if i != j:
                    # Vector to compare: vector_to_cmp
                    vector_to_cmp = sorted_mv_list[j]
                    if abs(vector_to_cmp[1] - vector[1]) <= pixels_threshold:
                        counts += 1
            counts_of_similar_vectors[i] = counts
        # move vectors after having being filtered by the similarity of the
        # first elements.
        second_elem_filtered_mv_vectors = []
        for i in range(len(sorted_mv_list)):
            counts = counts_of_similar_vectors[i]
            if counts >= threshold_similar_vectors:
                second_elem_filtered_mv_vectors.append(sorted_mv_list[i])

        # If there are not at least 2 vectors in the list repeat the procdure
        # for a lower threshold of number of similar vectors.
        for i in range(8):
            if len(second_elem_filtered_mv_vectors) < 2:
                second_elem_filtered_mv_vectors = []
                threshold_similarity = threshold_similarity - i*0.05
                threshold_similar_vectors = (threshold_similarity *
                                             len_filtered_vectors)
                for i in range(len(sorted_mv_list)):
                    counts = counts_of_similar_vectors[i]
                    if counts >= threshold_similar_vectors:
                        second_elem_filtered_mv_vectors.append(sorted_mv_list[i])
            else:
                break

        print("\nSorted second filtering vector list")
        print(second_elem_filtered_mv_vectors)

        # Finally, the average of the remaining move vectors after the
        # filtering, is calculated.
        mv_vectors = second_elem_filtered_mv_vectors
        len_filtered_mv_vect = len(second_elem_filtered_mv_vectors)
        np_total_mv_vector = np.array([0,0])
        for i in range(len_filtered_mv_vect):
            np_total_mv_vector += np.array(mv_vectors[i])

        # average move vector after filtering
        avg_mv_vector = np_total_mv_vector/float(len_filtered_mv_vect)
        avg_mv_vector = np.around(avg_mv_vector)
        avg_mv_vector = avg_mv_vector.astype(int)
        print(avg_mv_vector)

        return avg_mv_vector

    def tomo_subsequent_alignment(self):

        #################################################
        #  Get align and store aligned images in HDF5  ##
        #################################################

        self.input_nexusfile.opendata('TomoNormalized')
        self.central_img_num = int(self.nFrames) / 2

        self.image_proj1 = self.get_single_image(self.central_img_num)
        self.proj1_roi_selection = self.image_proj1[0, :, :]
        # cv2.imshow('proj1',proj1)
        # cv2.waitKey(0)
        slab_offset = [self.central_img_num, 0, 0]
        self.nxsfield = self.align[self.data_nxs]
        self.store_image_in_hdf(self.image_proj1, self.nxsfield, slab_offset)
        print('Initial reference image (%d) stored' % self.central_img_num)

        self.central_pixel_rows = int(self.numrows / 2)
        self.central_pixel_cols = int(self.numcols / 2)

        print('Initialization completed\n')
        print('Align tomography projections')

        # In openCV first we indicate the columns and then the rows.
        # Chose here the number of ROIS: Use odd numbers.
        num_rois_vertical = self.numroivert
        num_rois_horizontal = self.numroihoriz

        if self.user_roi_select == 1:
            # If ROI is selected by the user, a single ROI must be used
            # Do not modify this numbers
            __num_rois_vertical = 1
            __num_rois_horizontal = 1
            num_rois_vertical = __num_rois_vertical
            num_rois_horizontal = __num_rois_horizontal
            self.roi_selection(self.proj1_roi_selection)
            self.width_tem = self.roi_points[1][0] - self.roi_points[0][0]
            self.height_tem = self.roi_points[1][1] - self.roi_points[0][1]
            # Rows: Second coordinate from first point (cv2)
            origin_pixel_rows = self.roi_points[0][1]
            # Cols: First coordinate from second point (cv2)
            origin_pixel_cols = self.roi_points[0][0]
        else:
            origin_pixel_rows = self.central_pixel_rows - self.height_tem/2
            origin_pixel_cols = self.central_pixel_cols - self.width_tem/2

        # Offset in pixels for finding the different ROIs (templates)
        offset_vertical = 2*self.height_tem/3
        offset_horizontal = 2*self.width_tem/3

        # Use values of pixels divisible by to for width_tem and height_tem
        w = self.width_tem
        h = self.height_tem

        roimove_vectors = np.zeros((num_rois_vertical,
                                    num_rois_horizontal, 2), dtype=int)
        # Template zones
        col_tem_from = []
        row_tem_from = []

        for c_vertical in range(-num_rois_vertical/2+1,
                                num_rois_vertical/2+1):
            row_tem_from.append(origin_pixel_rows +
                                c_vertical * offset_vertical)

        for c_horiz in range(-num_rois_horizontal/2+1,
                             num_rois_horizontal/2+1):
            col_tem_from.append(origin_pixel_cols +
                                c_horiz * offset_horizontal)

        util_obj = Utils()
        self.counter = 0
        # From the middle image to the last image going fordward.
        # proj1 is the image from which we extract the template
        self.proj1 = self.image_proj1[0, :, :]
        for numimg in range(self.central_img_num+1, self.nFrames):
            # proj2 is the image in which we will map the template issued
            # from self.proj1
            image_proj2 = self.get_single_image(numimg)
            proj2 = image_proj2[0, :, :]

            total_mv_vector = [0, 0]

            for vert in range(num_rois_vertical):
                for horiz in range(num_rois_horizontal):
                    template = self.proj1[row_tem_from[vert]:
                                          row_tem_from[vert]+h,
                                          col_tem_from[horiz]:
                                          col_tem_from[horiz]+w]

                    # Apply template Matching: cross-correlation is used
                    # by using function matchTemplate from cv2.
                    result = cv2.matchTemplate(proj2, template, self.method)
                    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)

                    # Every ROI shall have a different top_left_base
                    # In openCV first indicate the columns and then the rows.
                    top_left_base = (col_tem_from[horiz],
                                     row_tem_from[vert])

                    top_left_move = (max_loc[0], max_loc[1])
                    mv_vector = self.find_mv_vector(top_left_base,
                                                    top_left_move)

                    roimove_vectors[vert][horiz] = mv_vector
                    #total_mv_vector = total_mv_vector + mv_vector

            if num_rois_horizontal == 1 and num_rois_vertical == 1:
                avg_mv_vector = mv_vector
            else:
                rmv = roimove_vectors
                avg_mv_vector = self.find_mv_vector_from_many_rois(rmv)

            # First we place the rows and then the columns
            # to be able to apply mv_projection.
            # Add one pixel for drift to rows for drift correction.
            # rows = total_mv_vector[1]/(num_rois_vertical*num_rois_horizontal)
            # cols = total_mv_vector[0]/(num_rois_vertical*num_rois_horizontal)

            rows = avg_mv_vector[1]
            cols = avg_mv_vector[0]
            avg_move_vector = [rows, cols]
            self.mv_vector_list.append(avg_move_vector)
            # print(avg_move_vector)
            zeros_img = np.zeros((self.numrows, self.numcols),
                                 dtype='float32')
            proj2_moved = self.mv_projection(zeros_img, proj2,
                                            avg_move_vector)
            proj2 = np.zeros([1, self.numrows, self.numcols],
                             dtype='float32')
            proj2[0] = proj2_moved
            slab_offset = [numimg, 0, 0]
            self.store_image_in_hdf(proj2, self.nxsfield, slab_offset)
            self.proj1 = proj2_moved
            self.counter = util_obj.count(self.counter)

        # From the middle image to the first image going backward.
        self.proj1 = self.image_proj1[0, :, :]
        for numimg in xrange(self.central_img_num-1, -1, -1):
            # proj2 is the image in which we will map the template issued
            # from proj1
            image_proj2 = self.get_single_image(numimg)
            proj2 = image_proj2[0, :, :]

            total_mv_vector = [0, 0]
            for vert in range(num_rois_vertical):
                for horiz in range(num_rois_horizontal):
                    template = self.proj1[row_tem_from[vert]:
                                          row_tem_from[vert]+h,
                                          col_tem_from[horiz]:
                                          col_tem_from[horiz]+w]

                    # Apply template Matching: cross-correlation is used
                    # by using function matchTemplate from cv2.
                    result = cv2.matchTemplate(proj2, template, self.method)
                    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)

                    # Every ROI shall have a different top_left_base
                    # In openCV first indicate the columns and then the rows.
                    top_left_base = (col_tem_from[horiz],
                                     row_tem_from[vert])
                    top_left_move = (max_loc[0], max_loc[1])
                    mv_vector = self.find_mv_vector(top_left_base,
                                                    top_left_move)

                    # Storing the ROI move vectors in a matrix
                    roimove_vectors[vert][horiz] = mv_vector

                    #total_mv_vector = total_mv_vector + mv_vector

            if num_rois_horizontal == 1 and num_rois_vertical == 1:
                avg_mv_vector = mv_vector
            else:
                rmv = roimove_vectors
                avg_mv_vector = self.find_mv_vector_from_many_rois(rmv)

            # First we place the rows and then the columns
            # to be able to apply mv_projections.
            # Add one pixel for drift to rows for drift correction.
            # rows = total_mv_vector[1]/(num_rois_vertical*num_rois_horizontal)
            # cols = total_mv_vector[0]/(num_rois_vertical*num_rois_horizontal)
            rows = avg_mv_vector[1]
            cols = avg_mv_vector[0]
            avg_move_vector = [rows, cols]

            zeros_img = np.zeros((self.numrows, self.numcols),
                                 dtype='float32')
            proj2_moved = self.mv_projection(zeros_img, proj2,
                                            avg_move_vector)
            proj2 = np.zeros([1, self.numrows, self.numcols],
                             dtype='float32')
            proj2[0] = proj2_moved
            slab_offset = [numimg, 0, 0]
            self.store_image_in_hdf(proj2, self.nxsfield, slab_offset)
            self.proj1 = proj2_moved

            self.mv_vector_list.append(avg_move_vector)
            self.counter = util_obj.count(self.counter)

        if self.printmv == 1:
            util_obj.print_move(self.mv_vect_filename, self.mv_vector_list)

        self.input_nexusfile.closedata()
        self.input_nexusfile.closegroup()
        self.input_nexusfile.close()



