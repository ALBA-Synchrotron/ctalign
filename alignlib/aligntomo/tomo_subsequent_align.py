#!/usr/bin/python

"""
(C) Copyright 2014 Marc Rosanes
(C) Copyright 2014 Miquel Garriga
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
                 num_roi_horizontal, num_roi_vertical):
        super(TomoSubsequentAlign, self).__init__(inputfile,
                                                  roi_select, spec,
                                                  firstimg, printmv,
                                                  num_roi_horizontal,
                                                  num_roi_vertical)

    def tomo_subsequent_alignment(self):

        print('Align tomography projections')

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
                    mv_vector = self.findMvVector(top_left_base,
                                                  top_left_move)

                    roimove_vectors[vert][horiz] = mv_vector
                    total_mv_vector = total_mv_vector + mv_vector

            # First we place the rows and then the columns
            # to be able to apply mvProjections.
            # Add one pixel for drift to rows for drift correction.
            rows = total_mv_vector[1]/(num_rois_vertical*num_rois_horizontal)
            cols = total_mv_vector[0]/(num_rois_vertical*num_rois_horizontal)
            avg_move_vector = [rows, cols]
            self.mv_vector_list.append(avg_move_vector)
            # print(avg_move_vector)
            zeros_img = np.zeros((self.numrows, self.numcols),
                                 dtype='float32')
            proj2_moved = self.mvProjection(zeros_img, proj2,
                                            avg_move_vector)
            proj2 = np.zeros([1, self.numrows, self.numcols],
                             dtype='float32')
            proj2[0] = proj2_moved
            slab_offset = [numimg, 0, 0]
            self.writeImageInHdf5(proj2, self.nxsfied, slab_offset)
            self.proj1 = proj2_moved

            util_obj = Utils()
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
                    mv_vector = self.findMvVector(top_left_base,
                                                  top_left_move)
                    roimove_vectors[vert][horiz] = mv_vector
                    total_mv_vector = total_mv_vector + mv_vector

            # First we place the rows and then the columns
            # to be able to apply mvProjections.
            # Add one pixel for drift to rows for drift correction.
            rows = total_mv_vector[1]/(num_rois_vertical*num_rois_horizontal)
            cols = total_mv_vector[0]/(num_rois_vertical*num_rois_horizontal)
            avg_move_vector = [rows, cols]

            zeros_img = np.zeros((self.numrows, self.numcols),
                                 dtype='float32')
            proj2_moved = self.mvProjection(zeros_img, proj2,
                                            avg_move_vector)
            proj2 = np.zeros([1, self.numrows, self.numcols],
                             dtype='float32')
            proj2[0] = proj2_moved
            slab_offset = [numimg, 0, 0]
            self.writeImageInHdf5(proj2, self.nxsfied, slab_offset)
            self.proj1 = proj2_moved

            self.mv_vector_list.append(avg_move_vector)
            self.counter = util_obj.count(self.counter)

        if self.printmv == 1:
            Utils.print_move(self.mv_vect_filename, self.mv_vector_list)
