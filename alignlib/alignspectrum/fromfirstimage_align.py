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


class SpectrumFromFirstImgAlign(Alignment):

    def __init__(self, inputfile, roi_select, spec, firstimg, printmv,
                 num_roi_horizontal, num_roi_vertical, width, height):
        super(SpectrumFromFirstImgAlign, self).__init__(inputfile, roi_select,
                                                        spec, firstimg, printmv,
                                                        num_roi_horizontal,
                                                        num_roi_vertical,
                                                        width,
                                                        height)
        # Initialize mv_vector_list with first element to [0, 0]
        # because the first image is the reference image in this case.
        self.mv_vector_list.append(np.zeros((2), dtype=np.int))

    # Align using the first image as reference.
    def spectrum_from_first_img_alignment(self):

        #################################################
        #  Get align and store aligned images in HDF5  ##
        #################################################
        self.input_nexusfile.opendata('spectroscopy_normalized')
        reference_img_num = 0

        self.image_proj1 = self.util_obj.get_single_image(self.input_nexusfile,
                                                          reference_img_num,
                                                          self.numrows,
                                                          self.numcols)
        self.proj1_roi_selection = self.image_proj1[0, :, :]

        slab_offset = [reference_img_num, 0, 0]
        self.nxsfield = self.align[self.data_nxs]
        self.util_obj.store_image_in_hdf(self.image_proj1, self.nxsfield, slab_offset)
        print('Initial reference image (%d) stored\n' % reference_img_num)

        self.central_pixel_rows = int(self.numrows / 2)
        self.central_pixel_cols = int(self.numcols / 2)

        # In openCV first we indicate the columns and then the rows.
        # Chose here the number of ROIS: Use odd numbers.
        num_rois_vertical = self.numroivert
        num_rois_horizontal = self.numroihoriz

        if self.user_roi_select == 1:
            # If ROI is selected by the user, a single ROI must be used
            # Do not modify this numbers
            num_rois_vertical = 1
            num_rois_horizontal = 1
            self.roi_selection(self.proj1_roi_selection, spectrum=1)
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

        # Matrix for storing the raw move vectors of all ROIs
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

        print('Initialization completed')

        print("Align spectroscopic images regarding the first image")
        self.counter = 0
        self.proj1 = self.image_proj1[0, :, :]
        # In openCV first we indicate the columns and then the rows.
        for numimg in range(reference_img_num+1, self.nFrames):
            # proj2 is the base image in which we will map the template
            image_proj2 = self.util_obj.get_single_image(self.input_nexusfile,
                                                         numimg,
                                                         self.numrows,
                                                         self.numcols)
            proj2 = image_proj2[0, :, :]

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

            if num_rois_horizontal == 1 and num_rois_vertical == 1:
                avg_mv_vector = mv_vector
            else:
                rmv = roimove_vectors
                avg_mv_vector = self.find_mv_vector_from_many_rois(rmv)

            # First we place the rows and then the columns
            # to be able to apply mv_projection.
            rows = avg_mv_vector[1]
            cols = avg_mv_vector[0]
            avg_move_vector = [rows, cols]

            # Move the projection thanks to the found move vector
            zeros_img = np.zeros((self.numrows, self.numcols), dtype='float32')
            proj2_moved = self.util_obj.mv_projection(zeros_img, proj2, 
                                                 avg_move_vector)
            proj2 = np.zeros([1, self.numrows, self.numcols], dtype='float32')
            proj2[0] = proj2_moved
            slab_offset = [numimg, 0, 0]
            self.util_obj.store_image_in_hdf(proj2, self.nxsfield, slab_offset)

            self.counter = self.util_obj.count(self.counter)
            self.mv_vector_list.append(avg_mv_vector)

        if self.printmv == 1:
            self.util_obj.print_move(self.mv_vect_filename, self.mv_vector_list)

        self.align['move_vectors'] = self.mv_vector_list
        self.align['move_vectors'].write()
       
        self.input_nexusfile.closedata()
        self.input_nexusfile.closegroup()
        self.input_nexusfile.close()


