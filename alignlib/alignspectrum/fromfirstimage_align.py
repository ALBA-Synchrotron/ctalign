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


class SpectrumFromFirstImgAlign(Alignment):

    def __init__(self, inputfile, roi_select, spec, firstimg, printmv,
                 num_roi_horizontal, num_roi_vertical, width, height):
        super(SpectrumFromFirstImgAlign, self).__init__(inputfile, roi_select,
                                                        spec, firstimg, printmv,
                                                        num_roi_horizontal,
                                                        num_roi_vertical,
                                                        width,
                                                        height)

    # Align using the first image as reference.
    def spectrum_from_first_img_alignment(self):

        #################################################
        #  Get align and store aligned images in HDF5  ##
        #################################################
        self.input_nexusfile.opendata('spectroscopy_normalized')
        self.central_img_num = 0

        self.image_proj1 = self.get_single_image(self.central_img_num)
        self.proj1 = self.image_proj1[0, :, :]

        slab_offset = [self.central_img_num, 0, 0]
        self.nxsfield = self.align[self.data_nxs]
        self.store_image_in_hdf(self.image_proj1, self.nxsfield, slab_offset)
        print('Initial reference image (%d) stored\n' % self.central_img_num)

        self.central_pixel_rows = int(self.numrows / 2)
        self.central_pixel_cols = int(self.numcols / 2)

        self.row_tem_from = self.central_pixel_rows - self.height_tem / 2
        self.row_tem_to = self.central_pixel_rows + self.height_tem / 2
        self.col_tem_from = self.central_pixel_cols - self.width_tem / 2
        self.col_tem_to = self.central_pixel_cols + self.width_tem / 2

        # In openCV first we indicate the columns and then the rows.
        self.top_left_base = (self.col_tem_from, self.row_tem_from)
        print('Initialization completed')

        print("Align spectroscopic images regarding the first image")
        util_obj = Utils()
        self.counter = 0
        template = self.proj1[self.row_tem_from:self.row_tem_to,
                              self.col_tem_from:self.col_tem_to]

        for numimg in range(self.central_img_num+1, self.nFrames):
            # proj2 is the base image in which we will map the template
            image_proj2 = self.get_single_image(numimg)
            proj2 = image_proj2[0, :, :]

            # cross-correlations are used in cv2 in function matchTemplate
            # Apply template Matching
            result = cv2.matchTemplate(proj2, template, self.method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            top_left_move = max_loc
            move_vector = self.find_mv_vector(self.top_left_base, top_left_move)
            mv_vector = [move_vector[1], move_vector[0]]
            self.mv_vector_list.append(mv_vector)
            zeros_img = np.zeros((self.numrows, self.numcols),
                                 dtype='float32')
            proj2_moved = self.mv_projection(zeros_img, proj2, mv_vector)

            proj2 = np.zeros([1, self.numrows, self.numcols],
                             dtype='float32')
            proj2[0] = proj2_moved
            slab_offset = [numimg, 0, 0]
            self.store_image_in_hdf(proj2, self.nxsfield, slab_offset)
            self.counter = util_obj.count(self.counter)

        if self.printmv == 1:
            util_obj.print_move(self.mv_vect_filename, self.mv_vector_list)

        self.input_nexusfile.closedata()
        self.input_nexusfile.closegroup()
        self.input_nexusfile.close()
