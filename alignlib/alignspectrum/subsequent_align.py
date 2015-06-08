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

__all__ = ["SpectrumSubsequentAlign"]

import cv2
import numpy as np
from alignlib.align import Alignment
from alignlib.utils import Utils


class SpectrumSubsequentAlign(Alignment):

    def __init__(self, inputfile, roi_select, spec, firstimg, printmv,
                 num_roi_horizontal, num_roi_vertical):
        super(SpectrumSubsequentAlign, self).__init__(inputfile, roi_select,
                                                      spec, firstimg, printmv,
                                                      num_roi_horizontal,
                                                      num_roi_vertical)

    def spectrum_subsequent_alignment(self):
        print("Align spectroscopic images regarding the precedent image")
        for numimg in range(self.central_img_num+1, self.nFrames):
            # proj2 is the base image in which we will map the template
            image_proj2 = self.get_single_image(numimg)
            proj2 = image_proj2[0, :, :]

            template = self.proj1[self.row_tem_from:self.row_tem_to,
                                  self.col_tem_from:self.col_tem_to]

            # cross-correlations are used in cv2 in function matchTemplate
            # Apply template Matching
            result = cv2.matchTemplate(proj2, template, self.method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            top_left_move = max_loc

            move_vector = self.findMvVector(self.top_left_base, top_left_move)
            mv_vector = [move_vector[1], move_vector[0]]
            self.mv_vector_list.append(mv_vector)
            zeros_img = np.zeros((self.numrows, self.numcols),
                                 dtype='float32')
            proj2_moved = self.mvProjection(zeros_img, proj2, mv_vector)

            proj2 = np.zeros([1, self.numrows, self.numcols],
                             dtype='float32')
            proj2[0] = proj2_moved
            slab_offset = [numimg, 0, 0]
            self.writeImageInHdf5(proj2, self.nxsfied, slab_offset)
            print('Image %d aligned' % numimg)
            self.proj1 = proj2_moved

        if self.printmv == 1:
            Utils.print_move(self.mv_vect_filename, self.mv_vector_list)

        self.input_nexusfile.closedata()
        self.input_nexusfile.closegroup()
        self.input_nexusfile.close()
