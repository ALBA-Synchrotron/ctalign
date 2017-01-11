#!/usr/bin/python

"""
(C) Copyright 2017 Marc Rosanes
The program is distributed under the terms of the
GNU General Public License.

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


class PostAlignRemoveJumps():

    def __init__(self, normalizedfile, alignfile):
        util_obj = Utils()
        print(alignfile)
        print(normalizedfile)

        self.normalized_nexusfile = nxs.open(inputfile, 'r')
        self.aligned_nexusfile = nxs.open(inputfile, 'r')


        if self.spec == 0:
            self.input_nexusfile.opengroup('TomoNormalized')
        elif self.spec == 1:
            self.input_nexusfile.opengroup('SpecNormalized')

        dataset_name = 'mv_vectors'
        try:
            self.input_nexusfile.opendata(dataset_name)
            self.mv_vectors = self.input_nexusfile.getdata()
            self.input_nexusfile.closedata()
        except: 
            print("\nMove vectors could NOT be found.\n")
            return


    # Processing and calculation method. Interpolation and/or extrapolation for
    # the first and the last image if they contain a big jump.
    def images_to_move(self):
        # To Return: A list indicating the images to be moved 
        # (the images that have a big jump after the first alignment) 
        # + the move_vectors by which have to be moved those images.
        # Ex: [  [[3], [5, 6]],   [[105], [-3, 10]], ...]
        return [[[3], [5, 6]],]



    # To move a projection use the method: util_obj.mv_projection
    # example can be found in alignspectrum/subsequent_align:
    # proj2_moved = util_obj.mv_projection(zeros_img, proj2,
    #                                      avg_move_vector)





