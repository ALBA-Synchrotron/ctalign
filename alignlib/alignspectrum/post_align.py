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

import nxs
import numpy as np
from alignlib.utils import Utils


class PostAlignRemoveJumps():

    def __init__(self, normalizedfile, alignfile):

        self.util_obj = Utils()
        print(alignfile)
        print(normalizedfile)

        self.normalized_nexusfile = nxs.open(normalizedfile, 'r')
        self.aligned_nexusfile = nxs.nxload(alignfile, 'rw')

        #print(self.aligned_nexusfile.tree)

        dataset_name = 'move_vectors'
        try:
            self.aligned_nexusfile.opendata(dataset_name)
            self.mv_vectors = self.aligned_nexusfile.getdata()
            self.aligned_nexusfile.closedata()
        except: 
            print("\nMove vectors could NOT be found.\n")
            return
   
        self.nFrames = 0
        self.numrows = 0
        self.numcols = 0



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


    def move_images(self):

        self.normalized_nexusfile.opengroup('SpecNormalized')
        self.normalized_nexusfile.opendata('spectroscopy_normalized')
        infoshape = self.normalized_nexusfile.getinfo()
        self.nFrames = infoshape[0][0]
        self.numrows = infoshape[0][1]
        self.numcols = infoshape[0][2]


        # Get image to move (and the vector for which it has to be moved) 
        # with method 'images_to_move'. This will have to be done with a for 
        # loop using 'images_to_move'.
        img_num = 10

        image_proj1 = self.util_obj.get_single_image(self.normalized_nexusfile,
                                                     img_num,
                                                     self.numrows,
                                                     self.numcols)

        size_img = np.shape(image_proj1)
        print(size_img)

        slab_offset = [img_num, 0, 0]
        nxsfield = self.aligned_nexusfile['FastAligned']['spec_aligned']

        self.util_obj.store_image_in_hdf(image_proj1, nxsfield, slab_offset)
        print('Alignment of image (%d) corrected\n' % img_num)







