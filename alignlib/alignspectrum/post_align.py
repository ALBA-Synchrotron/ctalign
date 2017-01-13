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

        self.normalized_nexusfile = nxs.open(normalizedfile, 'r')
        self.aligned_nexusfile = nxs.nxload(alignfile, 'rw')
        self.util_obj = Utils()

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

        # List of images to move and its moving vector
        images_to_mv = [[3, [5, 6]],[20, [-2, -3]], [124, [3, -7]]]
        return images_to_mv


    def move_images(self):

        self.normalized_nexusfile.opengroup('SpecNormalized')
        self.normalized_nexusfile.opendata('spectroscopy_normalized')
        infoshape = self.normalized_nexusfile.getinfo()
        self.nFrames = infoshape[0][0]
        self.numrows = infoshape[0][1]
        self.numcols = infoshape[0][2]

        # Get image to move (and the vector for which it has to be moved) 
        # with method 'images_to_move'.
        images_to_mv = self.images_to_move()
        
        # Move images that contained a big jump (incorrectly aligned), 
        # according to the output of images_to_move method.
        nxsfield = self.aligned_nexusfile['FastAligned']['spec_aligned']
        for i in range(len(images_to_mv)):
            img_num = images_to_mv[i][0]
            slab = self.util_obj.get_single_image(self.normalized_nexusfile,
                                                  img_num,
                                                  self.numrows,
                                                  self.numcols)

            mv_vector = images_to_mv[i][1]
            zeros_img = np.zeros((self.numrows, self.numcols), dtype='float32')
            image = slab[0, :, :]
            image_moved = self.util_obj.mv_projection(zeros_img, 
                                                      image, mv_vector)
            slab_moved = np.zeros([1, self.numrows, self.numcols], 
                                   dtype='float32')
            slab_moved[0] = image_moved
            slab_offset = [img_num, 0, 0]
            self.util_obj.store_image_in_hdf(slab_moved, nxsfield, slab_offset)
            print('Alignment of image (%d) corrected' % img_num)













