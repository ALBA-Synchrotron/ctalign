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

        try:
            vectors = self.aligned_nexusfile['FastAligned']['move_vectors']
            self.move_vectors = vectors
        except Exception: 
            print("\nMove vectors could NOT be found.\n")
            raise
   
        self.nFrames = 0
        self.numrows = 0
        self.numcols = 0


    # Processing and calculation method. Interpolation and/or extrapolation for
    # the first and the last image if they contain a big jump.
    def images_to_move(self):
        # Return: A list indicating the images to be moved 
        # (the images that have a big jump after the first alignment) 
        # + the move_vectors by which have to be moved those images.
        # List of images to move and its corrected moving vector
        # Ex:[[3, [121, 122]],[20, [-20, -30]], [124, [30, -70]]]
        
        
        ## With the two components of the moving vectors independently
        ## do a linear regression of those two lines. Do the subtraction
        ## between the line of the linear regression and the real data.
        ## Do it for one data component, and for the other component.
        ## Then find the outliers.

        rows = np.array(self.move_vectors[:, 0])
        columns = np.array(self.move_vectors[:, 1])
        len_vector = np.shape(self.move_vectors)[0]
        x = np.linspace(0, len_vector-1, len_vector, dtype=int)


        ###########################################
        # Find some indexes of jumped images thanks to the 
        # processing of rows data:
        z_rows = np.polyfit(x, rows, 1)
        p_rows = np.poly1d(z_rows)
        rows_interp = p_rows(x)
        difference_interp_from_data_rows = rows - rows_interp

        # Mean and standard deviation of the difference between 
        # the actual data of the row vectors and its linear regression.
        # The starndard deviation of vectors of difference 
        # (for rows and for cols) is calculated. This gives us the
        # an idea of the threshold when doing the subtraction between each value
        # of interpolation and data vector (for rows and for columns). 
        # If the value of this subtraction is higher than the choosen threshold,
        # it means that a jump exist in such moving vector.
        mean_diff_rows = np.mean(difference_interp_from_data_rows)
        std_diff_rows = np.std(difference_interp_from_data_rows)

        # We leave an offset of pixels. Jumps smaller than offset won't be 
        # corrected: they are not considered as big jumps.
        offset = 6

        # Thanks to the data of the moving rows some indexes are obtained.
        abs_diff_rows = np.abs(difference_interp_from_data_rows)
        idx_images_to_correct_rows = [i for i,v in enumerate(abs_diff_rows) if 
                                  v > (offset + mean_diff_rows + std_diff_rows)]

        ###########################################
        # Find the rest of indexes of jumped images thanks to the 
        # processing of columns data:
        z_cols = np.polyfit(x, columns, 1)
        p_cols = np.poly1d(z_cols)
        cols_interp = p_cols(x)
        difference_interp_from_data_cols = columns - cols_interp

        # Mean and standard deviation of the difference between 
        # the actual data of the column vectors and its linear regression.
        mean_diff_cols = np.mean(difference_interp_from_data_cols)
        std_diff_cols = np.std(difference_interp_from_data_cols)

        # We leave an offset of pixels. Jumps smaller than offset won't be 
        # corrected: they are not considered as big jumps.
        offset = 5

        # Thanks to the data of the moving columns some indexes are obtained.
        abs_diff_cols = np.abs(difference_interp_from_data_cols)
        idx_images_to_correct_cols = [i for i,v in enumerate(abs_diff_cols) if 
                                 v > (offset + mean_diff_cols + std_diff_cols)]

        idx_rows = idx_images_to_correct_rows
        idx_cols = idx_images_to_correct_cols
        idx_images_to_correct_move = sorted(list(set(idx_rows)|set(idx_cols)))

        print("The following images will be corrected:")        
        print(idx_images_to_correct_move)
        print("\n")

        # Looking for the new corrected moving vectors.
        # First, recalculate the regression line after correcting in a 
        # raw manner the jumping vectors.
        for i in idx_images_to_correct_move:
            if i != 0:
                rows[i] = rows[i-1]
                columns[i] = columns[i-1]

        z_rows = np.polyfit(x, rows, 1)
        p_rows = np.poly1d(z_rows)
        rows_interp = p_rows(x)

        z_cols = np.polyfit(x, columns, 1)
        p_cols = np.poly1d(z_cols)
        cols_interp = p_cols(x)

        # Then look for the moving corrected values for the jumping images.
        images_to_mv = []
        for i in idx_images_to_correct_move:
            if i != 0:
                rows[i] = int(rows_interp[i])
                columns[i] = int(cols_interp[i])
                images_to_mv.append([i, [rows[i], columns[i]]])
        
        #images_to_mv = [[3, [121, 122]],[2, [-20, -30]], [124, [30, -70]]]
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
        vects_field = self.aligned_nexusfile['FastAligned']['move_vectors']
        for i in range(len(images_to_mv)):
            img_num = images_to_mv[i][0]
            slab = self.util_obj.get_single_image(self.normalized_nexusfile,
                                                  img_num,
                                                  self.numrows,
                                                  self.numcols)

            mv_vector = images_to_mv[i][1]

            vects_field[img_num] = mv_vector
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

        vects_field.write()











