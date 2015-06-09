#!/usr/bin/python

"""
(C) Copyright 2014 Miquel Garriga 
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

import numpy as np
from alignlib.align import Alignment
from alignlib.utils import Utils
import cv2


class SpectrumLinearAlign(Alignment):

    def __init__(self, inputfile, roi_select, spec, firstimg, printmv,
                 num_roi_horizontal, num_roi_vertical):
        super(SpectrumLinearAlign, self).__init__(inputfile, roi_select, spec,
                                                  firstimg, printmv,
                                                  num_roi_horizontal,
                                                  num_roi_vertical)

    def spectrum_linear_alignment(self):

        #################################################
        #  Get align and store aligned images in HDF5  ##
        #################################################
        self.input_nexusfile.opendata('spectroscopy_normalized')
        img_num = 0

        self.image_proj1 = self.get_single_image(img_num )
        self.proj1 = self.image_proj1[0, :, :]
        # cv2.imshow('proj1',proj1)
        # cv2.waitKey(0)
        slab_offset = [img_num , 0, 0]
        self.nxsfield = self.align[self.data_nxs]
        self.store_image_in_hdf(self.image_proj1, self.nxsfield, slab_offset)
        print('Initial reference image (%d) stored\n' % img_num )

        self.central_pixel_rows = int(self.numrows / 2)
        self.central_pixel_cols = int(self.numcols / 2)

        self.row_tem_from = self.central_pixel_rows - self.height_tem / 2
        self.row_tem_to = self.central_pixel_rows + self.height_tem / 2
        self.col_tem_from = self.central_pixel_cols - self.width_tem / 2
        self.col_tem_to = self.central_pixel_cols + self.width_tem / 2

        # In openCV first we indicate the columns and then the rows.
        self.top_left_base = (self.col_tem_from, self.row_tem_from)
        print('Initialization completed')

        print("Linearly align spectroscopic images between " + 
              "first and last image")
        print "Region of reference used as template:"
        print ("from row", self.row_tem_from, "col", self.col_tem_from,
               "to row", self.row_tem_to, "col", self.col_tem_to)
        template = self.proj1[self.row_tem_from:self.row_tem_to,
                         self.col_tem_from:self.col_tem_to]
        image_proj2 = self.get_single_image(self.nFrames-1)
        proj2 = image_proj2[0, :, :]
        #toalignroi=proj2[self.row_tem_from:self.row_tem_to,
        #                 self.col_tem_from:self.col_tem_to]
        # display the reference image
        #cv2.namedWindow ('Reference', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Reference', 480, 480)
        #cv2.moveWindow  ('Reference',   0, 30)
        annotated_proj1 = self.proj1.copy()
        # mark with a white rectangle the region to be matched
        cv2.rectangle(annotated_proj1, (self.col_tem_from, self.row_tem_from),
                     (self.col_tem_to, self.row_tem_to), 1, 1)
        #cv2.imshow('Reference', annotated_proj1)

        # display the to be aligned image
        #cv2.namedWindow ('To align', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('To align', 480, 480)
        #cv2.moveWindow  ('To align', 960, 30)
        # mark with a white rectangle the same region as in reference frame
        annotated_proj2 = proj2.copy()
        cv2.rectangle(annotated_proj2, (self.col_tem_from, self.row_tem_from),
                     (self.col_tem_to, self.row_tem_to), 1, 1)
        #cv2.imshow('To align', annotated_proj2)

        # display the template to match and the corresponding 
        # region in 'to align' image
        #cv2.namedWindow ('Template', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Template', 480, 480)
        #cv2.moveWindow  ('Template', 480,  30)
        #cv2.imshow('Template', np.concatenate((template,toalignroi), 
        #            axis=1))
        #cv2.waitKey(0)

        h = template.shape[0]
        w = template.shape[1]

        ## cross-correlations are used in cv2 in function matchTemplate
        # Apply template Matching
        method = cv2.TM_CCOEFF_NORMED
        result = cv2.matchTemplate(proj2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print "Shift: ", self.col_tem_from-max_loc[0], self.row_tem_from-max_loc[1]
        # change to True to show show the correlation matrix
        if False:
            res_ptp = (result-result.min())/(result.max()-result.min())
            #cv2.imshow('Template', res2)
            #cv2.waitKey(0)

        # find the displacement vector
        top_left_move = max_loc
        bottom_right_move = (top_left_move[0] + w, top_left_move[1] + h)
        move_vector = self.find_mv_vector(self.top_left_base, top_left_move)

        # display the 'to align image', with final position in black
        cv2.rectangle(annotated_proj2, (max_loc[0],max_loc[1]), 
                                        (max_loc[0]+w,max_loc[1]+h), 0)
        #cv2.imshow('To align',annotated_proj2)
        #toalignroi=proj2[max_loc[1]:
        #                 (max_loc[1]+h), max_loc[0]:(max_loc[0]+w)]
        #cv2.imshow('Template', np.concatenate((template,toalignroi),
        #                        axis=1))
        #cv2.waitKey(0)

        # uniformly distribute the (x,y)-shifts
        # there are two ways of shifting the images:
        # A: filling an image with zeros and adding on top the shifted image
        # B: 'rolling' the image values along rows/columns
        # Case A is used in the other parts of this  
        # script (-s 0 | -s 1 -f 0 | -s 1 -f 1)
        # Here ICMAB uses the B way
        if False: # case A
            xshift=np.rint(np.linspace(0, np.float(move_vector[0]),
                                       self.nFrames)).astype('int32')
            yshift=np.rint(np.linspace(0, np.float(move_vector[1]),
                                       self.nFrames)).astype('int32')
        else: # case B
            xcorr = self.col_tem_from-max_loc[0]
            ycorr = self.row_tem_from-max_loc[1]
            xshift=np.rint(np.linspace(0, np.float(xcorr),
                                       self.nFrames)).astype('int32')
            yshift=np.rint(np.linspace(0, np.float(ycorr),
                                       self.nFrames)).astype('int32')
        print np.vstack((xshift, yshift))

        util_obj = Utils()
        self.counter = 0
        for numimg in range(1,self.nFrames):
            image_proj2 = self.get_single_image(numimg)
            proj2 = image_proj2[0, :, :]
            if False: # case A
                mv_vector = [yshift[numimg],xshift[numimg]]
                zeros_img = np.zeros((self.numrows, self.numcols), 
                                                        dtype='float32')
                proj2_moved = self.mv_projection(zeros_img, proj2, mv_vector)
            else: # case B
                proj2_moved = np.roll(np.roll(proj2, xshift[numimg], axis=1),
                                      yshift[numimg], axis=0)

            proj2 = np.zeros([1, self.numrows, self.numcols], dtype='float32')

            proj2[0] = proj2_moved
            slab_offset = [numimg, 0, 0]
            self.store_image_in_hdf(proj2, self.nxsfield, slab_offset)
            self.counter = util_obj.count(self.counter)

        self.input_nexusfile.closedata()
        self.input_nexusfile.closegroup()
        self.input_nexusfile.close()