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

import numpy as np
import nxs
import cv2
import os


class Alignment(object):
    # Constructor of Alignment object ###
    def __init__(self, inputfile, roi_select, spec, firstimg, printmv,
                 num_roi_horizontal, num_roi_vertical, width, height):

        self.input_nexusfile = nxs.open(inputfile, 'r')
        self.outputfilehdf5 = inputfile.split('.hdf')[0] + '_ali' + '.hdf5'
        self.path = os.path.dirname(inputfile)
        if not self.path:
            self.path = '.'
        self.user_roi_select = roi_select
        self.spec = spec
        self.firstimg = firstimg

        if self.spec == 0:
            self.data_nxs = 'tomo_aligned'
        elif self.spec == 1:
            self.data_nxs = 'spec_aligned'
        self.align = nxs.NXentry(name="FastAligned")
        self.align.save(self.outputfilehdf5, 'w5')

        self.method = eval('cv2.TM_CCOEFF_NORMED')
        # self.method = eval('cv2.TM_CCORR_NORMED')

        # Angles        
        self.angles = list()

        # Images
        self.nFrames = 0
        self.numrows = 0
        self.numcols = 0
        self.dim_images = (0, 0, 1)

        self.numroihoriz = num_roi_horizontal
        self.numroivert = num_roi_vertical

        self.mv_vector_list = []

        # create roi_points attribute
        self.roi_points = []

        self.counter = 0

        if num_roi_horizontal == 1 and num_roi_vertical == 1:
            self.width_tem = 300
            self.height_tem = 500
        else:
            self.width_tem = width
            self.height_tem = height

        self.central_pixel_rows = 0
        self.central_pixel_cols = 0

        self.row_tem_from = 0
        self.row_tem_to = 0
        self.col_tem_from = 0
        self.col_tem_to = 0

        self.image_proj1 = 0
        self.nxsfied = 0
        self.proj1 = 0
        self.central_img_num = 0
        self.top_left_base = 0
        self.proj1_roi_selection = 0

        # MV vectors file
        self.printmv = printmv
        if printmv == 1:
            self.mv_vect_filename = inputfile.split('.hdf')[
                                        0] + '_shift' + '.txt'

        return

    # Get image
    def get_single_image(self, numimg):
        image_retrieved = self.input_nexusfile.getslab([numimg, 0, 0],
                                                       [1, self.numrows,
                                                        self.numcols])
        return image_retrieved

    # Save image
    def store_image_in_hdf(self, image, nxsfield, slab_offset):
        nxsfield.put(image, slab_offset, refresh=False)
        nxsfield.write()

    def roi_selection(self, img_for_roi_select):

        cv2.startWindowThread()

        # Rescale the image grayscale colors in order to make it visible
        mult_factor = 1 / img_for_roi_select.max()
        img_for_roi_select = mult_factor * img_for_roi_select

        window_name = "projection_for_roi_selection"

        def click_and_select(event, x, y, flags, param):
            # record the starting (x, y) coordinates (cols, rows)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_points = [(x, y)]

            # check to see if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates (cols, rows)
                self.roi_points.append((x, y))
                # draw a rectangle around the region of interest
                cv2.rectangle(img_for_roi_select, self.roi_points[0],
                              self.roi_points[1], 255, 2)  # (0, 100, 0), 2)
                cv2.imshow(window_name, img_for_roi_select)

        # load the image, clone it, and setup the mouse callback function
        clone = img_for_roi_select.copy()
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_and_select)

        while True:
            # display the image and wait for a keypress
            cv2.imshow(window_name, img_for_roi_select)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the selected region
            if key == 27:  # Esc key to stop
                import sys

                sys.exit()
            elif key == ord("r"):
                img_for_roi_select = clone.copy()
                cv2.imshow(window_name, img_for_roi_select)
            # if 'ENTER' key is pressed, store ROI coordinates and break loop
            elif key == 13 or key == ord(u"\u000A"):
                # close all open windows
                cv2.destroyWindow(window_name)
                break

    # How much to move a projection ###
    def find_mv_vector(self, coords_base_proj, coords_mv_proj):
        # coords1 being the base projection (the one that will not move)
        coords_base_proj = np.array(coords_base_proj)
        coords_mv_proj = np.array(coords_mv_proj)
        # We have to bring the proj2 (mv_proj) to the base_proj (fixed proj)
        mv_vector = coords_base_proj - coords_mv_proj
        return mv_vector

    # Move a projection #
    def mv_projection(self, empty_img, proj_two, mv_vector):
        rows = proj_two.shape[0]
        cols = proj_two.shape[1]
        mvr = abs(mv_vector[0])
        mvc = abs(mv_vector[1])
        ei = empty_img
        pt = proj_two

        if mv_vector[0] == 0 and mv_vector[1] == 0:
            ei[:, :] = pt[:, :]

        elif mv_vector[0] > 0 and mv_vector[1] == 0:
            ei[mvr:rows, :] = pt[0:rows - mvr, :]

        elif mv_vector[0] < 0 and mv_vector[1] == 0:
            ei[0:rows - mvr, :] = pt[mvr:rows, :]

        elif mv_vector[0] == 0 and mv_vector[1] > 0:
            ei[:, mvc:cols] = pt[:, 0:cols - mvc]

        elif mv_vector[0] == 0 and mv_vector[1] < 0:
            ei[:, 0:cols - mvc] = pt[:, mvc:cols]

        elif mv_vector[0] > 0 and mv_vector[1] > 0:
            ei[mvr:rows, mvc:cols] = pt[0:rows - mvr, 0:cols - mvc]

        elif mv_vector[0] > 0 and mv_vector[1] < 0:
            ei[mvr:rows, 0:cols - mvc] = pt[0:rows - mvr, mvc:cols]

        elif mv_vector[0] < 0 and mv_vector[1] > 0:
            ei[0:rows - mvr, mvc:cols] = pt[mvr:rows, 0:cols - mvc]

        elif mv_vector[0] < 0 and mv_vector[1] < 0:
            ei[0:rows - mvr, 0:cols - mvc] = pt[mvr:rows, mvc:cols]

        moved_proj_two = ei
        # cv2.imshow('proj2mv',moved_proj_two)
        # cv2.waitKey(0)
        return moved_proj_two

    def store_currents(self):
        #############################################
        # Retrieving important data from currents  ##
        #############################################
        if self.spec == 0:
            currents_dataset_name = 'CurrentsTomo'
        elif self.spec == 1:
            currents_dataset_name = 'Currents'
        try:
            self.input_nexusfile.opendata(currents_dataset_name)
            currents = self.input_nexusfile.getdata()
            self.input_nexusfile.closedata()
            self.align['Currents'] = currents
            self.align['Currents'].write()
        except:
            print("\nCurrents could NOT be extracted.\n")
            try:
                self.input_nexusfile.closedata()
            except:
                pass

    def store_exposure_times(self):
        ###################################################
        # Retrieving important data from Exposure Times  ##
        ###################################################
        if self.spec == 0:
            exposure_dataset_name = 'ExpTimesTomo'
        elif self.spec == 1:
            exposure_dataset_name = 'ExpTimes'
        try:
            self.input_nexusfile.opendata(exposure_dataset_name)
            exptimes = self.input_nexusfile.getdata()
            self.input_nexusfile.closedata()
            self.align['ExpTimes'] = exptimes
            self.align['ExpTimes'].write()
        except:
            print("\nExposure Times could NOT be extracted.\n")
            try:
                self.input_nexusfile.closedata()
            except:
                pass

    def store_pixel_size(self):
        ###################################################
        # Retrieving important data from Pixel Size      ##
        ###################################################
        try:
            self.input_nexusfile.opendata('x_pixel_size')
            x_pixel_size = self.input_nexusfile.getdata()
            self.input_nexusfile.closedata()
            self.input_nexusfile.opendata('y_pixel_size')
            y_pixel_size = self.input_nexusfile.getdata()
            self.input_nexusfile.closedata()
            self.align['x_pixel_size'] = x_pixel_size
            self.align['x_pixel_size'].write()
            self.align['y_pixel_size'] = y_pixel_size
            self.align['y_pixel_size'].write()
        except:
            print("\nPixel size could NOT be extracted.\n")
            try:
                self.input_nexusfile.closedata()
            except:
                pass

    def store_energies(self):
        #############################################
        # Retrieving important data from energies  ##
        #############################################
        try:
            self.input_nexusfile.opendata('energy')
            energies = self.input_nexusfile.getdata()
            self.input_nexusfile.closedata()
            self.align['energy'] = energies
            self.align['energy'].write()
        except:
            print("\nEnergies could NOT be extracted.\n")
            try:
                self.input_nexusfile.closedata()
            except:
                pass

    def store_angles(self):
        ########################################
        # Storing tilt angles in a text file  ##
        ########################################
        try:
            self.input_nexusfile.opendata('rotation_angle')
            self.angles = self.input_nexusfile.getdata()
            self.input_nexusfile.closedata()
            self.align['rotation_angle'] = self.angles
            self.align['rotation_angle'].write()
        except:
            print("\nAngles could NOT be extracted.\n")
            try:
                self.input_nexusfile.closedata()
            except:
                pass
        if self.spec == 0:
            try:
                angles_file = self.path + "/angles.tlt"
                f = open(angles_file, 'w')
                for i in range(len(self.angles)):
                    angle_str = str(round(self.angles[i], 2))
                    if len(angle_str) == 5:
                        f.write(" " + angle_str)
                    elif len(angle_str) == 4:
                        f.write("  " + angle_str)
                    elif len(angle_str) == 3:
                        f.write("   " + angle_str)
                    elif len(angle_str) == 2:
                        f.write("    " + angle_str)
                    f.write("\n")
                f.close()
                print("Angles have been stored")
            except:
                print("\nError in saving angles in a .tlt file.\n")

    def retrieve_image_dimensions(self):
        ########################################
        #  Retrieving data from images shape  ##
        ########################################
        if self.spec == 0:
            self.input_nexusfile.opendata('TomoNormalized')
        elif self.spec == 1:
            self.input_nexusfile.opendata('spectroscopy_normalized')
        infoshape = self.input_nexusfile.getinfo()
        self.dim_images = (infoshape[0][0], infoshape[0][1], infoshape[0][2])
        self.nFrames = infoshape[0][0]
        self.numrows = infoshape[0][1]
        self.numcols = infoshape[0][2]
        print("Dimensions: {0}".format(self.dim_images))
        self.input_nexusfile.closedata()

    def create_image_storage_dataset(self):
        self.align[self.data_nxs] = nxs.NXfield(name=self.data_nxs,
                                                dtype='float32',
                                                shape=[nxs.UNLIMITED,
                                                       self.numrows,
                                                       self.numcols])
        self.align[self.data_nxs].attrs[
            'Number of Frames'] = self.nFrames
        self.align[self.data_nxs].attrs[
            'Pixel Rows'] = self.numrows
        self.align[self.data_nxs].attrs[
            'Pixel Columns'] = self.numcols
        self.align[self.data_nxs].write()

    def store_metadata(self):
        self.store_currents()
        self.store_exposure_times()
        self.store_pixel_size()
        self.store_energies()
        self.store_angles()

    def initialize_align(self):
        if self.spec == 0:
            self.input_nexusfile.opengroup('TomoNormalized')
        elif self.spec == 1:
            self.input_nexusfile.opengroup('SpecNormalized')

        ###################
        # Store metadata ##
        ###################
        print('Store metadata')
        self.store_metadata()
        print('Metadata stored\n')

        ################################################
        # Create empty dataset for image data storage ##
        ################################################
        print('Initialize store images')
        self.retrieve_image_dimensions()
        self.create_image_storage_dataset()
