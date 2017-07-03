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

import os
import numpy as np
import cv2

import h5py
import nxs
from alignlib.utils import Utils


class Alignment(object):
    # Constructor of Alignment object ###
    def __init__(self, inputfile, roi_select, spec, firstimg, printmv,
                 num_roi_horizontal, num_roi_vertical, width, height):

        self.filename_nexus = inputfile
        self.input_nexusfile = h5py.File(self.filename_nexus, 'r')
        self.norm_grp = 0

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
        self.align_file = h5py.File(self.outputfilehdf5, 'w')
        self.align = self.align_file.create_group("FastAligned")
        self.align.attrs['NX_class'] = "NXentry"

        self.util_obj = Utils()
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
        self.nxsfield = 0
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

    def roi_selection(self, img_for_roi_select, spectrum=0):

        cv2.startWindowThread()

        # Rescale the image grayscale colors in order to make it visible
        if spectrum == 0:
            mult_factor = 1 / img_for_roi_select.max()
            img_for_roi_select = mult_factor * img_for_roi_select
        else:
            img_without_borders = img_for_roi_select[
                int(self.numrows*0.15):self.numrows - int(self.numrows*0.07),
                int(self.numcols*0.15):self.numcols - int(self.numcols*0.07)]
            max_pixel_img = np.amax(img_without_borders)
            min_pixel_img = np.amin(img_without_borders)
            factor_feature_matching = (1.0/(max_pixel_img - min_pixel_img))
            img_for_roi_select = ((img_for_roi_select - min_pixel_img) *
                                  factor_feature_matching*255).astype(np.uint8)

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

            # Esc key to stop
            if key == 27:
                import sys
                sys.exit()
            # if the 'r' key is pressed, reset the selected region
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

    def find_mv_vector_from_many_rois(self, move_vectors_matrix):
        """Find move vector from many ROIs.
        Input argument: Matrix of move vectors
        Return variable: Average move vector of relevant move vectors.
        A move vector is considered to be relevant if it is similar to
        the other move vectors in a certain degree. If it is to distant from
        other move vectors it is not taken into account for being considered
        in the calculation of the returned average move vector."""

        # An error has been observed, if the thresholds are to high and
        # no vectors are stored after filtering, a problem exist. It
        # should be tried to lower the thresholds if no vectors are stored,
        # In order to at least find one of the vectors.

        # Useful for sorting list of lists based on second list element.
        # Our list of lists, is a list of vectors.

        from operator import itemgetter

        # Threshold in pixels for knowing if we increment or not the counter
        # indicating how many move vectors are similar to the processed move
        # vector currently being processed.
        pixels_threshold = 5

        # Threshold for deciding if we keep or not a given move vector,
        # according in how many similar vectors to itself exists.
        # Expressed in tant_per_one.
        threshold_similarity = 0.5

        mv_list = list_of_mv_vectors_for_processing = []

        for vert in range(self.numroivert):
            for horiz in range(self.numroihoriz):
                #print(move_vectors_matrix[vert][horiz])

                mv_list.append(list(move_vectors_matrix[vert][horiz]))

        # We sort the mv_vectors list by its first element
        sorted_mv_list = sorted(mv_list)
        len_mv_list = len(mv_list)
        threshold_similar_vectors = threshold_similarity * len_mv_list

        # Empty list that will contain the number of vectors similar between
        # them, based by the similarity between first vector elements.
        # We will use a given tolerance (in number of pixels to be moved),
        # in order to decided if the vectors are similar or not, based on the
        # first element.
        counts_of_similar_vectors = len_mv_list*[0]

        for i in range(len(sorted_mv_list)):
            vector = sorted_mv_list[i]
            counts = 0
            for j in range(len(sorted_mv_list)):
                if i != j:
                    # Vector to compare: vector_to_cmp
                    vector_to_cmp = sorted_mv_list[j]
                    if abs(vector_to_cmp[0] - vector[0]) <= pixels_threshold:
                        counts += 1
            counts_of_similar_vectors[i] = counts

        # move vectors after having being filtered by the similarity of the
        # first elements.
        first_elem_filtered_mv_vectors = []
        for i in range(len(sorted_mv_list)):
            counts = counts_of_similar_vectors[i]
            if counts >= threshold_similar_vectors:
                first_elem_filtered_mv_vectors.append(sorted_mv_list[i])

        # If there are not at least 2 vectors in the list repeat the procdure
        # for a lower threshold of number of similar vectors.
        for i in range(8):
            if len(first_elem_filtered_mv_vectors) < 2:
                first_elem_filtered_mv_vectors = []
                threshold_similarity = threshold_similarity - i*0.05
                threshold_similar_vectors = threshold_similarity * len_mv_list
                for i in range(len(sorted_mv_list)):
                    counts = counts_of_similar_vectors[i]
                    if counts >= threshold_similar_vectors:
                        first_elem_filtered_mv_vectors.append(sorted_mv_list[i])
            else:
                break

        # After filtering the vectors by its first element, a new filter will
        # be applied, filtering the move vectors by its second element.
        sorted_mv_list = sorted(first_elem_filtered_mv_vectors,
                                key=itemgetter(1))

        len_filtered_vectors = len(sorted_mv_list)
        threshold_similarity = 0.5
        threshold_similar_vectors = threshold_similarity * len_filtered_vectors
        counts_of_similar_vectors = len_filtered_vectors*[0]

        for i in range(len(sorted_mv_list)):
            vector = sorted_mv_list[i]
            counts = 0
            for j in range(len(sorted_mv_list)):
                if i != j:
                    # Vector to compare: vector_to_cmp
                    vector_to_cmp = sorted_mv_list[j]
                    if abs(vector_to_cmp[1] - vector[1]) <= pixels_threshold:
                        counts += 1
            counts_of_similar_vectors[i] = counts
        # move vectors after having being filtered by the similarity of the
        # first elements.
        second_elem_filtered_mv_vectors = []
        for i in range(len(sorted_mv_list)):
            counts = counts_of_similar_vectors[i]
            if counts >= threshold_similar_vectors:
                second_elem_filtered_mv_vectors.append(sorted_mv_list[i])

        # If there are not at least 2 vectors in the list repeat the procedure
        # for a lower threshold of number of similar vectors.
        for i in range(8):
            if len(second_elem_filtered_mv_vectors) < 2:
                second_elem_filtered_mv_vectors = []
                threshold_similarity = threshold_similarity - i*0.05
                threshold_similar_vectors = (threshold_similarity *
                                             len_filtered_vectors)
                for i in range(len(sorted_mv_list)):
                    counts = counts_of_similar_vectors[i]
                    if counts >= threshold_similar_vectors:
                        second_elem_filtered_mv_vectors.append(
                            sorted_mv_list[i])
            else:
                break

        # Finally, the average of the remaining move vectors after the
        # filtering, is calculated.
        mv_vectors = second_elem_filtered_mv_vectors
        len_filtered_mv_vect = len(second_elem_filtered_mv_vectors)
        np_total_mv_vector = np.array([0,0])
        for i in range(len_filtered_mv_vect):
            np_total_mv_vector += np.array(mv_vectors[i])

        # average move vector after filtering
        avg_mv_vector = np_total_mv_vector/float(len_filtered_mv_vect)
        avg_mv_vector = np.around(avg_mv_vector)
        avg_mv_vector = avg_mv_vector.astype(int)
        return avg_mv_vector

    def store_currents(self):
        #############################################
        # Retrieving important data from currents  ##
        #############################################
        if self.spec == 0:
            currents_dataset_name = 'CurrentsTomo'
        elif self.spec == 1:
            currents_dataset_name = 'Currents'

        try:
            currents = self.norm_grp[currents_dataset_name].value
            self.align["Currents"] = currents
        except:
            print("\nCurrents could NOT be extracted.\n")

    def store_exposure_times(self):
        ###################################################
        # Retrieving important data from Exposure Times  ##
        ###################################################
        if self.spec == 0:
            exposure_dataset_name = 'ExpTimesTomo'
        elif self.spec == 1:
            exposure_dataset_name = 'ExpTimes'
        try:
            exptimes = self.norm_grp[exposure_dataset_name].value
            self.align["ExpTimes"] = exptimes
        except:
            print("\nExposure Times could NOT be extracted.\n")

    def store_pixel_size(self):
        #########################
        # Retrieving Pixel Size #
        #########################
        try:
            x_pixel_size = self.norm_grp["x_pixel_size"].value
            y_pixel_size = self.norm_grp["y_pixel_size"].value
            self.align.create_dataset("x_pixel_size", data=x_pixel_size)
            self.align.create_dataset("y_pixel_size", data=y_pixel_size)
        except:
            print("\nPixel size could NOT be extracted.\n")

    def store_energies(self):
        #######################
        # Retrieving Energies #
        #######################
        try:
            energies = self.norm_grp["energy"].value
            self.align.create_dataset("energy", data=energies)
        except:
            print("\nEnergies could not be extracted.\n")

    def store_angles(self):
        #######################
        # Retrieving Angles #
        #######################
        try:
            self.angles = self.norm_grp["rotation_angle"].value
            self.align.create_dataset("rotation_angle", data=self.angles)
        except:
            print("\nAngles could not be extracted.\n")
        if self.spec == 0:
            try:
                angles_file = self.path + "/angles.tlt"
                from os import path
                file_exists = path.isfile(angles_file)
                if not file_exists:
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
            images = self.norm_grp['TomoNormalized']
        elif self.spec == 1:
            images = self.norm_grp['spectroscopy_normalized']

        # Shape information of data image stack
        infoshape = images.shape
        self.dim_images = (infoshape[0], infoshape[1], infoshape[2])
        self.nFrames = infoshape[0]
        self.numrows = infoshape[1]
        self.numcols = infoshape[2]
        print("Dimensions: {0}".format(self.dim_images))

    def create_image_storage_dataset(self):
        self.align.create_dataset(
            self.data_nxs,
            shape=(self.nFrames,
                   self.numrows,
                   self.numcols),
            chunks=(1,
                    self.numrows,
                    self.numcols),
            dtype='float32')

        self.align[self.data_nxs].attrs['Number of Frames'] = self.nFrames
        self.align[self.data_nxs].attrs['Pixel Rows'] = self.numrows
        self.align[self.data_nxs].attrs['Pixel Columns'] = self.numcols

    def store_metadata(self):
        self.store_currents()
        self.store_exposure_times()
        self.store_pixel_size()
        self.store_energies()
        self.store_angles()

    def initialize_align(self):
        if self.spec == 0:
            self.norm_grp = self.input_nexusfile["TomoNormalized"]
        elif self.spec == 1:
            self.norm_grp = self.input_nexusfile["SpecNormalized"]

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
