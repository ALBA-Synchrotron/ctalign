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


class Utils:

    def __init__(self):
        pass

    def print_move(self, mv_vect_filename, mv_vector_list):
        f = open(mv_vect_filename, 'w')
        f.write("Move vectors:\n")
        for i in range(len(mv_vector_list)):
            s = (str(mv_vector_list[i][0]) + " " +
                 str(mv_vector_list[i][1]) + "\n")
            f.write(s)
        f.close()
        print("\nMove vectors file has been created")

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


