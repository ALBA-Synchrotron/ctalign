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

    def count(self, counter):
        counter += 1
        if counter % 10 == 0:
            print(".")
        return counter