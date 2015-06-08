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


aep1: alignment enhancement proposal 1

The alignment projections project has born with the idea of make a fast 
alignment of a tomography projections in order to allow further reconstruction.
First we have tried to perform the alignment using the SAD algorithm
with a central ROI for each projection. But the results were only satisfactory
for some projections, others were really poorly aligned. 

Because of the fact exposed above, we have tried to explore the library openCV
in order to improve the alignment using its functions. This has been the
idea of aep1 and the alignment results have improved.
