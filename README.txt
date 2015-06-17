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


ctalign project has born with the initial idea of make an automatic
alignment of tomography projections in order to allow further reconstruction.
Nowadays it is used for not only for tomography projection alignment, but also 
for spectroscopic images alignment at ALBA BL09-Mistral beamline.

In the future, a generalization of the project could be studied, in order to be 
used in other beamlines; but this is not the case in the current stage. 

The alignment is being done by using the function matchTemplate from python
openCV (cv2).



---------------

TODO list:

- If thresholds are too high (for difference of move in pixels; and for
portion of vectors that have to be similar); no vectors are stored.
This is a problem, and in this case the process should be redone with 
lower thresholds.

- If ROIs choosen goes out of the image, inform the user, or simply take
less ROIs or take smaller ROIs automatically.

- Look for the simetry in the image. ROIs should be choosen from the center
to the top and to the bottom; and from the center to the left and to the 
right. 

- Allow the user not only choose the number or ROIs, but also the size of 
the ROIs (by giving some default values).

- Filtering images in order to only take into account the most contrasted
objects in the image. Apply the alignment after having the images filtered.
