#!/usr/bin/python

"""
(C) Copyright 2014-2017 Marc Rosanes
The program is distributed under the terms of the
GNU General Public License (or the Lesser GPL).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import time
import argparse

from aligntomo.tomo_subsequent_align import TomoSubsequentAlign

from alignspectrum.linear_align import SpectrumLinearAlign
from alignspectrum.subsequent_align import SpectrumSubsequentAlign
from alignspectrum.fromfirstimage_align import \
    SpectrumFromFirstImgAlign
from alignspectrum.post_align import PostAlignRemoveJumps


def main():
    parser = argparse.ArgumentParser(
                        description="Align the images \n")

    parser.add_argument('inputfile', type=str, default=None,
                        help='Enter hdf5 file containing the normalized '
                             'tomography or spectroscopy')
    parser.add_argument('-r', '--roi', type=int, default=0, 
                        help='Option for allowing ROI selection. \n' +
                        'Default = 0')
    parser.add_argument('-x','--nroihoriz', type=int, default=1,
                        help='Used for multiple ROIs: Number of ROIs " + '
                             'in horizontal direction. \nDefault = 1')
    parser.add_argument('-y', '--nroivert', type=int, default=1,
                        help='Used for multiple ROIs: Number of ROIs in '
                             'vertical direction.\nDefault = 1')
    parser.add_argument('-ww', '--width', type=int, default=200,
                        help='Used for multiple ROIs: Width of '
                             'ROI in pixels. \nDefault = 200')
    parser.add_argument('-hh', '--height', type=int, default=200,
                        help='Used for multiple ROIs: Height of ROI in '
                             'pixels. \nDefault = 200')
    parser.add_argument('-s', '--spectrum', type=int, default=0,
                        help='Align projections -s=0 || Align spectral '
                             'images -s=1. \nDefault = 0')
    parser.add_argument('-f', '--firstimg', type=int, default=1,
                        help='Align spectroscopic images regarding the '
                             'previous image (-f=0)\n, regarding the '
                             'first image (-f=1)\n or linearly between the '
                             'first and the last images (-f=2).\n'
                             'Default = 1')
    parser.add_argument('-m', '--printmv', type=int, default=0, 
                        help='Print move vectors between images. \n' +
                        'Default = 0')
    parser.add_argument('-j', '--rmjumpshdf', type=str, default=None,
                        help='Post-alignment for removing jumps in '
                             'spectroscopies.\n' +
                             'j= hdf5 containing the aligned spectroscopy.')
    args = parser.parse_args()

    print("\nAligning images\n")

    start_time = time.time()

    # Main alignment
    if args.rmjumpshdf is None:
        if args.spectrum == 0:
            tomo_alignment = TomoSubsequentAlign(args.inputfile,
                                                 args.roi,
                                                 args.spectrum,
                                                 args.firstimg,
                                                 args.printmv,
                                                 args.nroihoriz,
                                                 args.nroivert,
                                                 args.width,
                                                 args.height)
            tomo_alignment.initialize_align()
            tomo_alignment.tomo_subsequent_alignment()

        elif args.spectrum == 1:
            if args.firstimg == 0:
                spectrum_alignment = SpectrumSubsequentAlign(args.inputfile,
                                                             args.roi,
                                                             args.spectrum,
                                                             args.firstimg,
                                                             args.printmv,
                                                             args.nroihoriz,
                                                             args.nroivert,
                                                             args.width,
                                                             args.height)
                spectrum_alignment.initialize_align()
                spectrum_alignment.spectrum_subsequent_alignment()

            elif args.firstimg == 1:
                spectrum_alignment = SpectrumFromFirstImgAlign(args.inputfile,
                                                               args.roi,
                                                               args.spectrum,
                                                               args.firstimg,
                                                               args.printmv,
                                                               args.nroihoriz,
                                                               args.nroivert,
                                                               args.width,
                                                               args.height)
                spectrum_alignment.initialize_align()
                spectrum_alignment.spectrum_from_first_img_alignment()

            elif args.firstimg == 2:
                spectrum_alignment = SpectrumLinearAlign(args.inputfile,
                                                         args.roi,
                                                         args.spectrum,
                                                         args.firstimg,
                                                         args.printmv,
                                                         args.nroihoriz,
                                                         args.nroivert,
                                                         args.width,
                                                         args.height)
                spectrum_alignment.initialize_align()
                spectrum_alignment.spectrum_linear_alignment()

    else:
        # Optional post-alignment.
        # For the moment the remove jumps is only valid for spectroscopies.
        # The artifact movement observed in spectroscopies is normally a drift.
        # If a jump does not follow the tendance of the drift, the post_align   
        # shall correct this jump by interpolating the moving vectors. 
        post_alignment = PostAlignRemoveJumps(args.inputfile,
                                              args.rmjumpshdf)
        post_alignment.move_images()

    print("\nImages aligned\n\n")

    print("--- %s minutes ---\n" % ((time.time() - start_time)/60))


if __name__ == "__main__":
    main()
