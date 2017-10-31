#!/usr/bin/env python

''' Transform Praat TextGrid file to csv annotations 

    See the file documentation at:

    http://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html

'''

import os 
import re

INTERVAL = re.compile("intervals \[\d+\]:")

def dump_textgrid2csv(grid_file):
    '''dump_textgrid2csv decodes TextGrid files and dumps the resutls to the stdout '''

    grid_dir = os.path.dirname(os.path.realpath(grid_file)) 
    grid_basename = os.path.basename(grid_file)
    grid_ = os.path.join(grid_dir, grid_basename)
    with open(grid_file) as tfile:
        print("filename,start,end,label") 
	decode_int = False
	for lines in tfile.readlines():
	    l = lines.strip()
	    intev = []
	    if INTERVAL.search(l):
		xmin = xmax = text = ''
		decode_int = True

	    # to build the csv file it will be needed only the minimum and 
	    # maximum time for a given transcription
	    if decode_int:
		if 'xmin' in l:
		    xmin = re.match("xmin = (.*)", l).groups()[0]
		
		elif 'xmax' in l:
		    xmax = re.match("xmax = (.*)", l).groups()[0]

		# all empty texts are decoded as silences
		elif 'text' in l:
		    text = re.match("text = \"(.*)\"", l).groups()[0]
                    text = ' '.join(text.split())
                    if not text or text == ' ':
                        text = 'SIL'
		else:
		    pass

		if xmin and xmax and text:
		    print("{},{:.3f},{:.3f},{}".format(grid_, 
                        float(xmin), float(xmax), text))
		    decode_int = False



if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('textgrid_file', metavar='TEXTGRID_FILE', nargs=1, \
            help='File in TextGrid format')
    args = parser.parse_args()
    textgrid_file = args.textgrid_file[0]
    dump_textgrid2csv(textgrid_file)
