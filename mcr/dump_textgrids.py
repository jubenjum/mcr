#!/usr/bin/env python

"""
Routines to parse Praat TextGrid files, the decoded files will be
printed as csv annotations. The output file will have:

    file_name,start,end,annotation

The following url describes in detail the TextGrid file:

    http://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html

"""

import os
import string
from pathlib2 import Path

from pyparsing import *



__all__ = ['dump_textgrid2csv']

# symbols that are considered as silences, and will be translate to an
# unique string "SIL"
SILS = [ ' ', 'SIL', 'n', '#', 'X'] 

def dump_textgrid2csv(grid_file):
    """dump_textgrid2csv parse TextGrid files and dumps the results to the stdout

    
    Parameters
    ----------

    grid_file: [string]
        a string with the name of an existing TextGrid file
    
    Returns
    -------

    print out the decoded TextGrid annotations

    """

    assert Path(grid_file).is_file(), "file not found: {}".format(grid_file)

    # Parsed tokens
    integ = Word(nums).setParseAction(lambda t: int(t[0]))
    number = Word(nums+".").setParseAction(lambda t: float(t[0]))

    # removed tokens
    NL = Suppress(LineEnd())
    _lb = Suppress(Literal('['))
    _rb = Suppress(Literal(']'))
    _eg = Suppress(Literal('='))
    _c = Suppress(Literal(':'))

    # interval rules:
    interv_num = Suppress(Literal('intervals')) + _lb + integ("num_interv") + _rb + _c  + NL
    interv_max = Suppress(Literal('xmax')) + _eg + number("max") + NL
    interv_min = Suppress(Literal('xmin')) + _eg + number("min") + NL
    interv_annot = Suppress(Literal('text')) + _eg + \
                   QuotedString('"')("annotation") + NL

    # Fields in the item tag: the fields available in this tag are:
    #
    #       item, name, xmin, xmax, and "interval: size"
    #
    # xmax, xmin are timestamps of the annotation or the items, "interval: size" gives the
    # number of intervals in the tag.
    item_num = Suppress(Literal('item')) + _lb + integ("num_item") + _rb + _c  + NL
    class_name = Suppress(Literal('class')) + _eg + QuotedString('"')("class") + NL
    item_name =  Suppress(Literal('name')) + _eg + QuotedString('"')("name") + NL
    item_min = Suppress(Literal('xmin')) + _eg + number('item_min') + NL
    item_max = Suppress(Literal('xmax')) + _eg + number('item_max') + NL
    iter_size = Suppress(Literal('intervals: size')) + _eg + number('item_max') + NL

    # TextGrid header, it will look for the text "size" that contains the
    # number of items in the file 
    hdr_size = Suppress(Literal('size')) + _eg + number('total_items') + NL
    hdr_ = Suppress(Literal('item')) + _lb + _rb + _c  + NL

    # TextGrid grammars. In this file format The most inner 
    # that is what elements are the timestamps and its annotations (gram_intervals)
    gram_intevals = OneOrMore(Group(interv_num + interv_min + interv_max + interv_annot))
    gram_items = OneOrMore(Group(item_num + class_name + item_name + item_min + item_max +
                           iter_size + Group(gram_intevals)('interv_data')))
    TextGrid_grammar = hdr_size + hdr_ + gram_items

    # get the real name of the file.
    grid_dir = os.path.dirname(os.path.realpath(grid_file))
    grid_basename = os.path.basename(grid_file)
    grid_ = os.path.join(grid_dir, grid_basename)

    with open(grid_file) as tfile:
	##s = ''.join([str(x) for x in range(10)])
	##print ' '*10 + ' |' + s*5
	##data = tfile.readlines()[6:]
	##print ''.join(['{:10d} | {}'.format(n,x) for n,x in enumerate(data)])

	TextGrid_data = tfile.readlines()[6:]
	decoded_TextGrid = TextGrid_grammar.parseString(''.join(TextGrid_data))

	num_item = decoded_TextGrid.pop(0)
	print("filename,start,end,label")
	while decoded_TextGrid:
	    item = decoded_TextGrid.pop(0)
	    #print item.num_item
	    while item.interv_data:
	        interval = item.interv_data.pop(0)
		_, min_, max_, text_ = interval
		text_ = ' '.join(text_.split())
		if not text_ or text_ in SILS:
		   text = 'SIL'
		else:
		   text = interval.annotation

                # remove all non printable chars as they may cause  
                # problems with the rest of the pipeline
                filtered_text = filter(lambda x: x in string.printable, text)

                print("{},{:f},{:f},{}".format(grid_, float(min_), float(max_),
                      filtered_text.replace(' ','')))


def main():
    import argparse

    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="Parse and dump annotations from Praat's TextGrid files ")
    
    parser.add_argument('textgrid_file', metavar='TEXTGRID_FILE', \
            nargs=1, help='File in TextGrid format')

    args = parser.parse_args()

    textgrid_file = args.textgrid_file[0]

    dump_textgrid2csv(textgrid_file)


if __name__ == '__main__':
    main()
