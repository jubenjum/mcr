#!/usr/bin/env python

''' Transform Praat TextGrid file to csv annotations 

    See the file documentation at:

    http://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html

'''

import os 
from pyparsing import * 


SILS = [ ' ', 'SIL' ]

def dump_textgrid2csv(grid_file):
    '''dump_textgrid2csv decodes TextGrid files and dumps the resutls to the stdout '''


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
   
    # item rules:
    item_num = Suppress(Literal('item')) + _lb + integ("num_item") + _rb + _c  + NL
    class_name = Suppress(Literal('class')) + _eg + QuotedString('"')("class") + NL
    item_name =  Suppress(Literal('name')) + _eg + QuotedString('"')("name") + NL
    item_min = Suppress(Literal('xmin')) + _eg + number('item_min') + NL
    item_max = Suppress(Literal('xmax')) + _eg + number('item_max') + NL
    iter_size = Suppress(Literal('intervals: size')) + _eg + number('item_max') + NL 

    # header rules I using ... from line 7 that contains the size
    hdr_size = Suppress(Literal('size')) + _eg + number('total_items') + NL
    hdr_ = Suppress(Literal('item')) + _lb + _rb + _c  + NL

    # grammars, the most inner elements are the intervals
    gram_intevals = OneOrMore(Group(interv_num + interv_min + interv_max + interv_annot)) 
    gram_items = OneOrMore(Group(item_num + class_name + item_name + item_min + item_max +
                           iter_size + Group(gram_intevals)('interv_data')))
    TextGrid_grammar = hdr_size + hdr_ + gram_items

    # checking the 
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
		if not text_ or text_ == ' ': 
		   text = 'SIL'
		else:
		   text = interval.annotation
                print("{},{:f},{:f},{}".format(grid_, float(min_), float(max_), text))



if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('textgrid_file', metavar='TEXTGRID_FILE', nargs=1, \
            help='File in TextGrid format')
    args = parser.parse_args()
    textgrid_file = args.textgrid_file[0]
    dump_textgrid2csv(textgrid_file)
