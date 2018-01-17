.PHONY: test

test: mcr/extract_features.py mcr/reduce_features.py mcr/util.py
	python mcr/extract_features.py -v 

clean:
	python setup.py clean

dev:
	python setup.py develop 

