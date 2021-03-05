SOURCES := $(wildcard src/*/*.jl) $(wildcard test/*/*.jl)

image: exapf.so

exapf.so: $(SOURCES)
	julia --project deps/createsysimage.jl

.PHONY: image