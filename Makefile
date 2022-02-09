JULIA_EXEC = julia 
SOURCES := $(wildcard src/*/*.jl) $(wildcard test/*/*.jl)

benchmark: 
	$(JULIA_EXEC) --project=benchmark benchmark/make.jl

image: exapf.so

exapf.so: $(SOURCES)
	$(JULIA_EXEC) --project deps/createsysimage.jl

.PHONY: image benchmark
