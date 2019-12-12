.PHONY: test

test:
	@echo "Testing power flow module"
	julia --project=./ test/test_pf.jl
