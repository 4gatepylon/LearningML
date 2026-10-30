# nvcc comes from PATH and the GPU from CUDA_VISIBLE_DEVICES (both set by `source .env`).
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=native
LIBS := -lcublas -lgtest -lpthread
BUILD := .build

.PHONY: help clean
.PRECIOUS: $(BUILD)/%

help:
	@echo "make run-<name> [MATMUL_ARGS='--n 4096 --tries 50']   Build <name>.cu, test it, then benchmark it"
	@echo "make test-<name>                                     Build <name>.cu and run only the tests"
	@echo "make clean                                           Remove $(BUILD)/"

$(BUILD)/%: %.cu harness.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS)

run-%: $(BUILD)/%
	./$< $(MATMUL_ARGS)

test-%: $(BUILD)/%
	./$< --no-bench $(MATMUL_ARGS)

clean:
	rm -rf $(BUILD)
