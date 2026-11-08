CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -fPIC -Wall -Wextra -Werror -Iinclude
LDFLAGS ?= -shared

BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
TARGET_LIB = $(BUILD_DIR)/libcent_runtime.so

SRCS = src/topology.cpp \
       src/sram_buffer.cpp \
       src/isa_emitter.cpp \
       src/runtime_api.cpp

OBJS = $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

HEADERS = include/cent/isa.h \
          include/cent/telemetry.h \
          include/cent/runtime_api.h \
          include/cent/topology.hpp \
          include/cent/sram_buffer.hpp \
          include/cent/isa_emitter.hpp

.PHONY: all shared test demo simulate clean help

help:
	@echo "cent_runtime - Available Make targets:"
	@echo "  make (or make shared) - Build build/libcent_runtime.so"
	@echo "  make test             - Run test suite via unittest"
	@echo "  make demo             - Run Llama-2 7B attention & MLP demo"
	@echo "  make simulate         - Run demo with Ramulator 2.0 cycle simulation"
	@echo "  make clean            - Remove build artifacts, traces, and python cache"
	@echo "  make help             - Display this help message"

all: shared

shared: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: shared
	PYTHONPATH=python python3 -m unittest discover -s tests -v

demo: shared
	PYTHONPATH=python python3 examples/llama2_7b_demo.py

simulate: shared
	PYTHONPATH=python python3 examples/llama2_7b_demo.py --simulate

clean:
	rm -rf $(BUILD_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f *.trace
