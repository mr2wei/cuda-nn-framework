# CUDA Neural Network Regression Test Makefile

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O2 -std=c++11

# Target executable
TARGET = regression_test

# Source files
SOURCES = Regression_test.cu \
          NeuralNetwork.cu \
          LinearLayer.cu \
          ActivationLayer.cu \
          Optimizer.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Rule to build object files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJECTS) $(TARGET)

# Phony targets
.PHONY: all clean 