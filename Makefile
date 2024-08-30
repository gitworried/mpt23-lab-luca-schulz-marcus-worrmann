# The name of the main executable
TARGET=out/mpt_nn
TEST_TARGET=out/mpt_nn_test

# Benchmark output files
HELGRIND_OUTPUT=out/benchmark_helgrind.log
VALGRIND_OUTPUT=out/benchmark_valgrind.log

# Flags and options
OPTIMIZE=-O3
CPPFLAGS=-ggdb $(OPTIMIZE) -Wall -Werror -MMD -MP -fopenmp -I.
CFLAGS=-Wmissing-prototypes
LDFLAGS=-fopenmp
LDLIBS=-lm

# Compilers
CC=gcc

# Directories
SRC_DIR=source
OUT_DIR=out

# The sources that make up the main executable.
SRCS=$(filter-out %_test.c,$(wildcard $(SRC_DIR)/*.c))

# The source file for the tests
TEST_SRCS=$(SRC_DIR)/mpt_nn_test.c

# The objects corresponding to the source files
OBJS=$(patsubst $(SRC_DIR)/%.c,$(OUT_DIR)/%.o,$(SRCS))

# The dependency files
DEPS=$(patsubst $(SRC_DIR)/%.c,$(OUT_DIR)/%.d,$(SRCS))

# The test object files
TEST_OBJS=$(OUT_DIR)/mpt_nn_test.o
TEST_DEPS=$(filter-out out/main.o,$(OBJS))

# Default target: Build the main program and the tests
.PHONY: all
all: build test

# Ensure the output directory exists
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Build the main program
.PHONY: build
build: $(TARGET)

# The main program depends on the out directory being created
$(TARGET): $(OUT_DIR) $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $(TARGET) $(LDLIBS)

# Compile .c files to .o files
$(OUT_DIR)/%.o: $(SRC_DIR)/%.c | $(OUT_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Compile the test file
$(TEST_OBJS): $(TEST_SRCS) | $(OUT_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $(TEST_SRCS) -o $(TEST_OBJS)

# Link and create the test executable
$(TEST_TARGET): $(TEST_OBJS) $(TEST_DEPS)
	$(CC) $(LDFLAGS) $(TEST_OBJS) $(TEST_DEPS) -o $(TEST_TARGET) $(LDLIBS)

# Run the tests
.PHONY: test
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Benchmark with Helgrind
.PHONY: benchmark_helgrind
benchmark_helgrind: $(TEST_TARGET)
	valgrind --tool=helgrind --log-file=$(HELGRIND_OUTPUT) ./$(TEST_TARGET)
	@echo "Helgrind benchmark results saved to $(HELGRIND_OUTPUT)"

# Benchmark with Valgrind (Memory)
.PHONY: benchmark_valgrind
benchmark_valgrind: $(TEST_TARGET)
	valgrind --leak-check=full --track-origins=yes --log-file=$(VALGRIND_OUTPUT) ./$(TEST_TARGET)
	@echo "Valgrind benchmark results saved to $(VALGRIND_OUTPUT)"

# Cleanup
clean:
	rm -Rf $(OUT_DIR) build *.results

-include $(DEPS)

