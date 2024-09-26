# The name of the main executable
TARGET=out/mpt_nn
TEST_TARGET=out/mpt_nn_test

# Benchmark output files
BENCHMARK_RESULT=benchmarks/benchmark_results.md
PLOT_SCRIPT=scripts/mpt_nn_plot.R
PLOT_OUTPUT=benchmarks/benchmark_plot.png

# Flags and options
OPTIMIZE=-O3
CPPFLAGS=-ggdb -O3 -march=native -Wall -Werror -MMD -MP -fopenmp -I. -lm
CFLAGS=-Wmissing-prototypes
LDFLAGS=-fopenmp
LDLIBS=-lm

# Compilers
CC=gcc

# Directories
SRC_DIR=source
OUT_DIR=out
LOG_DIR=logs
HELGRIND_DIR=helgrind
VALGRIND_DIR=valgrind

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

# Ensure the logs directory exists
$(LOG_DIR):
	mkdir -p $(LOG_DIR)

# Ensure helgrind directory exists
$(LOG_DIR)/$(HELGRIND_DIR):
	mkdir -p $(LOG_DIR)/$(HELGRIND_DIR)

# Ensure valgrind directory exists
$(LOG_DIR)/$(VALGRIND_DIR):
	mkdir -p $(LOG_DIR)/$(VALGRIND_DIR)

# Ensure the benchmarks directory exists
benchmarks:
	mkdir -p benchmarks

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

# Test for thread errors with Helgrind
.PHONY: helgrind
helgrind: $(TEST_TARGET) $(LOG_DIR)/$(HELGRIND_DIR)
	valgrind --tool=helgrind --log-file=$(LOG_DIR)/$(HELGRIND_DIR)/helgrind.log ./$(TEST_TARGET)
	valgrind --tool=helgrind --log-file=$(LOG_DIR)/$(HELGRIND_DIR)/helgrind_sequential.log ./$(TARGET) -m1 -t10000 -i784 -h10 -o10 -e10 -l0.1 -d0.0
	valgrind --tool=helgrind --log-file=$(LOG_DIR)/$(HELGRIND_DIR)/helgrind_parallel.log ./$(TARGET) -m2 -t10000 -i784 -h10 -o10 -e10 -l0.1 -d0.0
	valgrind --tool=helgrind --log-file=$(LOG_DIR)/$(HELGRIND_DIR)/helgrind_simd.log ./$(TARGET) -m3 -t10000 -i784 -h10 -o10 -e10 -l0.1 -d0.0
	@echo "Helgrind thread error results saved to $(LOG_DIR)"

# Test for memory errors with Valgrind 
.PHONY: valgrind
valgrind: $(TEST_TARGET) $(LOG_DIR)/$(VALGRIND_DIR)
	valgrind -s --leak-check=full --show-leak-kinds=all  --track-origins=yes --log-file=$(LOG_DIR)/$(VALGRIND_DIR)/valgrind.log ./$(TEST_TARGET)
	valgrind -s --leak-check=full --show-leak-kinds=all  --track-origins=yes --log-file=$(LOG_DIR)/$(VALGRIND_DIR)/valgrind_sequential.log ./$(TARGET) -m1 -t10000 -i784 -h10 -o10 -e10 -l0.1 -d0.0
	valgrind -s --leak-check=full --show-leak-kinds=all  --track-origins=yes --log-file=$(LOG_DIR)/$(VALGRIND_DIR)/valgrind_parallel.log ./$(TARGET) -m2 -t10000 -i784 -h10 -o10 -e10 -l0.1 -d0.0 -n4 
	valgrind -s --leak-check=full --show-leak-kinds=all  --track-origins=yes --log-file=$(LOG_DIR)/$(VALGRIND_DIR)/valgrind_simd.log ./$(TARGET) -m3 -t10000 -i784 -h10 -o10 -e10 -l0.1 -d0.0 -n4 
	@echo "Valgrind memory error results saved to $(LOG_DIR)"

# Benchmark using hyperfine
benchmark: $(TARGET) | benchmarks
#	@echo "Running benchmark..."
#	@echo "System Information:" > benchmarks/system_info.txt
#	@echo "===================" >> benchmarks/system_info.txt
#	@echo "OS and Kernel:" >> benchmarks/system_info.txt
#	@uname -a >> benchmarks/system_info.txt
#	@echo "\nCPU Information:" >> benchmarks/system_info.txt
#	@lscpu >> benchmarks/system_info.txt
#	@echo "\nMemory Information:" >> benchmarks/system_info.txt
#	@free -h >> benchmarks/system_info.txt
#	@echo "\nFull system information can be found in benchmarks/system_info.txt"
	hyperfine \
		--warmup 3 \
		--show-output \
		--export-markdown $(BENCHMARK_RESULT) \
		'./out/mpt_nn -m1 -t60000 -i784 -h128 -o10 -e10 -l0.01 -d0.1' \
		'./out/mpt_nn -m2 -t60000 -i784 -h128 -o10 -e10 -l0.01 -d0.1' \
		'./out/mpt_nn -m3 -t60000 -i784 -h128 -o10 -e10 -l0.01 -d0.1'

# Plot generation target	
plot: $(BENCHMARK_RESULT)
	@echo "Running R script to generate the plot..."
	Rscript $(PLOT_SCRIPT)

# Ensure the benchmark has been run before plotting
$(BENCHMARK_RESULT):
	@echo "Benchmark results not found, running make benchmark..."
	make benchmark

# Cleanup
clean:
	rm -Rf $(OUT_DIR) build *.results $(LOG_DIR) documentation doxygen benchmarks $(PLOT_OUTPUT)

-include $(DEPS)

