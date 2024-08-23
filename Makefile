# The name of the main executable
TARGET=out/mpt_nn

# Flags and options, change as required
OPTIMIZE=-O3
# Flags for all languages
CPPFLAGS=-ggdb $(OPTIMIZE) -Wall -Werror -MMD -MP -fopenmp -I.
# Flags for C only
CFLAGS=-Wmissing-prototypes
# Flags for the linker
LDFLAGS=-fopenmp
# Additional linker libs
LDLIBS=-lm

# Compilers
CC=gcc

# Directories
SRC_DIR=source
OUT_DIR=out

# The sources that make up the main executable.
SRCS=$(filter-out %_test.c,$(wildcard $(SRC_DIR)/*.c))

# We make up the objects by replacing the source directory and .c suffixes with the out directory and .o suffixes
OBJS=$(patsubst $(SRC_DIR)/%.c,$(OUT_DIR)/%.o,$(SRCS))

# The dependency files
DEPS=$(patsubst $(SRC_DIR)/%.c,$(OUT_DIR)/%.d,$(SRCS))

# The first target (all) is always the default target
.PHONY: all
all: $(OUT_DIR) build

# Create the output directory
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Our build target depends on the real target
.PHONY: build
build: $(TARGET)

# Our target is built up from the objects
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $(TARGET) $(LDLIBS)

# Rule to compile .c files into .o files in the out directory
$(OUT_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Cleanup all generated files
clean:
	rm -Rf $(OUT_DIR) build *.results

-include $(DEPS)
