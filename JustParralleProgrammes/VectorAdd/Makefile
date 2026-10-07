CC      := clang
CFLAGS  := -std=c11 -O3 -mcpu=apple-m4 -Wall -Wextra
TARGET  := vecadd

# Add n_neon.c and n_thread.c here as you write them
SRCS := main.c harness.c n_scalar.c n_neon.c n_thread.c
OBJS := $(SRCS:.c=.o)
DEPS := $(OBJS:.o=.d)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@

# Scalar baseline: stop clang from turning it into SIMD or unrolling it
n_scalar.o: CFLAGS += -fno-vectorize -fno-slp-vectorize -fno-unroll-loops

%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	mkdir -p results
	./$(TARGET) > results/run.csv

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

-include $(DEPS)