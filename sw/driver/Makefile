# sw/driver/Makefile -- build the host test for the PS driver skeleton.
#   make          build test_driver
#   make test     build + run against the compiled SRCNN program
#   make clean    remove build artifacts
CC ?= gcc
CFLAGS ?= -std=c99 -Wall -Wextra -O2
SRC = driver.c mock_mmio.c test_driver.c
OBJ = $(SRC:.c=.o)
BIN = test_driver

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ)

%.o: %.c driver.h mock_mmio.h
	$(CC) $(CFLAGS) -c $< -o $@

test: $(BIN)
	./$(BIN) ../out/srcnn

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all test clean
