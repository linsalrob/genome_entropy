CC ?= cc
CFLAGS ?= -O3 -DNDEBUG
CPPFLAGS ?=
LDFLAGS ?=

TARGET := extract_genbank_locus

.PHONY: all clean test

all: $(TARGET)

$(TARGET): extract_genbank_locus.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 -Wall -Wextra -Wpedantic $< $(LDFLAGS) -o $@

test: $(TARGET)
	../tests/test_extract_genbank_locus.sh ./$(TARGET)

clean:
	$(RM) $(TARGET)
