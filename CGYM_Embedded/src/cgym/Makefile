all: cgym_sample

CC = g++
CFLAGS = -O3 -Wall

cgym_sample: cgym_sample.o cgym.o
	$(CC) $(CFLAGS) -o $@ $^

cgym_sample.o cgym.o: cgym.h

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o cgym_sample
