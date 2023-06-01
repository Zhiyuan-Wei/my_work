CC = g++
CFLAGS =-ggdb3 -std=c++11 -lcurl -pthread

SRC = $(wildcard *.cpp)
OBJ = $(patsubst %.cpp, %.o, $(SRC))
Target = main

$(Target): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rf $(Target) *.o

run :
	./$(Target)

.PHONY: clean run




