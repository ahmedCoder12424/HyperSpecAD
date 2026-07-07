# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -O0 -std=c++17
CXXFLAGS += -Wno-unused-but-set-variable


# Source files
SRCS = main.cpp louvain.cpp utils.cpp

# Object files (replace .cpp with .o)
OBJS = $(SRCS:.cpp=.o)

# Header files
HEADERS = louvain.h struct.h

# Output binary name
TARGET = louvain

# Default target
all: $(TARGET)

# Link the object files into the final binary
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) -ltbb

# Compile each .cpp into .o
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)
