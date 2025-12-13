NVCC            := nvcc

NVCC_FLAGS      := -std=c++17 -lg-lib

TARGET          := program

SRC             := prime-numbers.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)
