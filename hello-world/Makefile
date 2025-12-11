NVCC            := nvcc

NVCC_FLAGS      := -std=c++17 -O2

TARGET          := program

SRC             := hello-world.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)
