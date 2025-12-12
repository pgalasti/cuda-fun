#include<stdio.h>

__global__ void sayHelloFromGPULand() {
  printf("Printf'ing off of the GPU!\n");
}

int main(int argc, char** argv) {

  printf("Printf'ing off the CPU!\n");
  
  sayHelloFromGPULand<<<3,10>>>();
  cudaDeviceSynchronize();

  return 0;
}
