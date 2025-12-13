#include <g-lib/util/Stopwatch.h>

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cstdint>

long long convertArg(const char* arg);
bool checkPrimeNumberCPU(const long long num);
float checkPrimeNumberGPU(const long long start, const long long max);
__global__ void checkPrimeNumberKernal(long long start, long long end);


int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: program <number>" << std::endl;
    return 1;
  }
  
  const long long value = convertArg(argv[1]);
  if(value == -1) {
    std::cerr << "Invalid param!" << std::endl;
    return 1;
  }
  
  std::cout << "Max value: " << value << std::endl;

  GLib::Util::Stopwatch swCPU("CPU");
  swCPU.Start();
  for(long long i = 3; i < value; ++i) {
    bool isPrime = checkPrimeNumberCPU(i);
  }
  const std::int64_t cpuEnd = swCPU.Current();
  
  const float gpuTiming = checkPrimeNumberGPU(3, value); 
 
  // Maybe I should try out breaking the range up between number of CPU cores
  // and having a thread handle a chunk concurrently and compare? 
  std::cout << swCPU.GetLabel() << " timing (in milliseconds): " << cpuEnd/1000000 << std::endl;
  std::cout << "GPU timing (in milliseconds): " << gpuTiming << std::endl;

  return 0;
}

long long convertArg(const char* arg) {
  char* end = nullptr;
  errno = 0;

  const long long value = std::strtoll(arg, &end, 10);

  if (errno != 0 || end == arg || *end != '\0') {
    std::cerr << "Invalid number: " << arg << "\n";
    return -1;
  }

  return value;
}

bool checkPrimeNumberCPU(const long long num) {
  if(num < 2) return false;
  if(num == 2) return true;
  if(num % 2 == 0) return false;

  for(long long i = 3; i*i <= num; i+=2) {
    if(num %i == 0) return false;
  }

  return true;
}

float checkPrimeNumberGPU(const long long start, const long long max) {
  int threadsPerBlock = 256;
  int totalNumbers = (max-start)/2+1;
  int blocksPerGrid = (totalNumbers + threadsPerBlock -1)/threadsPerBlock;

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  cudaEventRecord(startEvent, 0);

  checkPrimeNumberKernal<<<blocksPerGrid, threadsPerBlock>>>(start, max);

  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);

  float gpuDuration = 0;
  cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  return gpuDuration;
}

__global__ void checkPrimeNumberKernal(long long start, long long end) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  long long num = start + (tid*2);
  if(num > end) return;
  bool isPrime = true;
  
  if(num<2) {
    isPrime = false;
    return;
  }

  if(num == 2) {
    isPrime = true;
    return;
  }

  if(num % 2 == 0) {
    isPrime = false;
    return;
  }

  for(long long i = 3; i * i <= num; i +=2) {
    if(num%i == 0) {
      isPrime = false;
      break;
    }
  }
}
