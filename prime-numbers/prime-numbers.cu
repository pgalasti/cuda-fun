#include <g-lib/util/Stopwatch.h>

#include <iostream>
#include <string>
#include <cstdint>

long long convertArg(const char* arg);
bool checkPrimeNumberCPU(const long long num);

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

  GLib::Util::Stopwatch swCPU("CPU");
  swCPU.Start();
  const bool cpuResult = checkPrimeNumberCPU(value);
  const std::int64_t cpuEnd = swCPU.Current();
  
  std::cout << value << " being a prime number: " << cpuResult << std::endl;
  
  // Use CUDA method here 
  
  std::cout << swCPU.GetLabel() << " timing (in nanoseconds): " << cpuEnd << std::endl;

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
