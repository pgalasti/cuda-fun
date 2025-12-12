#include<fstream>
#include<iostream>
#include<cstring>
#include<stdexcept>

struct DeviceInfo {
  int deviceNum;
  char szDeviceName[256];
  char szVersion[128];
  unsigned long sharedMem;
  unsigned long maxThreads;
};

void fetchDeviceInfo(DeviceInfo* pDeviceInfo);
void createFile(const char* pszFileName, const DeviceInfo& deviceInfo);

int main(int argc, char** argv) {

  DeviceInfo deviceInfo;
  memset(&deviceInfo, 0, sizeof(DeviceInfo));
  
  fetchDeviceInfo(&deviceInfo);
  createFile("device.csv", deviceInfo);
  
  return 0;
}

void createFile(const char* pszFileName, const DeviceInfo& deviceInfo) {


  std::ofstream outputFile(pszFileName);

  if(!outputFile) {
    throw std::runtime_error("Unable to access file for writing!");
  }

  outputFile << "Device Number" << ","
	     << "Device Name" << ","
	     << "Version Number" << ","
	     << "Shared Memory" << ","
	     << "Max Threads (Per Block)"
	     << std::endl;

  outputFile << deviceInfo.deviceNum << ","
	     << deviceInfo.szDeviceName << ","
	     << deviceInfo.szVersion << ","
	     << deviceInfo.sharedMem << ","
	     << deviceInfo.maxThreads
	     << std::endl;

  outputFile.close();
}

void fetchDeviceInfo(DeviceInfo* pDeviceInfo) {

  cudaDeviceProp deviceProp; 
  int device = 0;
  cudaGetDeviceProperties(&deviceProp, device);

  pDeviceInfo->deviceNum = device;
  strncpy(pDeviceInfo->szDeviceName, deviceProp.name, 255);
  sprintf(pDeviceInfo->szVersion, "%d.%d", deviceProp.major, deviceProp.minor);
  pDeviceInfo->sharedMem = deviceProp.sharedMemPerBlock;
  pDeviceInfo->maxThreads = deviceProp.maxThreadsPerBlock;
}
