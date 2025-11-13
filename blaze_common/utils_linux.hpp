#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <sstream>

// Helper: Execute a shell command and return stdout as a string
inline std::string exec_command(const std::string& cmd) {
    std::array<char, 4096> buffer;
    std::string result;
    //std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(cmd.c_str(), "r"), static_cast<int(*)(FILE*)>(pclose));
    if (!pipe) return "";
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// Find the first /dev/video* device whose v4l2-ctl -D output contains 'src'
inline std::string get_video_dev_by_name(const std::string& src) {
    std::vector<std::string> devices;
    for (const auto& entry : std::filesystem::directory_iterator("/dev")) {
        std::string path = entry.path().string();
        if (path.rfind("/dev/video", 0) == 0) {
            devices.push_back(path);
        }
    }
    std::sort(devices.begin(), devices.end());
    for (const auto& dev : devices) {
        std::string cmd = "v4l2-ctl -d " + dev + " -D";
        std::string output = exec_command(cmd);
        std::istringstream iss(output);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.find(src) != std::string::npos) {
                return dev;
            }
        }
    }
    return "";
}

// Find the first /dev/media* device whose media-ctl -p output contains 'src'
inline std::string get_media_dev_by_name(const std::string& src) {
    std::vector<std::string> devices;
    for (const auto& entry : std::filesystem::directory_iterator("/dev")) {
        std::string path = entry.path().string();
        if (path.rfind("/dev/media", 0) == 0) {
            devices.push_back(path);
        }
    }
    std::sort(devices.begin(), devices.end());
    for (const auto& dev : devices) {
        std::string cmd = "media-ctl -d " + dev + " -p";
        std::string output = exec_command(cmd);
        std::istringstream iss(output);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.find(src) != std::string::npos) {
                return dev;
            }
        }
    }
    return "";
}
