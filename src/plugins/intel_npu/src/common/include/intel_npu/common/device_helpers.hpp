// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>

#include "intel_npu/npu_private_properties.hpp"

namespace intel_npu {

namespace utils {
bool isNPUDevice(const uint32_t deviceId);
uint32_t getSliceIdBySwDeviceId(const uint32_t swDevId);
std::string getPlatformByDeviceName(const std::string& deviceName);
std::string getCompilationPlatform(const std::string_view platform,
                                   const std::string& deviceId,
                                   std::vector<std::string> availableDevicesNames);
}  // namespace utils

}  // namespace intel_npu
