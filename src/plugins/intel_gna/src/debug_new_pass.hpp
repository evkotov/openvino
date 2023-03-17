// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>

#undef DEBUG_USE_NEW_PASS
#define DEBUG_USE_NEW_PASS 1

#undef DEBUG_VISUALIZE
//#define DEBUG_VISUALIZE 1
#undef DEBUG_VISUALIZETREE
//#define DEBUG_VISUALIZETREE 1

#define EMUTEX_DEBUG_CHECKPOINT std::cout << "[EMUTEX DEBUG] CHECKPOINT " << __FILE__ << ":" << __LINE__ << std::endl;
#define EMUTEX_DEBUG_CHECKPOINT_MESSAGE(message) std::cout << "[EMUTEX DEBUG] CHECKPOINT " << __FILE__ << ":" << __LINE__ << \
                                        " " << message << std::endl;
#define EMUTEX_DEBUG_VALUE(value) std::cout << "[EMUTEX DEBUG] " << __FILE__ << ":" << __LINE__ << " " << #value << " = " << (value) << std::endl;

#include "openvino/pass/manager.hpp"

namespace intel_gna_debug {
void DebugVisualize(ov::pass::Manager& manager, const std::string& name);
void DebugVisualize(const std::shared_ptr<ov::Model>& model, const std::string& name);
} // namespace intel_gna_debug

#define EMUTEX_DEBUG_VISUALIZE(name) intel_gna_debug::DebugVisualize(manager, name);
#define EMUTEX_DEBUG_MODEL_VISUALIZE(model, name) intel_gna_debug::DebugVisualize(model, name);
