// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "openvino/reference/autobroadcast_binop.hpp"

#include <sstream>
#include <iostream>
#include <iomanip>

namespace ov {
namespace reference {
namespace func {
// Usage of custom function instead of lambda gives smaller binary size.

template <typename T>
std::string GetHexData(const T* data) {
    std::stringstream  ss;

    if constexpr (std::is_same_v<T, int>)
        ss << "Type T: int " << sizeof(T) << " bytes ";
    else if constexpr (std::is_same_v<T, double>)
        ss << "Type T: double " << sizeof(T) << " bytes ";
    else if constexpr (std::is_same_v<T, float>)
        ss << "Type T: float " << sizeof(T) << " bytes ";
    else if constexpr (std::is_same_v<T, float16>)
        ss << "Type T: float16 " << sizeof(T) << " bytes ";
    else if constexpr (std::is_same_v<T, bfloat16>)
        ss << "Type T: bfloat16 " << sizeof(T) << " bytes ";
    else
        ss << "Type T: unknown " << sizeof(T) << " bytes ";

    const auto size = sizeof(T);
    auto bytes = reinterpret_cast<const unsigned char*>(data);

    ss << " [";
    for (size_t i = 0; i < size; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i];
        if (i < size - 1) {
            ss << " ";
        }
    }
    ss << "]";
    return ss.str();
}

template <class T>
T power(const T x, const T y) {
    auto z = std::pow(x, y);
    auto z1 = static_cast<T>(z);
    std::cout << __FILE__ << ":" << __LINE__ << " x = " << x << " " << GetHexData(&x) << " " <<
              " y = " << y << " " << GetHexData(&y) << " std::pow(x, y) = " << z << " " << GetHexData(&z) << " ";
    std::cout << "static_cast<T>(std::pow(x, y)) = " << z1 << " " << GetHexData(&z1) << " ";
    if constexpr (std::is_same_v<T, int>)
        std::cout << "Type T: int " << sizeof(T) << " bytes ";
    else if constexpr (std::is_same_v<T, double>)
        std::cout << "Type T: double " << sizeof(T) << " bytes ";
    else if constexpr (std::is_same_v<T, float>)
        std::cout << "Type T: float " << sizeof(T) << " bytes ";
    else if constexpr (std::is_same_v<T, float16>)
        std::cout << "Type T: float16 " << sizeof(T) << " bytes ";
    else if constexpr (std::is_same_v<T, bfloat16>)
        std::cout << "Type T: bfloat16 " << sizeof(T) << " bytes ";
    else
        std::cout << "Type T: unknown " << sizeof(T) << " bytes ";
    std::cout << std::endl;
    return static_cast<T>(std::pow(x, y));
}
}  // namespace func

template <typename T>
void power(const T* arg0, const T* arg1, T* out, size_t count) {
    std::transform(arg0, arg0 + count, arg1, out, func::power<T>);
}

/**
 * @brief Reference implementation of binary elementwise Power operator.
 *
 * @param arg0            Pointer to input 0 data.
 * @param arg1            Pointer to input 1 data.
 * @param out             Pointer to output data.
 * @param arg0_shape      Input 0 shape.
 * @param arg1_shape      Input 1 shape.
 * @param broadcast_spec  Broadcast specification mode.
 */
template <typename T>
void power(const T* arg0,
           const T* arg1,
           T* out,
           const Shape& arg0_shape,
           const Shape& arg1_shape,
           const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, func::power<T>);
}
}  // namespace reference
}  // namespace ov
