#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"

#include <iomanip>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <sstream>

using namespace testing;
using namespace std;
using namespace ov;

namespace {
std::string GetBroadCastSpec(const std::shared_ptr<ov::op::util::BroadcastBase>& node) {
    const auto spec = node->get_broadcast_spec();
    stringstream ss;
    ss << "broadcast spec axis=" << spec.m_axis << " type=" << spec.m_type;
    return ss.str();
}

string GetBroadcastAxes(const std::shared_ptr<ov::op::util::BroadcastBase>& node) {
    stringstream ss;
    const auto axes = node->get_broadcast_axes();
    ss << "broadcast axes " << axes.first;
    ss << " {";
    for (auto item : axes.second) {
        ss << " " << item;
    }
    ss << "}";
    return ss.str();
}

string GetHexData(const void* data, size_t size) {
    auto bytes = static_cast<const unsigned char*>(data);
    stringstream  ss;
    ss << "[";
    for (size_t i = 0; i < size; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i];
        if (i < size - 1) {
            ss << " ";
        }
    }
    ss << "]";
    return ss.str();
}

string GetHexData(const std::shared_ptr<ov::op::v0::Constant>& node) {
    return GetHexData(node->get_data_ptr(), node->get_byte_size());
}

template <typename T>
std::string get_const_value(const std::shared_ptr<ov::op::v0::Constant>& node) {
    std::stringstream value_stream;
    const auto value = node->cast_vector<T>();
    value_stream << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        if (i)
            value_stream << ",";
        value_stream << value[i];
    }
    value_stream << "]" << std::endl;
    return value_stream.str();
}

string GetConstantValues(const std::shared_ptr<ov::op::v0::Constant>& node) {
    std::stringstream ss;
    switch (node->get_output_element_type(0)) {
        case ov::element::Type_t::dynamic:
            ss << "[ dynamic value ]";
            break;
        case ov::element::Type_t::nf4:
            ss << "[ nf4 value ]";
            break;
        case ov::element::Type_t::f8e4m3:
            ss << "[ f8e4m3 value ]";
            break;
        case ov::element::Type_t::f8e5m2:
            ss << "[ f8e5m2 value ]";
            break;
        case ov::element::Type_t::string:
            ss << "[ string value ]";
            break;
        case ov::element::Type_t::f4e2m1:
            ss << "[ f4e2m1 value ]";
            break;
        case ov::element::Type_t::f8e8m0:
            ss << "[ f8e8m0 value ]";
            break;
        case ov::element::Type_t::bf16:
        case ov::element::Type_t::f16:
        case ov::element::Type_t::f32:
        case ov::element::Type_t::f64:
            ss << node->get_output_element_type(0) <<
            " cast to double: " << get_const_value<double>(node);
            break;
        case ov::element::Type_t::i4:
        case ov::element::Type_t::i8:
        case ov::element::Type_t::i16:
        case ov::element::Type_t::i32:
        case ov::element::Type_t::i64:
            ss << node->get_output_element_type(0) <<
            " cast to int64_t: " << get_const_value<int64_t>(node);
            break;
        case ov::element::Type_t::boolean:
        case ov::element::Type_t::u1:
        case ov::element::Type_t::u2:
        case ov::element::Type_t::u3:
        case ov::element::Type_t::u4:
        case ov::element::Type_t::u6:
        case ov::element::Type_t::u8:
        case ov::element::Type_t::u16:
        case ov::element::Type_t::u32:
        case ov::element::Type_t::u64:
            ss << node->get_output_element_type(0) <<
            " cast to uint64_t: " << get_const_value<uint64_t>(node);
            break;
        default:
            ss << "[ undefined value ]";
            break;
    }
    return ss.str();
}

string GetModelInfo(const std::shared_ptr<Model>& model) {
    stringstream  ss;
    unordered_set<string> op_types;
    unordered_map<string, int> ops;
    for (auto node: model->get_ops()) {
        if (auto broadcast_node = dynamic_pointer_cast<ov::op::util::BroadcastBase>(node)) {
            ss << node->get_friendly_name() << " " << node->get_type_name() <<
                 " spec " << GetBroadCastSpec(broadcast_node) << " " <<
                 GetBroadcastAxes(broadcast_node) << std::endl;
        }
        if (auto concat_node = dynamic_pointer_cast<ov::op::v0::Concat>(node)) {
            ss << node->get_friendly_name() << " " << node->get_type_name() <<
                 " axis " << concat_node->get_axis() << std::endl;
        }
        if (auto const_node = dynamic_pointer_cast<ov::op::v0::Constant>(node)) {
            ss << node->get_friendly_name() << " " << node->get_type_name() <<
                 " " <<
               #if 0
                 GetHexData(const_node)
               #else
                 GetConstantValues(const_node)
               #endif
                 << std::endl;
        }

        const string type_name = node->get_type_name();
        if (ops.find(type_name) != ops.end()) {
            ops[type_name]++;
        } else {
            ops[type_name] = 1;
        }
        op_types.insert(type_name);
    }

    vector<string> v_types(op_types.begin(), op_types.end());
    sort(v_types.begin(), v_types.end());
    for (auto type: v_types) {
        ss << type << " " << ops[type] << std::endl;
    }

    return ss.str();
}

}

TEST(TransformationTests, EkotovTest) {
    Core core;
    auto model = core.read_model("before_constantfolding.xml");

    std::cout << "BEFORE START" << std::endl;
    std::cout << GetModelInfo(model) << std::endl;
    std::cout << "BEFORE END" << std::endl;

    pass::Manager manager;
    manager.register_pass<pass::ConstantFolding>();
    manager.run_passes(model);

    std::cout << "AFTER START" << std::endl;
    std::cout << GetModelInfo(model) << std::endl;
    std::cout << "AFTER END" << std::endl;
}
