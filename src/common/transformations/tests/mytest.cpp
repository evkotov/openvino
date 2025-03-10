#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"

#include <iomanip>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <sstream>



#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>

#include "openvino/pass/manager.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/pass/visualize_tree.hpp"

using namespace testing;
using namespace std;
using namespace ov;

namespace {
#if 0
using NodeDecorator = std::function<std::string(const std::shared_ptr<ov::Node>& node)>;
using NodeDecorators = std::vector<NodeDecorator>;

class NodeHighliter {
public:
    NodeHighliter(const std::string& node_name) : _node_name(node_name) {}
    std::string operator()(const std::shared_ptr<ov::Node>& node) {
        if (node->get_friendly_name() == _node_name) {
            return "style=filled";
        }
        return "";
    }

private:
    const std::string _node_name;
};

class SubGraphDotGenerator {
public:
    SubGraphDotGenerator() = default;

    void save(const std::string &filename) {
        std::ofstream outfile;
        outfile.open(filename, std::fstream::out | std::fstream::trunc);
        if (!outfile.is_open()) {
            std::cerr << "cannot open file " << filename << std::endl;
            return;
        }
        outfile << _contents.str();
        outfile.close();
    }

    void start() {
        _contents << "digraph model {" << std::endl;
        _contents << "node [shape = box, ordering=in];" << std::endl;
    }

    void finalize() {
        _contents << "}" << std::endl;
    }

    void add(NodeDecorator decorator) {
        _node_decorators.push_back(decorator);
    }

    void add_node(const std::shared_ptr<ov::Node> &node) {
        _contents << get_internal_node_name(node) << " [shape=box label=\"";
        _contents << "name: " << node->get_friendly_name() << std::endl;
        _contents << "type: " << node->get_type_name() << std::endl;

        append_label(node);

        for (size_t input_idx = 0;
             input_idx < node->get_input_size(); ++input_idx) {
            _contents << "input #" << input_idx << ": ";
            _contents << node->get_input_element_type(input_idx) << " ";
            _contents << node->get_input_partial_shape(input_idx) << std::endl;
        }
        for (size_t output_idx = 0;
             output_idx < node->get_output_size(); ++output_idx) {
            _contents << "output #" << output_idx << ": ";
            _contents << node->get_output_element_type(output_idx) << " ";
            _contents << node->get_output_partial_shape(output_idx)
                      << std::endl;
        }
        _contents << "\" ";

        for (const auto &node_decorator: _node_decorators) {
            _contents << node_decorator(node);
        }

        _contents << "]" << std::endl;
    }

    void add_connection(const std::shared_ptr<ov::Node> &from,
                        size_t from_output_index,
                        const std::shared_ptr<ov::Node> &to,
                        size_t to_input_index) {
        _contents << get_internal_node_name(from) << " -> "
                  << get_internal_node_name(to);
        _contents << " [label=\"" << from_output_index << "->" << to_input_index
                  << "\"]";
        _contents << ";" << std::endl;
    }

protected:
    template<typename T>
    static std::string
    get_const_value(const std::shared_ptr<ov::opset12::Constant> &node) {
        std::stringstream value_stream;
        const auto value = node->cast_vector<T>();
        value_stream << "[";
        for (size_t i = 0;
             i < std::min(static_cast<size_t>(4), value.size()); ++i) {
            if (i)
                value_stream << ",";
            value_stream << value[i];
        }
        if (value.size() > 4) {
            value_stream << "...";
        }
        value_stream << "]" << std::endl;
        return value_stream.str();
    }

    void append_label_constant_value(
            const std::shared_ptr<ov::opset12::Constant> &node) {
        switch (node->get_output_element_type(0)) {
            case ov::element::Type_t::dynamic:
                _contents << "[ dynamic value ]";
                break;
            case ov::element::Type_t::nf4:
                _contents << "[ nf4 value ]";
                break;
            case ov::element::Type_t::f8e4m3:
                _contents << "[ f8e4m3 value ]";
                break;
            case ov::element::Type_t::f8e5m2:
                _contents << "[ f8e5m2 value ]";
                break;
            case ov::element::Type_t::string:
                _contents << "[ string value ]";
                break;
            case ov::element::Type_t::f4e2m1:
                _contents << "[ f4e2m1 value ]";
                break;
            case ov::element::Type_t::f8e8m0:
                _contents << "[ f8e8m0 value ]";
                break;
            case ov::element::Type_t::bf16:
            case ov::element::Type_t::f16:
            case ov::element::Type_t::f32:
            case ov::element::Type_t::f64:
                _contents << node->get_output_element_type(0) <<
                          " cast to double: " << get_const_value<double>(node);
                break;
            case ov::element::Type_t::i4:
            case ov::element::Type_t::i8:
            case ov::element::Type_t::i16:
            case ov::element::Type_t::i32:
            case ov::element::Type_t::i64:
                _contents << node->get_output_element_type(0) <<
                          " cast to int64_t: "
                          << get_const_value<int64_t>(node);
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
                _contents << node->get_output_element_type(0) <<
                          " cast to uint64_t: "
                          << get_const_value<uint64_t>(node);
                break;
            default:
                _contents << "[ undefined value ]";
                break;
        }
    }

        void append_label(const std::shared_ptr<ov::Node> &node) {
            if (const auto &const_node = ov::as_type_ptr<ov::opset12::Constant>(
                    node))
                append_label_constant_value(const_node);
        }
        std::string
        get_internal_node_name(const std::shared_ptr<ov::Node> &node) {
            std::stringstream ss;
            ss << "node_" << node->get_instance_id();
            return ss.str();
        }

        private:
        NodeDecorators _node_decorators;
        std::stringstream _contents;
    };

    template<typename GraphVisitor>
    void traverse_subgraph(const std::shared_ptr<ov::Node> &current_node,
                           size_t max_depth,
                           std::unordered_set<size_t> &visited,
                           GraphVisitor &graph_visitor) {
        if (!max_depth)
            return;

        if (visited.find(current_node->get_instance_id()) != visited.end()) {
            return;
        }
        visited.insert(current_node->get_instance_id());

        for (const auto &input_node_output: current_node->input_values()) {
            const auto &input_node = input_node_output.get_node_shared_ptr();
            traverse_subgraph(input_node, max_depth - 1, visited,
                              graph_visitor);
        }

        graph_visitor.add_node(current_node);

        if (max_depth > 1) {
            for (size_t input_idx = 0;
                 input_idx < current_node->get_input_size(); ++input_idx) {
                const auto &input_node_output = current_node->get_input_source_output(
                        input_idx);
                const auto &input_node = input_node_output.get_node_shared_ptr();
                graph_visitor.add_connection(input_node,
                                             input_node_output.get_index(),
                                             current_node, input_idx);
            }
        }

        for (const auto &output: current_node->outputs()) {
            for (const auto &connected_input: output.get_target_inputs()) {
                const auto &output_node = connected_input.get_node()->shared_from_this();
                traverse_subgraph(output_node, max_depth - 1, visited,
                                  graph_visitor);
            }
        }
    }

    std::string getNamePrefixCounter(const std::string& name) {
        static size_t counter = 0;
        std::stringstream current_name;
        current_name << counter << "_" << name;
        ++counter;
        return current_name.str();
    }

    class SubGraphDump {
    public:
        SubGraphDump(const std::shared_ptr<ov::Node>& main_node, size_t max_depth = std::numeric_limits<size_t>::max())
                : _main_node(main_node),
                  _max_depth(max_depth) {}
        void highlight_node(const std::string& node_name) {
            _highlight_node_names.push_back(node_name);
        }
        void dump(const std::string& subgraph_path) {
            SubGraphDotGenerator dot_generator;
            for (const auto& node_name : _highlight_node_names) {
                dot_generator.add(NodeHighliter(node_name));
            }
            std::unordered_set<size_t> visited;
            dot_generator.start();
            traverse_subgraph(_main_node, _max_depth, visited, dot_generator);
            dot_generator.finalize();

            std::string path = subgraph_path;
            if (_enable_dump_path_prefix)
                path = getNamePrefixCounter(path);
            dot_generator.save(path);
        }
        void set_enable_dump_path_prefix(bool enable) {
            _enable_dump_path_prefix = enable;
        }

    private:
        const std::shared_ptr<ov::Node>& _main_node;
        const size_t _max_depth;
        std::vector<std::string> _highlight_node_names;
        bool _enable_dump_path_prefix = {false};
    };
#endif

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


shared_ptr<Node> find_op_by_name(const std::shared_ptr<Model>& model,
                           const std::string& name) {
    for (auto& node: model->get_ops()) {
        if (node->get_friendly_name() == name) {
            return node;
        }
    }

    return {};
}


#if 0
void search_and_print_node_by_name(const std::shared_ptr<Model>& model,
                             const std::string& name) {
    const auto node = find_op_by_name(model, name);
    if (node) {
        std::cout << "FOUND NODE " << name << " " << node->get_type_name() << std::endl;
    } else {
        std::cout << "NOT FOUND NODE " << name << std::endl;
    }
}
#endif

void visit_node_all_parents(const std::shared_ptr<Node>& node, unordered_set<string>& visited) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        const auto &input_node_output = node->get_input_source_output(input_idx);
        const auto &input_node = input_node_output.get_node_shared_ptr();
        if (visited.find(input_node->get_friendly_name()) != visited.end()) {
            continue;
        }
        visited.insert(input_node->get_friendly_name());
        visit_node_all_parents(input_node, visited);
    }
}

void print_node_all_parents(const std::shared_ptr<Node>& node) {
    unordered_set<string> visited;
    visit_node_all_parents(node, visited);
    for (const auto& name: visited) {
        std::cout << name << std::endl;
    }
}

void print_node_parents(const std::shared_ptr<Node>& node) {
    std::cout << "PARENTS OF NODE " << node->get_friendly_name() << ": " << std::endl;
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        const auto &input_node_output = node->get_input_source_output(input_idx);
        const auto &input_node = input_node_output.get_node_shared_ptr();
        std::cout << "\t" << input_node->get_friendly_name() << " " << input_node->get_type_name() << std::endl;
    }
}

void print_node_children(const std::shared_ptr<Node>& node) {
    std::cout << "CHILDREN OF NODE " << node->get_friendly_name() << ": " << std::endl;
    for (const auto& output : node->outputs()) {
        for (const auto& connected_input : output.get_target_inputs()) {
            const auto& output_node = connected_input.get_node()->shared_from_this();
            std::cout << "\t" << output_node->get_friendly_name() << " " << output_node->get_type_name() << std::endl;
        }
    }
}


#if 0
std::shared_ptr<Node> try_fold_unary_output(const std::shared_ptr<Node>& node) {
    const auto& num_outputs = node->get_output_size();
    OPENVINO_ASSERT(num_outputs == 1, "Unary has unexpected number of outputs:" + std::to_string(num_outputs));
    OutputVector output(num_outputs);
    return node->constant_fold(output, node->input_values()) ? output[0].get_node_shared_ptr() : node;
}

template <typename T, typename... Args>
std::shared_ptr<Node> make_try_fold(Args&&... args) {
    auto unary_output_node = std::make_shared<T>(std::forward<Args>(args)...);
    return try_fold_unary_output(unary_output_node);
}

void fold_reshape_node(const std::shared_ptr<op::v1::Reshape>& node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        const auto &input_node_output = node->get_input_source_output(input_idx);
    }

    auto folded = make_try_fold<op::v1::Reshape>(node->get_input_source_output(0),
                                                 node->get_input_source_output(1),
                                                 node->get_special_zero());
    auto folded_const = dynamic_pointer_cast<ov::op::v0::Constant>(folded);
    if (!folded_const) {
        std::cout << "cannot fold reshape node" << std::endl;
        return;
    }
    std::cout << GetConstantValues(folded_const) << std::endl;
}
#endif

void proceed_node(const std::shared_ptr<Model>& model,
                  const std::string& name) {
    auto node = find_op_by_name(model, name);
    if (!node) {
        std::cout << "Node " << name <<  " not found" << std::endl;
        return;
    }

    print_node_all_parents(node);
    //print_node_parents(node);
    //print_node_children(node);

    //fold_reshape_node(dynamic_pointer_cast<ov::op::v1::Reshape>(node));

#if 0
    SubGraphDump dump(node, 5);
    dump.set_enable_dump_path_prefix(true);
    dump.highlight_node(node->get_friendly_name());
    dump.dump("subgraph.dot");
#endif
}

template <class T>
constexpr T emutex_div(const T x, const T y) {
    T z = x / y;
    std::cout << __FILE__ << ":" << __LINE__ << " x = " << x <<
              " y = " << y << " x / y = " << z << " ";
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
    return x / y;
}

}

TEST(TransformationTests, EkotovTest) {
#if 0
    emutex_div(37.0f, 1474.44f);
    emutex_div(85.0f, 1474.44f);
#endif
#if 1
    Core core;
    auto model = core.read_model("before_constantfolding.xml");

    //proceed_node(model, "/crosstransformer/Reshape_17");
#if 1
    pass::Manager manager;
    manager.register_pass<pass::ConstantFolding>();
    manager.run_passes(model);
#endif
#endif
}
