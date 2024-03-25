// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/push_constant_to_subgraph.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"

using MultiSubGraphOp = ov::op::util::MultiSubGraphOp;

static std::shared_ptr<ov::op::v0::Constant> try_constantfold_input(
    const std::shared_ptr<MultiSubGraphOp>& op,
    const MultiSubGraphOp::InputDescription::Ptr& input_desc,
    std::map<ov::Output<ov::Node>, std::shared_ptr<ov::op::v0::Constant>>& cache) {
    const std::string& friendly_name = op->input_value(input_desc->m_input_index).get_node()->get_friendly_name();
#if 0
    std::cout << "[EMUTEX DEBUG] PushConstantToSubgraph friendly_name " << friendly_name << std::endl;
    if (friendly_name != "model/lstm/zeros" &&
            friendly_name != "model/lstm/zeros_1" &&
            friendly_name != "lstm/lstm_cell/kernel" &&
            friendly_name != "lstm/lstm_cell/recurrent_kernel" &&
            friendly_name != "lstm/lstm_cell/bias")
        return nullptr;
    std::cout << "[EMUTEX DEBUG] PushConstantToSubgraph try_constantfold_input " << op->input_value(input_desc->m_input_index).get_node()->get_friendly_name() << std::endl;
#endif
#if 1
    if (!std::dynamic_pointer_cast<MultiSubGraphOp::InvariantInputDescription>(input_desc)) {
        std::cout << "[EMUTEX DEBUG] " << __LINE__ << std::endl;
        return nullptr;
    }
#endif
    const auto outer_input = op->input_value(input_desc->m_input_index);
    auto it = cache.find(outer_input);
    if (it == cache.end()) {
        auto constant = ov::util::constantfold_subgraph(outer_input);
        if (!constant)
            std::cout << "[EMUTEX DEBUG] PushConstantToSubgraph try_constantfold_input FAIL " << outer_input.get_node()->get_friendly_name() << std::endl;
        if (constant) {
            cache.insert({outer_input, constant});
        }
        return constant;
    }
    return it->second;
}

static void replace_body_parameter(const std::shared_ptr<ov::Model>& body,
                                   const std::shared_ptr<ov::op::v0::Parameter>& body_param,
                                   size_t body_parameter_index,
                                   const std::shared_ptr<ov::op::v0::Constant>& constant,
                                   MultiSubGraphOp::MultiSubgraphInputDescriptionVector& descriptions) {
    body_param->output(0).replace(constant);
    body->remove_parameter(body_param);
    // update all input descriptions to reflect that body parameter was removed
    for (auto& desc : descriptions) {
        if (desc->m_body_parameter_index > body_parameter_index) {
            desc->m_body_parameter_index--;
        }
    }
}

static void update_multi_sub_graph_op_inputs(const std::shared_ptr<MultiSubGraphOp>& multi_sub_graph_op,
                                             const std::vector<bool>& remove_inputs_mask) {
    int num_subgraphs = static_cast<int>(multi_sub_graph_op->get_internal_subgraphs_size());
    auto inputs = multi_sub_graph_op->input_values();
    for (size_t i = multi_sub_graph_op->get_input_size(); i > 0; i--) {
        const auto input_index = i - 1;
        if (remove_inputs_mask[input_index]) {
            // remove MultiSubGraphOp's input if it was marked to be removed
            // (meaning it was constfolded and pushed to inner subgraph)
            inputs.erase(inputs.begin() + input_index);

            // update input descriptions to reflect that the input was removed
            for (int body_idx = 0; body_idx < num_subgraphs; body_idx++) {
                auto& descriptions = multi_sub_graph_op->get_input_descriptions(body_idx);
                for (auto& desc : descriptions) {
                    if (desc->m_input_index > input_index) {
                        desc->m_input_index--;
                    }
                }
            }
        }
    }
    multi_sub_graph_op->set_arguments(inputs);
}

bool ov::pass::PushConstantToSubgraph::run_on_model(const std::shared_ptr<Model>& model) {
    RUN_ON_FUNCTION_SCOPE(PushConstantToSubgraph);

    bool result = false;
    for (const auto& op : model->get_ordered_ops()) {
        const auto multi_sub_graph_op = as_type_ptr<op::util::MultiSubGraphOp>(op);
        if (!multi_sub_graph_op) {
            continue;
        }
        std::cout << "[EMUTEX DEBUG] PushConstantToSubgraph multi_sub_graph_op " << multi_sub_graph_op->get_friendly_name() << std::endl;

        // cache for already constant folded inputs
        std::map<ov::Output<ov::Node>, std::shared_ptr<op::v0::Constant>> cache;
        // bitmask describing which MultiSubGraphOp's input to remove
        std::vector<bool> remove_inputs_mask(multi_sub_graph_op->get_input_size(), false);
        int num_subgraphs = static_cast<int>(multi_sub_graph_op->get_internal_subgraphs_size());

        for (int body_idx = 0; body_idx < num_subgraphs; body_idx++) {
            const auto& body = multi_sub_graph_op->get_function(body_idx);
            auto& body_params = body->get_parameters();
            {
                std::cout << "[EMUTEX DEBUG] PushConstantToSubgraph process body " << body_idx << " params ";
                for (const auto& param : body_params) {
                    std::cout << param->get_friendly_name() << " ";
                }
                std::cout << std::endl;
            }
            auto& descriptions = multi_sub_graph_op->get_input_descriptions(body_idx);
            for (auto desc_it = descriptions.begin(); desc_it < descriptions.end();) {
                const auto& desc = *desc_it;
                const auto input_index = desc->m_input_index;
                const auto constant = try_constantfold_input(multi_sub_graph_op, desc, cache);
                if (!constant) {
                    remove_inputs_mask[input_index] = false;
                    desc_it++;
                    continue;
                }
                const auto body_parameter_index = desc->m_body_parameter_index;
                desc_it = descriptions.erase(desc_it);
                auto& body_param = body_params[body_parameter_index];
                std::cout << "[EMUTEX DEBUG] PushConstantToSubgraph replace body_param " << body_param->get_friendly_name() << std::endl;
                replace_body_parameter(body, body_param, body_parameter_index, constant, descriptions);
                remove_inputs_mask[input_index] = true;
                result = true;
            }
        }

        if (result) {
            std::cout << "[EMUTEX DEBUG] PushConstantToSubgraph update_multi_sub_graph_op_inputs" << std::endl;
            update_multi_sub_graph_op_inputs(multi_sub_graph_op, remove_inputs_mask);
        }

        for (int body_idx = 0; body_idx < num_subgraphs; body_idx++) {
            bool model_changed = run_on_model(multi_sub_graph_op->get_function(body_idx));
            result = result || model_changed;
        }
    }

    std::cout << "[EMUTEX DEBUG] PushConstantToSubgraph finished" << std::endl;

    return result;
}
