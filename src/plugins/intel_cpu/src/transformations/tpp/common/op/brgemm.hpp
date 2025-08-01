// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "modifiers.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "snippets/op/brgemm.hpp"

namespace ov::intel_cpu::tpp::op {

/**
 * @interface BrgemmTPP
 * @brief BrgemmTPP is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support of several precisions on plugin level
 * @ingroup snippets
 */
class BrgemmTPP : virtual public modifier::TensorProcessingPrimitive, public snippets::op::Brgemm {
public:
    OPENVINO_OP("Brgemm", "TppOpset", snippets::op::Brgemm);

    BrgemmTPP(const Output<Node>& A,
              const Output<Node>& B,
              size_t offset_a = 0,
              size_t offset_b = 0,
              size_t offset_c = 0,
              std::vector<size_t> layout_a = {},
              std::vector<size_t> layout_b = {},
              std::vector<size_t> layout_c = {},
              float beta = 1);
    BrgemmTPP(const Output<Node>& A,
              const Output<Node>& B,
              const PortDescriptor& desc_a,
              const PortDescriptor& desc_b,
              const PortDescriptor& desc_c,
              std::vector<size_t> layout_a = {},
              std::vector<size_t> layout_b = {},
              std::vector<size_t> layout_c = {},
              float beta = 1);
    BrgemmTPP() = default;

    [[nodiscard]] std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    [[nodiscard]] float get_beta() const {
        return m_beta;
    }
    void set_beta(float beta) {
        m_beta = beta;
    }

private:
    float m_beta = 0.F;
};

}  // namespace ov::intel_cpu::tpp::op
