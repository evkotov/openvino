// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface CleanupLoopOffsets
 * @brief Loops are inserted with finalization offsets that reset all managed pointers to their initial values.
 *        This transformation "fuses" the offsets with an outer loop's ptr_increments, and zeroes the offsets before
 *        Results.
 * @ingroup snippets
 */
class CleanupLoopOffsets : public RangedPass {
public:
    OPENVINO_RTTI("CleanupLoopOffsets", "", RangedPass);
    bool run(lowered::LinearIR& linear_ir,
             lowered::LinearIR::constExprIt begin,
             lowered::LinearIR::constExprIt end) override;
};

}  // namespace ov::snippets::lowered::pass
