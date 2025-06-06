// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "registry.hpp"
#include "intel_gpu/primitives/rnn.hpp"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/gru_seq_onednn.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<gru_seq>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::GRUSeqImplementationManager, shape_types::static_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
