// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "edge.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <new>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/node_config.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/util/pp.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
#include "weights_cache.hpp"

using namespace dnnl;

namespace ov::intel_cpu {

Edge::Edge(const NodePtr& parent, const NodePtr& child, int pr_port, int ch_port)
    : parent(parent),
      child(child),
      parent_port(pr_port),
      child_port(ch_port) {}

NodePtr Edge::getParent() const {
    auto parentPtr = parent.lock();
    OPENVINO_ASSERT(parentPtr, "Edge contains empty parent node");
    return parentPtr;
}

NodePtr Edge::getChild() const {
    auto childPtr = child.lock();
    OPENVINO_ASSERT(childPtr, "Edge contains empty child node");
    return childPtr;
}

bool Edge::isUseExternalMemory() const {
    return useExternalMemory;
}

bool Edge::isDropped() const {
    bool not_in_parent = true;
    bool not_in_child = true;

    auto parent_ptr = parent.lock();
    if (parent_ptr) {
        for (auto& edge : parent_ptr->childEdges) {
            if (edge.lock().get() == this) {
                not_in_parent = false;
            }
        }
    }

    auto child_ptr = child.lock();
    if (child_ptr) {
        for (auto& edge : child_ptr->parentEdges) {
            if (edge.lock().get() == this) {
                not_in_child = false;
            }
        }
    }
    return not_in_parent && not_in_child;
}

void Edge::collectConsumers(std::vector<NodePtr>& result) const {
    auto add_result_node = [](std::vector<NodePtr>& result, const NodePtr& node) -> bool {
        if (Type::ShapeOf == node->getType()) {
            // ShapeOf doesn't actually read the data, it only reads shape
            return false;
        }
        result.push_back(node);
        return true;
    };
    auto childNode = this->getChild();
    if (childNode->getChildEdges().empty()) {
        add_result_node(result, childNode);
        return;
    }

    if (this->inPlace(LOOK_DOWN)) {
        if (auto* peerChildSPD = childNode->getSelectedPrimitiveDescriptor()) {
            auto peerOutputNum = this->getOutputNum();
            auto peerInPlacePort = peerChildSPD->getConfig().inConfs[peerOutputNum].inPlace();
            auto vecChildEdges = getChild()->getChildEdgesAtPort(peerInPlacePort);
            for (const auto& childEdge : vecChildEdges) {
                childEdge->collectConsumers(result);
            }
        }
    } else {
        if (!add_result_node(result, childNode)) {
            return;
        }

        // collect consumers in case of an upstream in-place memory reference
        if (auto* peerChildSPD = childNode->getSelectedPrimitiveDescriptor()) {
            auto&& conf = peerChildSPD->getConfig();
            for (size_t i = 0; i < conf.outConfs.size(); i++) {
                const auto peerOutInPlacePort = conf.outConfs[i].inPlace();
                if (peerOutInPlacePort == this->getOutputNum()) {
                    for (auto&& childEdge : childNode->getChildEdgesAtPort(i)) {
                        childEdge->collectConsumers(result);
                    }
                }
            }
        }
    }
}

bool Edge::enforceReorder() {
    auto parentNode = getParent();
    auto* parentSPD = parentNode->getSelectedPrimitiveDescriptor();
    auto childNode = getChild();
    auto* childSPD = childNode->getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(parentSPD && childSPD,
                    "Cannot make a decision about reorder. Primitive descriptors weren't selected.");

    bool in_place = inPlace();

    if (in_place) {
        if (inPlace(LOOK_DOWN) && inPlace(LOOK_UP)) {
            return true;
        }
    }

    int inNumber = getInputNum();
    const auto portChildEdges = parentNode->getChildEdgesAtPort(inNumber);

    if (portChildEdges.size() > 1) {
        if (in_place) {
            for (const auto& p_edge_peer : portChildEdges) {
                if (p_edge_peer.get() == this) {
                    continue;
                }
                if (p_edge_peer->inPlace(LOOK_DOWN)) {
                    return true;
                }
            }
        }
    }

    return false;
}

static inline bool isPhycicalMemCompatible(const MemoryDesc& lhsMemDesc, const MemoryDesc& rhsMemDesc) {
    if (!lhsMemDesc.isDefined() || !rhsMemDesc.isDefined() || !(lhsMemDesc.getType() & MemoryDescType::Blocked) ||
        !(rhsMemDesc.getType() & MemoryDescType::Blocked) ||
        (lhsMemDesc.getType() == DnnlBlocked && !lhsMemDesc.as<const DnnlMemoryDesc>()->hasEmptyExtraData()) ||
        (rhsMemDesc.getType() == DnnlBlocked && !rhsMemDesc.as<const DnnlMemoryDesc>()->hasEmptyExtraData())) {
        return false;
    }

    const auto* const lhsBlockMemDesc = lhsMemDesc.as<BlockedMemoryDesc>();
    const auto* const rhsBlockMemDesc = rhsMemDesc.as<BlockedMemoryDesc>();

    if (lhsBlockMemDesc->getShape() != rhsBlockMemDesc->getShape() ||
        lhsBlockMemDesc->getPrecision() != rhsBlockMemDesc->getPrecision()) {
        return false;
    }

    // dims padding check
    bool isZeroDimsPaddings = std::all_of(lhsBlockMemDesc->getOffsetPaddingToData().begin(),
                                          lhsBlockMemDesc->getOffsetPaddingToData().end(),
                                          [](size_t x) {
                                              return x == 0;
                                          }) &&
                              std::all_of(rhsBlockMemDesc->getOffsetPaddingToData().begin(),
                                          rhsBlockMemDesc->getOffsetPaddingToData().end(),
                                          [](size_t x) {
                                              return x == 0;
                                          });
    bool isSameElementsCount = lhsBlockMemDesc->getPaddedElementsCount() == rhsBlockMemDesc->getPaddedElementsCount();
    if (!isZeroDimsPaddings || !isSameElementsCount) {
        return false;
    }

    // tensor padding check
    if (lhsBlockMemDesc->getOffsetPadding() != rhsBlockMemDesc->getOffsetPadding()) {
        return false;
    }

    // stride check
    const auto lhsBlockDims = lhsBlockMemDesc->getBlockDims();
    std::vector<size_t> lhsStridesDefault(lhsBlockDims.size());
    lhsStridesDefault[lhsBlockDims.size() - 1] = 1;
    for (size_t i = 2; i <= lhsBlockDims.size(); i++) {
        lhsStridesDefault[lhsBlockDims.size() - i] =
            lhsStridesDefault[lhsBlockDims.size() - (i - 1)] * lhsBlockDims[lhsBlockDims.size() - (i - 1)];
    }

    auto rhsBlockDims = rhsBlockMemDesc->getBlockDims();
    std::vector<size_t> rhsStridesDefault(rhsBlockDims.size());
    rhsStridesDefault[rhsBlockDims.size() - 1] = 1;
    for (size_t i = 2; i <= rhsBlockDims.size(); i++) {
        rhsStridesDefault[rhsBlockDims.size() - i] =
            rhsStridesDefault[rhsBlockDims.size() - (i - 1)] * rhsBlockDims[rhsBlockDims.size() - (i - 1)];
    }

    // this check needed to avoid inserting unnecessary reorders if the memory is used in place and the batch size is
    // equal to 1 in nodes like concate and split
    size_t lhsSkipAxis = !lhsBlockDims.empty() && lhsBlockDims[0] == 1 ? 0 : Shape::UNDEFINED_DIM;
    size_t rhsSkipAxis = !rhsBlockDims.empty() && rhsBlockDims[0] == 1 ? 0 : Shape::UNDEFINED_DIM;

    bool isDenseTensor = dimsEqualStrong(lhsStridesDefault, lhsBlockMemDesc->getStrides(), lhsSkipAxis) &&
                         dimsEqualStrong(rhsStridesDefault, rhsBlockMemDesc->getStrides(), rhsSkipAxis);
    if (!isDenseTensor) {
        return false;
    }

    auto getCleanDim = [&](const VectorDims& dims, const VectorDims& flag) {
        if (dims.size() != flag.size()) {
            return dims;
        }
        std::vector<size_t> ret;
        for (size_t i = 0; i < dims.size(); i++) {
            if (flag[i] != 1) {
                ret.push_back(dims[i]);
            }
        }
        return ret;
    };

    // block dim check
    auto lhsBlockDimsClean = getCleanDim(lhsBlockDims, lhsBlockDims);
    auto rhsBlockDimsClean = getCleanDim(rhsBlockDims, rhsBlockDims);
    if (!dimsEqualStrong(lhsBlockDimsClean, rhsBlockDimsClean)) {
        return false;
    }

    // order check
    auto lhsOrderClean = getCleanDim(lhsBlockMemDesc->getOrder(), lhsBlockDims);
    auto rhsOrderClean = getCleanDim(rhsBlockMemDesc->getOrder(), rhsBlockDims);
    return dimsEqualStrong(lhsOrderClean, rhsOrderClean);
}

Edge::ReorderStatus Edge::needReorder() {
    bool optimized = false;
    auto inputPortDesc = getInputPortDesc();
    auto outPortDesc = getOutputPortDesc();
    if (inputPortDesc->getMemDesc()->getPrecision() == element::dynamic) {
        return ReorderStatus::No;
    }

    // Check whether the child node may accept the parent produced tensor
    if (!outPortDesc->isCompatible(*inputPortDesc)) {
        // Performance optimization which exploit the fact that some tensors do not need actual data reordering to be
        // read using different descriptors
        if (isPhycicalMemCompatible(*inputPortDesc->getMemDesc(), *outPortDesc->getMemDesc()) &&
            !getParent()->isConstant()) {
            optimized = true;
        } else {
            return ReorderStatus::Regular;
        }
    }

    // put here as more costly than compatible check
    if (enforceReorder()) {
        return ReorderStatus::Regular;
    }

    if (optimized) {
        return ReorderStatus::Optimized;
    }

    return ReorderStatus::No;
}

void Edge::reuse(MemoryPtr ptr) {
    OPENVINO_ASSERT(ptr, "Attempt to reuse uninitialized memory in ", *this);
    memoryPtr = std::move(ptr);
    changeStatus(Status::Allocated);

    DEBUG_LOG(*this, " memoryPtr=", memoryPtr);
}

int Edge::getInputNum() const {
    return parent_port;
}

int Edge::getOutputNum() const {
    return child_port;
}

void Edge::allocateCommon(const std::function<MemoryPtr(const MemoryDesc&)>& allocate) {
    OPENVINO_ASSERT(!memoryPtr, "Unexpected behaviour: status == NeedAllocation but memory is already allocated.");

    const auto& inputDesc = getInputDesc();
    const auto& outputDesc = getOutputDesc();
    OPENVINO_ASSERT(inputDesc.isCompatible(outputDesc), "Cannot allocate memory for incompatible descriptors.");

    memoryPtr = allocate(inputDesc);
    DEBUG_LOG(*this, " memoryPtr=", memoryPtr);
    status = Status::Allocated;
}

void Edge::allocate(const void* mem_ptr) {
    auto allocateFunc = [OV_CAPTURE_CPY_AND_THIS](const MemoryDesc& inputDesc) -> MemoryPtr {
        auto parentPtr = getParent();
        return std::make_shared<Memory>(parentPtr->getEngine(), inputDesc, mem_ptr, false);  // no pads zeroing
    };

    allocateCommon(allocateFunc);
}

void Edge::allocate(MemoryBlockPtr memBlock) {
    OPENVINO_ASSERT(memBlock, "Unexpected: Memory block ptr is NULL");
    auto allocateFunc = [this, block = std::move(memBlock)](const MemoryDesc& inputDesc) mutable -> MemoryPtr {
        auto parentPtr = getParent();
        return std::make_shared<Memory>(parentPtr->getEngine(), inputDesc, std::move(block));
    };

    allocateCommon(allocateFunc);
}

std::string Edge::hash() const {
    auto parentPtr = getParent();
    auto childPtr = getChild();

    return parentPtr->getName() + "_" + std::to_string(parent_port) + "_" + childPtr->getName() + "_" +
           std::to_string(child_port);
}

void Edge::externalAllocate(const WeightsSharing::Ptr& weightsCache) {
    if (status != Status::NeedAllocation) {
        return;
    }

    if (weightsCache) {
        auto alloc = [this]() {
            auto allocateFunc = [this](const MemoryDesc& inputDesc) -> MemoryPtr {
                auto parentPtr = getParent();
                return std::make_shared<StaticMemory>(parentPtr->getEngine(),
                                                      inputDesc,
                                                      nullptr,
                                                      false);  // no pads zeroing
            };

            allocateCommon(allocateFunc);
            return memoryPtr;
        };

        auto ptr = weightsCache->findOrCreate(hash(), alloc, false);
        memoryPtr = *ptr;
        DEBUG_LOG(*this, " memoryPtr=", memoryPtr);
        useExternalMemory = true;
        status = Status::Allocated;
    } else {
        allocate();
    }
}

void Edge::changeStatus(Edge::Status state) {
    OPENVINO_ASSERT(state != Status::NotAllocated, "Incorrect behaviour! Use method sharedMemFrom()");
    OPENVINO_ASSERT(state != Status::Validated, "Incorrect behaviour! Use method validate()");
    OPENVINO_ASSERT(Status::Validated != this->status, "Unexpected attempt of memory change on edge: ", *this);
    if (this->status != Status::Uninitialized && state == Status::NeedAllocation) {
        return;
    }
    if (this->status == Status::NotAllocated) {
        memoryFromEdge.reset();
    }
    this->status = state;
}

PortDescBaseCPtr Edge::getInputPortDesc() const {
    auto parentPtr = getParent();
    OPENVINO_ASSERT(parentPtr->getSelectedPrimitiveDescriptor(),
                    "Primitive descriptor for node ",
                    parentPtr->getName(),
                    " is not selected.");

    int inputIdx = getInputNum();
    OPENVINO_ASSERT(inputIdx >= 0, "Edge cannot be found for node", parentPtr->getName(), ".");
    const auto& outConfs = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs;
    OPENVINO_ASSERT(!outConfs.empty(), "Node ", parentPtr->getName(), " has empty output config list.");

    if (static_cast<size_t>(inputIdx) >= outConfs.size()) {
        inputIdx = 0;
    }

    auto inputPortDesc = outConfs[inputIdx].getPortDesc();
    OPENVINO_ASSERT(inputPortDesc, "Node", parentPtr->getName(), " has unitialized input port desc on port ", inputIdx);
    return inputPortDesc;
}

PortDescBaseCPtr Edge::getOutputPortDesc() const {
    auto childPtr = getChild();

    OPENVINO_ASSERT(childPtr->getSelectedPrimitiveDescriptor(),
                    "Primitive descriptor for node ",
                    childPtr->getName(),
                    " is not selected.");

    int outputIdx = getOutputNum();
    OPENVINO_ASSERT(outputIdx >= 0, "Edge cannot be found for node", childPtr->getName(), ".");
    const auto& inConfs = childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs;
    OPENVINO_ASSERT(!inConfs.empty(), "Node ", childPtr->getName(), " has empty input config list.");

    if (static_cast<size_t>(outputIdx) >= inConfs.size()) {
        outputIdx = 0;
    }

    auto outPortDesc = inConfs[outputIdx].getPortDesc();
    OPENVINO_ASSERT(outPortDesc, "Node", childPtr->getName(), " has unitialized output port desc on port ", outputIdx);
    return outPortDesc;
}

const MemoryDesc& Edge::getInputDesc() const {
    auto memDescPtr = getInputPortDesc()->getMemDesc();
    OPENVINO_ASSERT(memDescPtr,
                    "Cannot get input memory descriptor for edge: ",
                    getParent()->getName(),
                    "->",
                    getChild()->getName());
    return *memDescPtr;
}

const MemoryDesc& Edge::getOutputDesc() const {
    auto memDescPtr = getOutputPortDesc()->getMemDesc();
    OPENVINO_ASSERT(memDescPtr,
                    "Cannot get output memory descriptor for edge: ",
                    getParent()->getName(),
                    "->",
                    getChild()->getName());
    return *memDescPtr;
}

const MemoryDesc& Edge::getOriginalDesc() const {
    OPENVINO_ASSERT(none_of(status, Status::Validated, Status::Allocated),
                    "Desc of an Allocated edge ",
                    *this,
                    " must be accessed through the memory object");

    if (getInputDesc().getPrecision() == element::dynamic) {
        return getInputDesc();
    }

    OPENVINO_ASSERT(getInputDesc().isCompatible(getOutputDesc()), "Cannot get descriptor for edge: ", *this);

    return getInputDesc();
}

const IMemory& Edge::getMemory() {
    auto memPtr = getMemoryPtr();
    OPENVINO_ASSERT(memPtr, " Dereferencing NULL memory in edge: ", *this);
    return *memPtr;
}

MemoryPtr Edge::getMemoryPtr() const {
    return memoryPtr;
}

void Edge::sharedMemFrom(const EdgePtr& edge) {
    memoryFromEdge = edge;
    DEBUG_LOG(*this, " sharedMemFrom ", *edge);
    status = Status::NotAllocated;
}

void Edge::validate() {
    if (status == Status::Validated) {
        return;
    }

    std::ignore = getParent();
    std::ignore = getChild();

    OPENVINO_ASSERT(status == Status::Allocated && memoryPtr, "Error memory is not allocated for edge: ", *this);
    status = Status::Validated;
}

EdgePtr Edge::getSharedEdge() const {
    auto memoryFromEdgePtr = memoryFromEdge.lock();
    OPENVINO_ASSERT(memoryFromEdgePtr,
                    "Cannot get memory ptr for edge( ",
                    *this,
                    " ). The pointer on the edge with memory is empty!");
    return memoryFromEdgePtr;
}

EdgePtr Edge::getSharedEdge([[maybe_unused]] std::nothrow_t nothrow_tag) const {
    return memoryFromEdge.lock();
}

void Edge::init() {
    if (none_of(status, Status::NeedAllocation, Status::Uninitialized)) {
        return;
    }
    DEBUG_LOG(*this);
    EdgePtr edgePtr = getBaseEdge();
    if (edgePtr.get() == this) {
        DEBUG_LOG(*this, " getBaseEdge() return itself");
        changeStatus(Status::NeedAllocation);
    } else {
        if (Type::Input == edgePtr->getParent()->getType() && Type::MemoryInput != getParent()->getType() &&
            edgePtr->getParent()->isConstant() && !edgePtr->getChild()->isConstant()) {
            changeStatus(Status::NeedAllocation);
            DEBUG_LOG(*this, " edge inplace from ", *edgePtr, " is broken!");
            return;
        }
        sharedMemFrom(edgePtr);
    }
}

/**
 * Should analyze graph node dependencies, inplace node information and return root memory(edge) it view on
 *
 * @param type some magic enum values... description needed
 * @return root of view-on-memory subgraph
 */
EdgePtr Edge::getBaseEdge(int look) {
    const int inputNum = getInputNum();
    const int outputNum = getOutputNum();

    const int parentInPlacePort = getParent()->inPlaceOutPort(inputNum);
    const int childInPlacePort = getChild()->inPlaceInputPort(outputNum);

    const bool parent_valid = parentInPlacePort >= 0;
    const bool child_valid = childInPlacePort >= 0;
    OPENVINO_ASSERT(!parent_valid || !child_valid, "Unresolved in place memory conflict detected on edge: ", *this);

    if ((childInPlacePort >= 0) && (look & LOOK_DOWN)) {
        auto ch_edges = getChild()->getChildEdgesAtPort(childInPlacePort);
        auto& next_ch_edge = ch_edges[0];

        // Multiple connection to some out port
        // Will try to find inplace consumer
        for (auto& ch_edge : ch_edges) {
            if (ch_edge->getChild()->inPlaceInputPort(ch_edge->getOutputNum()) >= 0) {
                next_ch_edge = ch_edge;
                // To align with upstream-inplace, we stop searching once found the first inplace consumer
                break;
            }
        }
        return next_ch_edge;
    }
    if (parentInPlacePort >= 0 && (look & LOOK_UP)) {
        return getParent()->getParentEdgeAt(parentInPlacePort);
    }

    auto edgesForSamePort = getParent()->getChildEdgesAtPort(inputNum);
    for (auto edge : edgesForSamePort) {
        if (edge.get() != this) {
            // Return once found the first inplace consumer
            if (edge->inPlace()) {
                return edge;
            }
        }
    }

    // Return the first output edge as the base if there is no inPlace consumers
    // thus benefits zero-copy of outputs.
    for (auto edge : edgesForSamePort) {
        if (Type::Output == edge->getChild()->getType()) {
            return edge;
        }
    }

    return edgesForSamePort[0];
}

bool Edge::inPlace(LOOK look) const {
    int inputNum = getInputNum();
    if (look & LOOK_UP) {
        if (getParent()->inPlaceOutPort(inputNum) >= 0) {
            return true;
        }
    }

    int outputNum = getOutputNum();
    if (look & LOOK_DOWN) {
        if (getChild()->inPlaceInputPort(outputNum) >= 0) {
            return true;
        }
    }
    return false;
}

NodePtr Edge::modifiedInPlace() const {
    auto childNode = getChild();
    if (!childNode || !childNode->isInPlace() || childNode->getChildEdges().empty()) {
        return nullptr;
    }
    // check if the children nodes are able to modify the memory
    auto childPort = getOutputNum();
    auto inPlaceInputPort = childNode->inPlaceInputPort(childPort);
    if (inPlaceInputPort >= 0) {
        if (childNode->isExecutable()) {
            // Node can modify the memory
            return childNode;
        }
        for (auto&& edge : childNode->getChildEdgesAtPort(inPlaceInputPort)) {
            // continue searching
            if (auto result = edge->modifiedInPlace()) {
                return result;
            }
        }
    }
    // check backward dependency
    if (auto* childSPD = childNode->getSelectedPrimitiveDescriptor()) {
        const auto& outConfs = childSPD->getConfig().outConfs;
        for (size_t i = 0; i < outConfs.size(); ++i) {
            const auto& conf = outConfs[i];
            if (childPort < 0 || conf.inPlace() != childPort ||
                Type::MemoryInput == childNode->getType()) {  // exception type, it doesn't modify memory
                continue;
            }
            if (childNode->isExecutable()) {
                // Node can modify the memory
                return childNode;
            }
            for (auto&& edge : childNode->getChildEdgesAtPort(i)) {
                // continue searching
                if (auto result = edge->modifiedInPlace()) {
                    return result;
                }
            }
        }
    }

    // nothing has been found
    return nullptr;
}

std::ostream& operator<<(std::ostream& os, const Edge& edge) {
    return os << "(" << edge.getParent()->getName() << ")" << "[" << edge.getInputNum() << "] " << "<->" << "("
              << edge.getChild()->getName() << ")" << "[" << edge.getOutputNum() << "]" << ":"
              << Edge::statusToString(edge.getStatus());
}

}  // namespace ov::intel_cpu
