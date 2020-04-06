/**************************************************************************
 * Copyright (c) 2020 Chimney Xu. All Rights Reserve.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **************************************************************************/
/* *************************************************************************
   * File Name     : TemplatedNode.hpp
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2020-04-05 20:18:18
   * Last Modified : smallchimney
   * Modified Time : 2020-04-05 20:36:47
************************************************************************* */

#ifndef __ROCKAUTO_TDBOW_TEMPLATED_NODE_HPP__
#define __ROCKAUTO_TDBOW_TEMPLATED_NODE_HPP__

#include "BowVector.h"
#include "TemplatedDescriptor.hpp"

namespace TDBoW {

/** @brief Vocabulary tree node */
template <typename TScalar, size_t DescL>
struct sNode {
    typedef TemplatedDescriptorUtil<TScalar, DescL> util;

    /** @brief Node id */
    NodeId id;
    /** @brief Weight if the node is a word */
    WordValue weight, weightBackup;
    /** Children */
    std::vector<NodeId> children;
    /** Parent node (undefined in case of root) */
    NodeId parent;
    /** Node descriptor */
    typename util::Descriptor descriptor;

    /** Word id if the node is a word */
    WordId word_id;

    typedef std::shared_ptr<sNode> Ptr;
    typedef std::shared_ptr<sNode const> ConstPtr;

    /**
     * Empty constructor
     */
    sNode(): id(0), weight(0), weightBackup(0),
            parent(0), descriptor{}, word_id(0) {}

    /**
     * Constructor
     * @param _id node id
     */
    sNode(NodeId _id): id(_id), weight(0),
            weightBackup(0), parent(0), descriptor{}, word_id(0) {}

    /**
     * Returns whether the node is a leaf node
     * @return true iff the node is a leaf
     */
    inline bool isLeaf() const { return children.empty(); }
};

} // namespace TDBoW

#endif //__ROCKAUTO_TDBOW_TEMPLATED_NODE_HPP__
