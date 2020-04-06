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
   * File Name     : TemplatedTask.hpp
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2020-04-05 20:28:45
   * Last Modified : smallchimney
   * Modified Time : 2020-04-05 20:37:30
************************************************************************* */

#ifndef __ROCKAUTO_TDBOW_TEMPLATED_TASK_HPP__
#define __ROCKAUTO_TDBOW_TEMPLATED_TASK_HPP__

#include "BowVector.h"
#include "TemplatedDescriptor.hpp"

namespace TDBoW {

/** @brief Vocabulary build task */
template <typename TScalar, size_t DescL>
struct sTask {
    typedef TemplatedDescriptorUtil<TScalar, DescL> util;
    typedef typename util::DescriptorConstPtr DescriptorConstPtr;

    typedef std::tuple<
            NodeId,                           // Task's parent node ID
            std::vector<DescriptorConstPtr>,  // Descriptors content address
            unsigned,                         // BoW tree nodes level
            std::vector<size_t>               // Descriptors' indices
    > Task;
};

} // namespace TDBoW

#endif //__ROCKAUTO_TDBOW_TEMPLATED_TASK_HPP__
