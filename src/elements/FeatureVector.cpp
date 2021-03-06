/**************************************************************************
 * Copyright (c) 2019-2020 Chimney Xu. All Rights Reserve.
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
/**
 * File: FeatureVector.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: feature vector
 * License: see the LICENSE.txt file
 *
 */

#include <TDBoW/elements/FeatureVector.h>

namespace TDBoW {

FeatureVector::FeatureVector(const FeatureVector& _Obj) : m_bLocked(false) {
    if(_Obj.m_bLocked) {
        while(true) {
            bool excepted = false;
            if(_Obj.m_bLocked.compare_exchange_strong(excepted, true)) {
                break;
            }
        }
    }
    for(const auto& pair : _Obj) {
        insert(end(), pair);
    }
    _Obj.m_bLocked = false;
}

// ---------------------------------------------------------------------------

void FeatureVector::addFeature(NodeId _ID, size_t _FeatureIdx) {
    SpinLock locker(m_bLocked);
    auto iter = lower_bound(_ID);
    if(iter != end() && iter -> first == _ID) {
        iter -> second.emplace_back(_FeatureIdx);
    } else {
        iter = insert(iter, value_type(_ID, std::vector<size_t>()));
        iter -> second.emplace_back(_FeatureIdx);
    }
}

// ---------------------------------------------------------------------------

FeatureVector& FeatureVector::operator+=(const FeatureVector& _Another) {
    if(_Another.m_bLocked) {
        while(true) {
            bool excepted = false;
            if(_Another.m_bLocked.compare_exchange_strong(excepted, true)) {
                break;
            }
        }
    }
    SpinLock locker(m_bLocked);
    auto iter1 = this -> begin();
    auto iter2 = _Another.begin();
    while(iter1 != this -> end() && iter2 != _Another.end()) {
        if(iter1 -> first == iter2 -> first) {
            auto& list1 = iter1 -> second;
            const auto& list2 = iter2 -> second;
            list1.insert(list1.end(), list2.begin(), list2.end());
            iter1++, iter2++;
        } else if(iter1 -> first < iter2 -> first) {
            iter1 = this -> lower_bound(iter2 -> first);
        } else {
            this -> insert(this -> end(), *iter2++);
        }
    }
    while(iter2 != _Another.end()) {
        this -> insert(this -> end(), *iter2++);
    }
    _Another.m_bLocked = false;
    return *this;
}

// ---------------------------------------------------------------------------

std::ostream& operator <<(std::ostream& _Out,
        const FeatureVector::value_type& pair) {
    _Out << '<' << pair.first << ": [";
    const auto& data = pair.second;
    auto iter = data.begin();
    if(!data.empty()) {
        _Out << *iter++;
    }
    while(iter != data.end()) {
        _Out << ", " << *iter++;
    }
    return _Out << "]>";
}

// ---------------------------------------------------------------------------

std::ostream& operator <<(std::ostream& _Out, const FeatureVector& _Vec) {
    SpinLock locker(_Vec.m_bLocked);
    if(_Vec.empty())return _Out << "[empty]";
    auto iter = _Vec.begin();
    _Out << '[' << *iter++;
    while(iter != _Vec.end()) {
        _Out << ", " << *iter++;
    }
    return _Out << "]";
}

// ---------------------------------------------------------------------------

} // namespace TDBoW
