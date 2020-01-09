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
/* *************************************************************************
   * File Name     : TemplatedKMeans.hpp
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-11-27 14:01:21
   * Last Modified : smallchimney
   * Modified Time : 2020-01-12 21:06:26
************************************************************************* */
#ifndef __ROCKAUTO_TEMPLATED_K_MEANS_HPP__
#define __ROCKAUTO_TEMPLATED_K_MEANS_HPP__

#include "utils/BarProgress.h"
#include "TemplatedDescriptor.hpp"

#include <random>
#include <functional>
#include <flann/flann.hpp>
#ifdef FOUND_OPENMP
#include <omp.h>
#endif

namespace TDBoW {

// local methods

/**
 * @brief  Random float point value from [_Left, _Right)
 * @author smallchimney
 * @tparam Scalar  The scalar should be float point type.
 * @param  _Left   The min value of the range.
 * @param  _Right  The max value of the range.
 * @return         A random float point value
 */
template <typename Scalar>
static Scalar randomReal(const Scalar& _Left, const Scalar& _Right);

/**
 * @brief  Random integral value from [_Left, _Right]
 * @author smallchimney
 * @tparam Scalar  The scalar should be integral type.
 * @param  _Left   The min value of the range.
 * @param  _Right  The max value of the range.
 * @return         A random float point value
 */
template <typename Scalar>
static Scalar randomInt(const Scalar& _Left, const Scalar& _Right);

/**
 * @brief  Actually this is a robust k-means++ implement, can easily reused
 *         by define the template parameter as data type and distance type.
 *         e.g. {@code template <typename DataType, typename distance_type>}
 *         And using the function as parameter:
 *         {@code std::function<distance_type(const DataType&, const DataType&)>}
 * @author smallchimney
 * @tparam DescriptorUtil  The template parameters set of data type, distance
 *                         type, `meanValue()` and `distance()`
 */
template <class DescriptorUtil>
class TemplatedKMeans {
protected:
    // Load the typename from DescriptorUtil
    typedef typename DescriptorUtil::MeanCallback     MeanCallback;
    typedef typename DescriptorUtil::DistanceCallback DistanceCallback;
    TDBOW_DESCRIPTOR_DEF(DescriptorUtil)
    typedef typename Descriptor::Scalar Scalar;

    // typedef for Flann
    typedef ::flann::L2<Scalar> FLANNDist;
    template <typename Scalar>
    using FLANNMatrix = ::flann::Matrix<Scalar>;
    typedef ::flann::Index<FLANNDist> FLANNIndex;

public:
    typedef std::function<void(const size_t&, const std::vector<DescriptorConstPtr>&,
            DescriptorArray&, DistanceCallback, MeanCallback)> InitMethods;

    TemplatedKMeans() = delete;
    explicit TemplatedKMeans(const size_t& _K): m_ulK(_K) {
        assert(m_ulK != 0);
    }
    ~TemplatedKMeans() = default;

    /**
     * @brief  Run k-means++ on the descriptors
     * @author smallchimney
     * @param  _Descriptors     All descriptors to be grouped.
     * @param  _Centers  (out)  Centers value of each result cluster.
     * @param  _Clusters (out)  Descriptors of each cluster.
     * @param  _Init            Initial seed selected methods, default
     *                          by K-Means++.
     * @param  _DistF           Distance function.
     * @param  _MeanF           Mean value function.
     */
    void process(const std::vector<DescriptorConstPtr>& _Descriptors,
                 DescriptorArray& _Centers,
                 std::vector<std::vector<DescriptorConstPtr>>& _Clusters,
                 InitMethods _Init = initiateClustersKM2nd,
                 DistanceCallback _DistF = &DescriptorUtil::distance,
                 MeanCallback _MeanF = &DescriptorUtil::meanValue) noexcept(false);

    /**
     * @breif  Found k clusters' center from the given descriptor sets
     *         by running the initial step of k-means
     * @author smallchimney
     * @param  _Descriptors   Input descriptors.
     * @param  _Centers (out) resulting clusters.
     * @param  _DistF         Distance function.
     */
    static void initiateClustersKM(const size_t& _K,
            const std::vector<DescriptorConstPtr>& _Descriptors,
            DescriptorArray& _Centers, DistanceCallback _F, MeanCallback _M);

    /**
     * @breif  Found k clusters' center from the given descriptor sets
     *         by running the initial step of k-means++
     * @author smallchimney
     * @param  _Descriptors   Input descriptors.
     * @param  _Centers (out) resulting clusters.
     * @param  _DistF         Distance function.
     */
    static void initiateClustersKMpp(const size_t& _K,
            const std::vector<DescriptorConstPtr>& _Descriptors,
            DescriptorArray& _Centers, DistanceCallback _F, MeanCallback _M);

    /**
     * @breif  Found k clusters' center from the given descriptor sets
     *         by running the initial step of k-meansⅡ
     * @author smallchimney
     * @param  _Descriptors   Input descriptors.
     * @param  _Centers (out) resulting clusters.
     * @param  _DistF         Distance function.
     */
    static void initiateClustersKM2nd(const size_t& _K,
            const std::vector<DescriptorConstPtr>& _Descriptors,
            DescriptorArray& _Centers, DistanceCallback _F, MeanCallback _M);


private:
    size_t m_ulK;
};

template <class DescriptorUtil>
void TemplatedKMeans<DescriptorUtil>::process(
        const std::vector<DescriptorConstPtr>& _Descriptors,
        DescriptorArray& _Centers, std::vector<std::vector<DescriptorConstPtr>>& _Clusters,
        InitMethods _Init, DistanceCallback _DistF, MeanCallback _MeanF) noexcept(false) {
    _Centers.clear(); _Centers.shrink_to_fit();
    _Centers.reserve(m_ulK);
    _Clusters.clear(); _Clusters.shrink_to_fit();
    _Clusters.reserve(m_ulK);
    // No need for run k-means
    if(_Descriptors.size() <= m_ulK) {
        // Trivial case: one cluster per feature
        _Clusters.assign(_Descriptors.size(), std::vector<DescriptorConstPtr>());
#ifdef FOUND_OPENMP
        _Centers.resize(_Descriptors.size());
        #pragma omp parallel for
        for(size_t i = 0; i < _Descriptors.size(); i++) {
            _Clusters[i].emplace_back(_Descriptors[i]);
            _Centers[i] = *_Descriptors[i];
        }
#else
        _Centers.reserve(m_ulK);
        for(size_t i = 0; i < _Descriptors.size(); i++) {
            _Clusters[i].emplace_back(_Descriptors[i]);
            _Centers.emplace_back(*_Descriptors[i]);
        }
#endif
        return;
    }

    // stride on _Center for flann matrix mapping
    size_t stride = m_ulK > 1 ?
            (char*)(_Centers[1].data()) - (char*)(_Centers[0].data()) : 0;

    // select clusters and groups with k-means
    bool firstTime = true;
    // to check if clusters move after iterations
    std::vector<size_t> currentBelong, previousBelong;
    size_t iterCount = 0;
    distance_type prevLoss = 0;
    while(true) {
        // 1. Calculate clusters
        if(firstTime) {
            // random sample
            _Init(m_ulK, _Descriptors, _Centers, _DistF, _MeanF);
            firstTime = false;
        } else {
            // re-run the k-means if any cluster is empty
            for(const auto& cluster : _Clusters) {
                if(cluster.empty()) {
                    firstTime = true;
                    iterCount = 0;
                    break;
                }
            }
            if(firstTime) {
                std::cerr << TDBOW_LOG("Bad cluster founded, "
                             "re-run the k-means iterations.");
                continue;
            }
            // calculate cluster centres
#ifdef FOUND_OPENMP
            #pragma omp parallel for
#endif
            for(size_t i = 0; i < _Centers.size(); i++) {
                _Centers[i] = _MeanF(_Clusters[i]);
            }
        }

        // 2. Associate features with clusters
        // calculate distances to cluster centers
        _Clusters.assign(_Centers.size(), std::vector<DescriptorConstPtr>());
        currentBelong.resize(_Descriptors.size());
        distance_type loss = 0;
        if(stride && !std::is_same<Scalar, typename DescriptorUtil::binary_type>()) {
            FLANNIndex kdTree(FLANNMatrix<Scalar>(_Centers.data() -> data(),
                    _Centers.size(), DescriptorUtil::DescL, stride),
                    ::flann::KDTreeIndexParams(1));
            kdTree.buildIndex();
            BarProgress progress(_Descriptors.size());   //todo: Add logger switch
#ifdef FOUND_OPENMP
            std::vector<unsigned> __single_thread_count(
                    static_cast<size_t>(omp_get_num_procs()));
            #pragma omp parallel for reduction(+:loss)
#endif
            for(size_t i = 0; i < _Descriptors.size(); i++) {
                auto query = *_Descriptors[i];
                size_t index;
                typename FLANNDist::ResultType distance;
                FLANNMatrix<decltype(distance)> dis(&distance, 1, 1);
                FLANNMatrix<size_t> idx(&index, 1, 1);
                kdTree.knnSearch(FLANNMatrix<Scalar>(query.data(),
                                1, DescriptorUtil::DescL),
                        idx, dis, 1, ::flann::SearchParams(-1));
                loss += distance;
                currentBelong[i] = index;
#ifdef FOUND_OPENMP
                if(++__single_thread_count[omp_get_thread_num()] == 100) {
                    __single_thread_count[omp_get_thread_num()] = 0;
                    #pragma omp critical
                    progress.update(100);   //todo: Add logger switch
                }
#else
                progress.update();  //todo: Add logger switch
#endif
            }
            progress.finished();
        } else {
            BarProgress progress(   //todo: Add logger switch
                    _Descriptors.size() * _Centers.size());
#ifdef FOUND_OPENMP
            #pragma omp parallel for reduction(+:loss)
#endif
            for(size_t i = 0; i < _Descriptors.size(); i++) {
                const auto& descriptor = *_Descriptors[i];
                distance_type min = _DistF(descriptor, _Centers[0]);
                size_t minIdx = 0;
                for(size_t idx = 1; idx < _Centers.size(); idx++) {
                    distance_type distance =
                            DescriptorUtil::distance(descriptor, _Centers[idx]);
                    if(distance < min) {
                        min = distance;
                        minIdx = idx;
                    }
                    if(idx % 100 == 0) {//todo: Add logger switch
#ifdef FOUND_OPENMP
                        #pragma omp critical
#endif
                        progress.update(100);
                    }
                }
                {//todo: Add logger switch
                    auto left = _Centers.size() % 100;
#ifdef FOUND_OPENMP
                    #pragma omp critical
#endif
                    progress.update(left);
                }
                loss += min;
                currentBelong[i] = minIdx;
            }
        }
        for(size_t i = 0; i < _Descriptors.size(); i++) {
            const auto idx = currentBelong[i];
            _Clusters[idx].emplace_back(_Descriptors[i]);
        }
        if(iterCount++ && loss > prevLoss) {
            // Return the previous result
            _Clusters.assign(_Centers.size(), std::vector<DescriptorConstPtr>());
            for(size_t i = 0; i < _Descriptors.size(); i++) {
                const auto idx = previousBelong[i];
                _Clusters[idx].emplace_back(_Descriptors[i]);
            }
#ifdef FOUND_OPENMP
            #pragma omp parallel for
#endif
            for(size_t i = 0; i < _Centers.size(); i++) {
                _Centers[i] = _MeanF(_Clusters[i]);
            }
            return;
        }
        std::cout << "<Iter " << iterCount << ": loss "
                  << loss / _Descriptors.size() << '>' << std::endl;
        // k-means++ ensures all the clusters has any feature associated with them

        // 3. check convergence
        if(currentBelong == previousBelong) {
            return;
        }
        previousBelong = currentBelong;
        prevLoss = loss;
    }
}

template <class DescriptorUtil>
void TemplatedKMeans<DescriptorUtil>::initiateClustersKM(const size_t& _K,
        const std::vector<DescriptorConstPtr>& _Descriptors,
        DescriptorArray& _Centers, DistanceCallback _F, MeanCallback _M) {
    // Random K seeds
    _Centers.clear();
    _Centers.shrink_to_fit();

    std::vector<size_t> choices(_Descriptors.size());
#ifdef FOUND_OPENMP
    #pragma omp parallel for
#endif
    for(size_t i = 1; i < _Descriptors.size(); i++) {
        choices[i] = i;
    }
    _Centers.reserve(_K);
    for(size_t i = 0; i < _K; i++) {
        auto idx = randomInt<size_t>(0, choices.size() - 1);
        _Centers.emplace_back(*_Descriptors[choices[idx]]);
        choices[idx] = choices.back();
        choices.pop_back();
    }
}

template <class DescriptorUtil>
void TemplatedKMeans<DescriptorUtil>::initiateClustersKMpp(const size_t& _K,
        const std::vector<DescriptorConstPtr>& _Descriptors,
        DescriptorArray& _Centers, DistanceCallback _F, MeanCallback _M) {
    // Implements k-means++ seeding algorithm
    // Algorithm:
    // 1. Choose one center uniformly at random from among the data points.
    // 2. For each data point x, compute D(x), the distance between x and the nearest
    //    center that has already been chosen.
    // 3. Add one new data point as a center. Each point x is chosen with probability
    //    proportional to D(x)^2.
    // 4. Repeat Steps 2 and 3 until k centers have been chosen.
    // 5. Now that the initial centers have been chosen, proceed using standard k-means
    //    clustering.
    _Centers.clear();
    _Centers.reserve(_K);

    // 1.
    // create first cluster
    auto featureIdx = randomInt<size_t>(0, _Descriptors.size() - 1);
    _Centers.emplace_back(*_Descriptors[featureIdx]);

    std::cout << "Start k-means++ initialization." << std::endl;
    BarProgress progress(_K * _Descriptors.size());

    // compute the initial distances
#ifdef FOUND_OPENMP
    std::vector<distance_type> minDist(_Descriptors.size());
    size_t __init_update_count = 0;
    #pragma omp parallel for reduction(+:__init_update_count)
    for(size_t i = 0; i < _Descriptors.size(); i++) {
        minDist[i] = _F(*_Descriptors[i], _Centers.back());
        if(++__init_update_count % 1000 == 0) {
            progress.update(1000);
        }
    }
    progress.update(_Descriptors.size() - __init_update_count);
#else
    std::vector<distance_type> minDist(0);
    minDist.reserve(_Descriptors.size());
    for(const auto& descriptor : _Descriptors) {
        minDist.emplace_back(_F(*descriptor, _Centers.back()));
        progress.update();
    }
#endif

    while(_Centers.size() < _K) {
        // 2.
        const auto& center = _Centers.back();
#ifdef FOUND_OPENMP
        std::vector<unsigned> __single_thread_count(
                static_cast<size_t>(omp_get_num_procs()));
        #pragma omp parallel for
#endif
        for(size_t i = 0; i < _Descriptors.size(); i++) {
            auto& distance = minDist[i];
            if(distance > 0) {
                distance = std::min(distance, _F(*_Descriptors[i], center));
            }
#ifdef FOUND_OPENMP
            if(++__single_thread_count[omp_get_thread_num()] == 1000) {
                __single_thread_count[omp_get_thread_num()] = 0;
                #pragma omp critical
                progress.update(1000);
            }
#else
            progress.update();
#endif
        }
#ifdef FOUND_OPENMP
        auto __left_updated = std::accumulate(
                __single_thread_count.begin(), __single_thread_count.end(), 0u);
        progress.update(__left_updated);
#endif

        // 3.
#ifdef FOUND_OPENMP
        distance_type sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for(size_t i = 0; i < minDist.size(); i++) {
            sum += minDist[i];
        }
#else
        distance_type sum = std::accumulate(minDist.begin(), minDist.end(), 0.);
#endif
        if(sum == 0) {
            // Trivial case: one cluster per feature
            return;
        }
        if(sum < 0) {
            throw MethodNotMatchException(TDBOW_LOG(
                    "get negative sum, please check the `distance()`"));
        }

        auto limit = randomReal<distance_type>(0, sum);
        size_t idx = 0; sum = 0;
        while(idx < minDist.size() && sum <= limit) {
            sum += minDist[idx++];
        }
        _Centers.emplace_back(*_Descriptors[idx - 1]);
    }
    progress.finished();
    std::cerr << TDBOW_LOG("[DEBUG]: Finish k-means++ initialization");
}

template <class DescriptorUtil>
void TemplatedKMeans<DescriptorUtil>::initiateClustersKM2nd(const size_t& _K,
        const std::vector<DescriptorConstPtr>& _Descriptors,
        DescriptorArray& _Centers, DistanceCallback _F, MeanCallback _M) {
    // Implements k-meansⅡ seeding algorithm
    const auto LIMIT = static_cast<size_t>(log(_Descriptors.size()) / log(2)) * _K;

    if(_Descriptors.size() <= LIMIT) {
        initiateClustersKMpp(_K, _Descriptors, _Centers, _F, _M);
        return;
    }

    std::vector<size_t> choices(_Descriptors.size());
#ifdef FOUND_OPENMP
    #pragma omp parallel for
#endif
    for(size_t i = 1; i < _Descriptors.size(); i++) {
        choices[i] = i;
    }
    std::vector<DescriptorConstPtr> seeds;
    seeds.reserve(LIMIT);
    std::vector<size_t> indices;
    for(size_t i = 0; i < LIMIT; i++) {
        auto idx = randomInt<size_t>(0, choices.size() - 1);
        seeds.emplace_back(std::make_shared<Descriptor>(*_Descriptors[choices[idx]]));
        indices.emplace_back(choices[idx]);
        choices[idx] = choices.back();
        choices.pop_back();
    }
    std::cerr << TDBOW_LOG("[DEBUG]: Start k-means 2nd initialization at " << seeds.size());
    std::vector<std::vector<DescriptorConstPtr>> ignored;
    TemplatedKMeans<DescriptorUtil>(_K).process(seeds,
            _Centers, ignored, initiateClustersKMpp, _F, _M);
    std::cerr << TDBOW_LOG("[DEBUG]: Finish k-means 2nd initialization at " << seeds.size());
}

/* ********************************************************************************
 *                               RANDOM METHODS                                   *
 ******************************************************************************** */

template <typename Scalar>
Scalar randomReal(const Scalar& _Left, const Scalar& _Right) {
    static_assert(std::is_floating_point<Scalar>::value,
                  "result_type must be a floating point type");
    static std::default_random_engine e(std::random_device().operator()());
    typedef std::uniform_real_distribution<Scalar> Uniform;
    typedef typename Uniform::param_type param;
    static Uniform uniform;
    uniform.param(param(_Left, _Right));
    return uniform(e);
}

template <typename Scalar>
Scalar randomInt(const Scalar& _Left, const Scalar& _Right) {
    static_assert(std::is_integral<Scalar>::value,
                  "template argument must be an integral type");
    static std::default_random_engine e(std::random_device().operator()());
    typedef std::uniform_int_distribution<Scalar> Uniform;
    typedef typename Uniform::param_type param;
    static Uniform uniform;
    uniform.param(param(_Left, _Right));
    return uniform(e);
}

}   // namespace TDBoW

#endif //__ROCKAUTO_TEMPLATED_K_MEANS_HPP__
