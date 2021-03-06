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

//DBoW2: bag-of-words library for C++ with generic descriptors
//
//Copyright (c) 2015 Dorian Galvez-Lopez. http://doriangalvez.com
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions
//are met:
//1. Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//2. Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//3. The original author of the work must be notified of any
//   redistribution of source code or in binary form.
//4. Neither the name of copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS
//BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//POSSIBILITY OF SUCH DAMAGE.

/**
 * File: TemplatedVocabulary.hpp
 * Date: February 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated vocabulary 
 */

#ifndef __ROCKAUTO_TDBOW_TEMPLATED_VOCABULARY_HPP__
#define __ROCKAUTO_TDBOW_TEMPLATED_VOCABULARY_HPP__

#include <fstream>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <queue>

#include "FeatureVector.h"
#include "ScoringObject.h"
#include "TemplatedDescriptor.hpp"
#include "TemplatedNode.hpp"
#include "TemplatedTask.hpp"
#include <TDBoW/utils/TemplatedKMeans.hpp>
#include <TDBoW/utils/TemplatedCheckPoint.hpp>

#include <boost/dynamic_bitset.hpp>
#include <yaml-cpp/yaml.h>

namespace TDBoW {

// local methods

/**
 * @brief  Convert the data type and assign the data, please confirm the scalar
 *         can be convert successful
 * @author smallchimney
 * @tparam T    It's only recommend to operate with C++ standard data type
 *              unless you know what you are doing
 * @param node  Data node inside the yaml file
 * @param i     The output data to be assigned
 */
template<typename T>
static void operator << (T& i, const YAML::Node& node);

/// @param Descriptor class of descriptor
template <typename TScalar, size_t DescL>
/// Generic Vocabulary
class TemplatedVocabulary {
public:
    // typedef
    typedef TScalar ScalarType;
    typedef TemplatedDescriptorUtil<TScalar, DescL> util;
    typedef TemplatedKMeans<util> KMeansUtil;
    typedef TemplatedCheckPoint<TScalar, DescL> CheckPointUtil;
    typedef typename CheckPointUtil::Ptr CheckPointUtilPtr;
    /* Type define */
    typedef sNode<TScalar, DescL> Node;
    typedef std::vector<Node> Nodes;
    typedef std::unique_ptr<Nodes> NodesPtr;
    TDBOW_DESCRIPTOR_DEF(util)
    friend CheckPointUtil;

    // typedef for Flann
    typedef ::flann::L2<TScalar> FLANNDist;
    template <typename Scalar>
    using FLANNMatrix = ::flann::Matrix<Scalar>;
    typedef ::flann::Index<FLANNDist> FLANNIndex;
    typedef std::unique_ptr<FLANNIndex> FLANNIndexPtr;

    /** The mode of operation when loading a vocabulary */
    enum LoadMode {AUTO, BINARY, YAML};

    typedef typename sTask<TScalar, DescL>::Task Task;

    // Constructors

    /**
     * @breif Initiates an empty vocabulary
     * @param _K          Branching factor
     * @param _L          Depth levels
     * @param _Weighting  Weighting type
     * @param _Scoring    Scoring type
     */
    explicit TemplatedVocabulary(unsigned _K = 10, unsigned _L = 1,
            WeightingType _Weighting = TF_IDF, ScoringType _Scoring = L1_NORM);
  
    /**
     * @breif  Creates the vocabulary by loading a file
     * @author smallchimney
     * @param  _Filename
     * @param  _Mode      Which method to process the file
     */
    explicit TemplatedVocabulary(
            const std::string& _Filename, LoadMode _Mode = AUTO) noexcept(false);

    /**
     * @breif Memory copy constructor is delete
     * @param _Vocab  original vocabulary which will drop all the data
     */
    TemplatedVocabulary(TemplatedVocabulary<TScalar, DescL>& _Vocab) = delete;

    /**
     * @breif Memory move constructor
     * @param _Vocab  original vocabulary which will drop all the data
     */
    TemplatedVocabulary(TemplatedVocabulary<TScalar, DescL>&& _Vocab) noexcept;

    /**
     * Destructor
     * We use smart pointer, so no need for free memory manually.
     */
    virtual ~TemplatedVocabulary() {
        m_pScoringObj.reset();
        clear();
    }

    // Build methods

    /**
     * @breif Creates a vocabulary from the training features with the already
     * defined parameters
     * @param _TrainingData  The train dataset
     * @param _Log           The logger filename, keep empty in small scale.
     *                       In case logger was active, will automatically
     *                       save/load the checkpoint during the creating,
     *                       all the checkpoints will be removed after the
     *                       processing.
     * @param _WiseK         Whether dynamic adjust the K parameter
     * @param _ConsoleLog    Whether log to the console, only valid if
     *                       {@code _Log} is not empty.
     */
    void create(const ConstDataSet& _TrainingData,
                const std::string& _Log = "",
                bool _WiseK = true,
                bool _ConsoleLog = true) noexcept(false);

    /**
     * @breif Creates a vocabulary from the training features, setting the branching
     *        factor and the depth levels of the tree, and the weighting and scoring
     * @param _TrainingData
     * @param _K             Branching factor
     * @param _L             Depth levels
     * @param _Log           The logger filename, keep empty in small scale.
     *                       In case logger was active, will automatically
     *                       save/load the checkpoint during the creating,
     *                       all the checkpoints will be removed after the
     *                       processing.
     * @param _ConsoleLog    Whether log to the console, only valid if
     *                       {@code _Log} is not empty.
     * @param _Weighting     Weighting type
     * @param _Scoring       Scoring type
     */
    void create(
            const ConstDataSet& _TrainingData, unsigned _K, unsigned _L = 1,
            const std::string& _Log = "", bool _ConsoleLog = true,
            WeightingType _Weighting = TF_IDF, ScoringType _Scoring = L1_NORM) noexcept(false);

    // Function methods

    /**
     * @breif  Transforms a single feature into a word (without weight)
     * @param  _Feature
     * @return word id
     */
    virtual WordId transform(const Descriptor& _Feature) const noexcept(false);

    /**
     * @breif Transform a set of descriptors into a bow vector and a feature vector
     * @param _Features
     * @param _BowVec  (out)   Bow vector
     * @param _FeatVec (out)   Feature vector of nodes and feature indexes
     * @param _LevelsUp        Levels to go up the vocabulary tree to get the node index
     */
    void transform(const DescriptorArray& _Features, BowVector& _BowVec,
            const FeatureVectorPtr& _FeatVec = nullptr, unsigned _LevelsUp = 0) const noexcept(false);

    /**
     * @breif Transform a set of descriptors into a bow vector and a feature vector
     * @param _Features
     * @param _BowVec  (out)   Bow vector
     * @param _FeatVec (out)   Feature vector of nodes and feature indexes
     * @param _LevelsUp        Levels to go up the vocabulary tree to get the node index
     */
    void transform(const Descriptors& _Features, BowVector& _BowVec,
            const FeatureVectorPtr& _FeatVec = nullptr, unsigned _LevelsUp = 0) const noexcept(false);
  
    /**
     * @breif  Returns the score of two vectors
     * @param  _A vector
     * @param  _B vector
     * @return score between vectors
     * @note   the vectors must be already sorted and normalized if necessary
     */
    inline double score(const BowVector& _A, const BowVector& _B) const {
        return m_pScoringObj -> score(_A, _B);
    }

    /**
     * @breif  Destroy all nodes, and free the memory.
     * @author smallchimney
     * @param  _Alloc  Whether alloc a new space for nodes
     */
    void clear(bool _Alloc = false);

    /**
     * @breif  Stops those words whose weight is below minWeight.
     *         Words are stopped by setting their weight to 0. There are not returned
     *         later when transforming image features into vectors.
     *         Note that when using IDF or TF_IDF, the weight is the idf part, which
     *         is equivalent to -log(f), where f is the frequency of the word
     *         (f = Ni/N, Ni: number of training images where the word is present,
     *         N: number of training images).
     * @author smallchimney
     * @param  _Percent  the stop threshold in percent
     * @return           number of words stopped now
     */
    virtual size_t stopPercent(double _Percent);

    /**
     * @breif  Stops those words whose weight is below minWeight.
     *         Words are stopped by setting their weight to 0. There are not returned
     *         later when transforming image features into vectors.
     *         Note that when using IDF or TF_IDF, the weight is the idf part, which
     *         is equivalent to -log(f), where f is the frequency of the word
     *         (f = Ni/N, Ni: number of training images where the word is present,
     *         N: number of training images).
     * @param  _MinLimit minimum value
     * @return           number of words stopped now
     */
    virtual size_t stopWords(WordValue _MinLimit);

    // Get methods
    
    /**
     * @breif  Whether the vocabulary is ready.
     * @author smallchimney
     * @return {@code true} after the words are created and weighted
     */
    virtual bool ready() const noexcept { return m_bInit; }

    /**
     * @breif  Returns the number of words in the vocabulary
     * @return number of words
     */
    virtual size_t size() const noexcept { return m_aWords.size(); }

    /**
     * @breif  Returns whether the vocabulary is empty (i.e. it has not been trained)
     * @return We assume the words as the actual vocabulary, so will return {@code true}
     *         if the words is empty
     */
    virtual bool empty() const noexcept { return m_aWords.empty(); }
  
    /**
     * @breif  Returns the id of the node that is "levelsup" levels from the word given
     * @param  _Wid        word ID
     * @param  _LevelsUp   0..L
     * @return             Node ID. if {@code _LevelsUp} is 0, returns the node
     *                     ID associated to the word ID.
     * @throws             The required node out of range.
     */
    virtual NodeId getParentNode(WordId _Wid, int _LevelsUp) const noexcept(false);

    /**
     * @breif Returns the ids of all the words that are under the given node ID,
     *        by traversing any of the branches that goes down from the node
     * @param _Nid    Starting node ID
     * @param _Words  IDs of words
     * @throws        The required word out of range.
     */
    void getWordsFromNode(NodeId _Nid, std::vector<WordId>& _Words) const noexcept(false);
  
    /**
     * @breif Returns the branching factor of the tree (k)
     * @return k
     */
    inline unsigned getBranchingFactor() const { return m_uiK; }
  
    /**
     * @breif  Returns the depth levels of the tree (L)
     * @return L
     */
    inline unsigned getDepthLevels() const { return m_uiL; }
  
    /**
     * @breif  Returns the real depth levels of the tree on average
     * @return average of depth levels of leaves
     */
    float getEffectiveLevels() const;
  
    /**
     * @breif Returns the descriptor of a word
     * @param _Wid  Word id
     * @return      Descriptor
     * @throws      Out of range.
     */
    virtual Descriptor getWord(WordId _Wid) const noexcept(false);
  
    /**
     * @breif Returns the weight of a word
     * @param _Wid  Word id
     * @return      Weight
     * @throws      Out of range.
     */
    virtual WordValue getWordWeight(WordId _Wid) const noexcept(false);

    const GeneralScoring& getScoringObj() const noexcept(false) {
        return *m_pScoringObj;
    }
  
    /**
    * @breif  Returns the weighting method
    * @return weighting method
    */
    WeightingType getWeightingType() const { return m_eWeighting; }

    /**
    * @breif  Returns the scoring method
    * @return scoring method
    */
    ScoringType getScoringType() const { return m_eScoring; }

    // Set methods

    /**
     * @breif Changes the scoring method
     * @param _Type new scoring type
     */
    TemplatedVocabulary& setScoringType(ScoringType _Type) noexcept(false);

    /**
     * @breif Changes the weighting method
     * @param _Type new weighting type
     */
    inline TemplatedVocabulary& setWeightingType(WeightingType _Type) noexcept(false);

    // I/O methods

    /**
     * @breif  Saves the vocabulary to a file
     * @author smallchimney
     * @param  _Filename  The relative/absolute filename, will be
     *                    regarded as relative to the `vocab` directory.
     * @param  _Binary    Whether save in binary file format.
     */
    void save(const std::string& _Filename,
              bool _Binary = true) const noexcept(false);

    /**
     * @breif  Load the vocabulary from a file
     * @author smallchimney
     * @param  _Filename  The relative/absolute filename, will be
     *                    regarded as relative to the `vocab` directory.
     * @param  _Mode      The mode of operation when loading.
     *                    {@code AUTO} will decided by filename.
     *                    {@code BINARY} will load as binary file.
     *                    {@code YAML} will load as yaml file.
     */
    void load(const std::string& _Filename, LoadMode _Mode = AUTO) noexcept(false);

    /**
     * @breif  Saves the vocabulary to a DBoW2 format yaml file
     * @author smallchimney
     * @param  _File    The vocabulary file.
     * @param  _F       The `toString()` method for each descriptor,
     *                  keep {@code nullptr} for `Eigen::<<()`.
     */
    void saveYAML(
            const boost::filesystem::path& _File,
            const std::function<std::string(Descriptor)>& _F = nullptr) const noexcept(false);

    /**
     * @breif  Saves the vocabulary to a 3DBoW format binary file
     * @author smallchimney
     * @param  _Filename    The vocabulary file.
     * @param  _Compressed  Whether do compress operation during the output
     */
    void saveBinary(const boost::filesystem::path& _File,
                    bool _Compressed = true) const noexcept(false);

    /**
     * @breif  Loads the vocabulary from a file
     * @author smallchimney
     * @param  _File    yaml format vocabulary in DBoW2
     * @param  _F       The `toString()` method for each descriptor,
     *                  keep {@code nullptr} for `Eigen::<<()`.
     */
    void loadYAML(const boost::filesystem::path& _File,
                  const std::function<Descriptor(std::string)>& _F = nullptr);

    /**
     * @breif  Loads the vocabulary from a file
     * @author smallchimney
     * @param  _File   binary format vocabulary in 3DBoW
     */
    void loadBinary(const boost::filesystem::path& _File);

    /**
     * @breif  Output the vocabulary to a binary stream
     * @author smallchimney
     * @param  _Out         output stream
     * @param  _Compressed  Whether do compress operation during the output
     */
    void write(std::ostream& _Out, bool _Compressed = true) const noexcept(false);

    /**
     * @breif  Output the vocabulary to a binary stream
     * @author smallchimney
     * @param  _Out         output stream
     * @param  _Compressed  Whether do compress operation during the output
     */
    void read(std::istream& _In) noexcept(false);

    /**
     * @brief  The `=()` operator is not recommended because it usually
     *         with huge memory usage, `move()` method should be instead use.
     * @author smallchimney
     * @param  _Vocab
     * @return reference to this vocabulary
     */
    TemplatedVocabulary<TScalar, DescL>& operator =(
            const TemplatedVocabulary<TScalar, DescL>& _Vocab) = delete;

protected:

    /**
     * @breif Creates an instance of the scoring object according to
     *        the value of {@code m_eScoring}.
     */
    void _createScoringObject();

    /**
     * @breif Returns a set of pointers to descriptors
     * @param _TrainingData
     * @return  Flattened raw pointers of the features,
     *          using return value to avoid memory leak
     *          of original pointers.
     */
    void _getFeatures(const ConstDataSet& _TrainingData,
                     std::vector<DescriptorConstPtr>& _Features) const noexcept(false);

    /**
     * @breif Returns the word id associated to a feature (thread safe)
     * @param _Feature  single feature to be transformed
     * @param _ID (out) word id
     */
    virtual void _transform(const Descriptor& _Feature, WordId& _ID) const noexcept(false);

    /**
     * @breif Returns the word id associated to a feature (thread safe)
     * @param _Feature
     * @param _WordId (out)  Word id
     * @param _Weight (out)  Word weight
     * @param _LevelsUp      The number of layers of returned node ID
     *                       upper than word
     * @return               The leaf/parent ID of the word
     */
    NodeId _transform(const Descriptor& _Feature,
            WordId& _WordId, WordValue& _Weight, unsigned _LevelsUp = 0) const noexcept(false);

    /**
     * @breif Creates a level in the tree, under the parent, by running
     *        k-means with a descriptor set, and recursively creates the
     *        subsequent levels too in BFS order.
     * @param _Descriptors  Descriptors' address to run the k-means on
     */
    void _buildTree(const std::vector<DescriptorConstPtr>& _Descriptors);

    /**
     * @brief  build the search tree on words
     * @author smallchimney
     */
    void _buildSearchTree();

    /**
     * @breif Create the words of the vocabulary once the tree has been built
     */
    void _createWords();

    /**
     * @breif Sets the weights of the nodes of tree according to the given features.
     *        Before calling this function, the nodes and the words must be already
     *        created (by calling `_buildTree()` and `_createWords()`)
     * @param features
     */
    void _setNodeWeights(const ConstDataSet& _TrainingData);

protected:
    // Initialization indicator
    bool m_bInit = false;

    /// Branching factor
    unsigned int m_uiK{};

    /// Depth levels
    unsigned int m_uiL{};

    /// Weighting method
    WeightingType m_eWeighting{};

    /// Scoring method
    ScoringType m_eScoring{};

    /// Object for computing scores
    std::unique_ptr<GeneralScoring> m_pScoringObj;

    /// Tree nodes pointer
    /// Since the vocabulary usually not in a small scale,
    /// so make it unique to avoid redundant copy
    NodesPtr m_pNodes;

    /// Words of the vocabulary (tree leaves)
    /// this condition holds: (*m_pNodes)[m_aWords[wid]] -> word_id == wid
    std::vector<NodeId> m_aWords;

    /** @brief the leaves in search tree, stands for all the words' descriptors */
    std::vector<Descriptor> m_aSearchTreeLeaves;

    /**
     * @brief FLANN based kd-tree implement search tree
     * // TODO: maybe use paracel::balltree better in this case?
     * @author smallchimney
     */
    FLANNIndexPtr m_pWordsSearchTree;

    CheckPointUtilPtr m_pCheckPoint = nullptr;

};

/* ********************************************************************************
 *                        CONSTRUCTION && INITIALIZATION                          *
 ******************************************************************************** */

template <typename TScalar, size_t DescL>
TemplatedVocabulary<TScalar, DescL>::TemplatedVocabulary(
        const unsigned _K, const unsigned _L, const WeightingType _Weighting,
        const ScoringType _Scoring) noexcept(false)
        : m_uiK(_K), m_uiL(_L), m_pScoringObj(nullptr), m_pNodes(nullptr),
        m_pWordsSearchTree(nullptr) {
    if(pow(_K, _L) >= std::numeric_limits<NodeId>::max()) {
        throw ParametersException(TDBOW_LOG("Too large for the vocabulary scale."));
    }
    setWeightingType(_Weighting);
    setScoringType(_Scoring);
}

template <typename TScalar, size_t DescL>
TemplatedVocabulary<TScalar, DescL>::TemplatedVocabulary(
        const std::string& _Filename, const LoadMode _Mode) noexcept(false)
        : m_pScoringObj(nullptr), m_pNodes(nullptr), m_pWordsSearchTree(nullptr) {
    load(_Filename, _Mode);
}

template <typename TScalar, size_t DescL>
TemplatedVocabulary<TScalar, DescL>::TemplatedVocabulary(
        TemplatedVocabulary<TScalar, DescL>&& _Vocab) noexcept {
    clear();  // Safety free the memory
    m_uiL = _Vocab.m_uiL;
    m_uiK = _Vocab.m_uiK;
    m_pNodes = std::move(_Vocab.m_pNodes);
    m_aWords = std::move(_Vocab.m_aWords);
    m_aSearchTreeLeaves = std::move(_Vocab.m_aSearchTreeLeaves);
    m_pWordsSearchTree = std::move(_Vocab.m_pWordsSearchTree);
    setWeightingType(_Vocab.m_eWeighting);
    setScoringType(_Vocab.m_eScoring);
    m_bInit = _Vocab.m_bInit;
    _Vocab.clear();
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::_createScoringObject() {
    switch(m_eScoring) {
        case L1_NORM:
        default:
            m_pScoringObj.reset(new L1Scoring);
            break;

        case L2_NORM:
            m_pScoringObj.reset(new L2Scoring);
            break;

        case CHI_SQUARE:
            m_pScoringObj.reset(new ChiSquareScoring);
            break;

        case KL:
            m_pScoringObj.reset(new KLScoring);
            break;

        case BHATTACHARYYA:
            m_pScoringObj.reset(new BhattacharyyaScoring);
            break;

        case DOT_PRODUCT:
            m_pScoringObj.reset(new DotProductScoring);
            break;
    }
}

/* ********************************************************************************
 *                                 BUILD METHODS                                  *
 ******************************************************************************** */

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::create(const ConstDataSet& _TrainingData,
        const std::string& _Log, bool _WiseK, const bool _ConsoleLog) noexcept(false) {
    // wise K adjust require TDBoW's format vocabulary, set the level as just single layer
    if(_WiseK && m_uiL != 1)_WiseK = false;

    clear(true);
    std::vector<DescriptorConstPtr> features(0);
    _getFeatures(_TrainingData, features);
    if(features.size() < 100) {
        throw DataException(TDBOW_LOG("Too few data for a dataset to be trained."));
    }
    if(_WiseK) {
        auto logN = features.size() <= 1000 ? 1. : log10(features.size()) - 2;
        auto sumLogN = logN * (logN + 1);   // "/ 2 is transform to the below"
        m_uiK = static_cast<unsigned>(features.size() / (5 * sumLogN));
        m_pNodes -> reserve(m_uiK + 1);
    } else {
        // expectedSize = Sum_{i=0..L} ( k^i )
        auto expectedSize = static_cast<size_t>((pow(m_uiK, m_uiL + 1) - 1) / (m_uiK - 1));
        m_pNodes -> reserve(expectedSize); // avoid allocations when creating the tree
    }

    // For checkpoint, maybe we should check the digest of dataset, but the
    // dataset is unordered. As now we need users to grantee the dataset
    // is unchanged in case the last creating is failed as well the tag name
    // is unchanged, too.
    if(!_Log.empty()) {
        m_pCheckPoint.reset(new CheckPointUtil(
                _Log, features.size(), m_uiK, m_uiL, _ConsoleLog));
    } else {
        m_pCheckPoint.reset();
    }

    // Insert the root node
    m_pNodes -> emplace_back(0);
    // Create the tree
    // Checkpoint mini require: K L
    if(!m_pCheckPoint || !(m_pCheckPoint -> completed(
            CheckPointUtil::CLUSTER_END)))_buildTree(features);
    // Create the words
    // No checkpoint saving inside, only load
    _createWords();
    // build the search tree
    // No checkpoint, this operation using flann in this version
    _buildSearchTree();
    // Set the initialized label
    m_bInit = !empty();
    // Set the weight of each node of the tree
    // No checkpoint, this operation based on flann result
    _setNodeWeights(_TrainingData);
    // After this, the checkpoint can be removed safety.
    if(m_pCheckPoint) {
        // So will remove the checkpoint when program exit
        m_pCheckPoint -> setStatus(CheckPointUtil::FINISHED);
    }
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::create(
        const ConstDataSet& _TrainingData, const unsigned _K, const unsigned _L,
        const std::string& _Log, const bool _ConsoleLog,
        const WeightingType _Weighting, const ScoringType _Scoring) noexcept(false) {
    m_uiK = _K; m_uiL = _L;
    clear(true);
    setWeightingType(_Weighting);
    setScoringType(_Scoring);
    create(_TrainingData, _Log, _ConsoleLog, true);
}

/* ********************************************************************************
 *                              FUNCTIONAL METHODS                                *
 ******************************************************************************** */

template <typename TScalar, size_t DescL>
WordId TemplatedVocabulary<TScalar, DescL>::transform(
        const Descriptor& _Feature) const noexcept(false) {
    WordId wid = 0;
    if(!empty()) {
        _transform(_Feature, wid);
    }
    return wid;
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::transform(
        const DescriptorArray& _Features, BowVector& _BowVec,
        const FeatureVectorPtr& _FeatVec, const unsigned _LevelsUp) const noexcept(false) {
    if(!ready()) {
        throw NotInitailizedException(TDBOW_LOG(
                "The vocabulary is empty, must be created before transform."));
    }
    _BowVec.clear();
    if(_FeatVec) {
        _FeatVec -> clear();
    }
    if(empty()) { // Safe for subclasses
        return;
    }

    void(BowVector::*bowAdd)(WordId, WordValue) = nullptr;
    switch(m_eWeighting) {
        case WeightingType::TF: case WeightingType::TF_IDF:
            bowAdd = &BowVector::addWeight;
            break;

        case WeightingType::IDF: case WeightingType::BINARY:
            bowAdd = &BowVector::addIfNotExist;
            break;
    }
    assert(bowAdd != nullptr);
#ifdef FOUND_OPENMP
    std::vector<BowVector> bowVectors(
            static_cast<size_t>(omp_get_num_procs()));
    std::vector<FeatureVector> featVectors(
            static_cast<size_t>(omp_get_num_procs()));
    #pragma omp parallel for
    for(size_t i = 0; i < _Features.size(); i++) {
        const auto& feature = _Features[i];
        WordId id = 0;
        WordValue w = 0.;
        // TF_IDF/IDF -- idf value
        // TF/BINARY  -- 1
        NodeId nid = _transform(feature, id, w, _LevelsUp);    // thread safe method
        // If not stopped
        if(w > 0) {
            // NOTE: These two method must be safety in multi-threads
            (bowVectors[omp_get_thread_num()].*bowAdd)(id, w);
            if(_FeatVec) {
                featVectors[omp_get_thread_num()].addFeature(nid, i);
            }
        }
    }
    for(const auto& bowVector : bowVectors) {
        _BowVec += bowVector;
    }
    if(_FeatVec) {
        for(const auto& featVector : featVectors) {
            _FeatVec -> operator+=(featVector);
        }
    }
#else
    for(size_t i = 0; i < _Features.size(); i++) {
        const auto& feature = _Features[i];
        WordId id = 0;
        WordValue w = 0.;
        // TF_IDF/IDF -- idf value
        // TF/BINARY  -- 1
        NodeId nid = _transform(feature, id, w, _LevelsUp);    // thread safe method
        // If not stopped
        if(w > 0) {
            // NOTE: These two method must be safety in multi-threads
            (_BowVec.*bowAdd)(id, w);
            if(_FeatVec) {
                _FeatVec -> addFeature(nid, i);
            }
        }
    }
#endif
    // Normalize
    LNorm norm{};
    if(m_pScoringObj -> mustNormalize(norm)) {
        _BowVec.normalize(norm);
    } else if(!_BowVec.empty() && (m_eWeighting == TF || m_eWeighting == TF_IDF)) {
        const double nd = _BowVec.size();
        for(auto& v : _BowVec) v.second /= nd;
    }
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::transform(
        const Descriptors& _Features, BowVector& _BowVec,
        const FeatureVectorPtr& _FeatVec, const unsigned _LevelsUp) const noexcept(false) {
    DescriptorArray features(static_cast<size_t>(_Features.rows()));
    Descriptors::Map(features.data() -> data(), _Features.rows(), DescL) = _Features;
    transform(features, _BowVec, _FeatVec, _LevelsUp);
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::clear(const bool _Alloc) {
    if(_Alloc) {
        // Empty the current data (if has)
        if(m_pNodes) {
            m_pNodes -> clear();
            m_aWords.clear();
            m_pNodes -> shrink_to_fit();
            m_aWords.shrink_to_fit();
        } else {
            m_pNodes.reset(new std::vector<Node>());
        }
    } else {
        m_pNodes.reset();
        m_aWords.clear();
        m_aWords.shrink_to_fit();
    }
    m_bInit = false;
}

template <typename TScalar, size_t DescL>
size_t TemplatedVocabulary<TScalar, DescL>::stopPercent(double _Percent) {
    if(_Percent >= 100) {
        throw LogicException(TDBOW_LOG("Take no means for stop all the words!"));
    }
    if(_Percent < 0) {
        throw ParametersException(TDBOW_LOG("negative percent is illegal"));
    }
    if(_Percent >= 1)_Percent /= 100;

    std::vector<std::pair<WordValue, WordId>> words;
    words.reserve(m_aWords.size());
    // Be careful if change the word's structure
    for(size_t wordId = 0; wordId < m_aWords.size(); wordId++) {
        const auto& nodeId = m_aWords[wordId];
        const auto& node = (*m_pNodes)[nodeId];
        words.emplace_back(std::make_pair(node.weightBackup, nodeId));
    }
    std::sort(words.begin(), words.end());
    auto idx = static_cast<size_t>(words.size() * _Percent);
    return stopWords(words[idx].first);
}

template <typename TScalar, size_t DescL>
size_t TemplatedVocabulary<TScalar, DescL>::stopWords(const WordValue _MinLimit) {
    size_t count = 0;
    for(const auto& nodeId : m_aWords) {
        auto& node = (*m_pNodes)[nodeId];
        if(node.weight <= _MinLimit) {
            count++;
            node.weight = 0;
        } else {
            node.weight = node.weightBackup;
        }
    }
    return count;
}

/* ********************************************************************************
 *                               GET/SET METHODS                                  *
 ******************************************************************************** */

template <typename TScalar, size_t DescL>
NodeId TemplatedVocabulary<TScalar, DescL>::getParentNode(
        WordId _Wid, int _LevelsUp) const noexcept(false) {
    // confirm the parameters, be careful when change the structure of words.
    if(_LevelsUp < 0 || _Wid < 0 || _Wid >= m_aWords.size()) {
       throw OutOfRangeException(TDBOW_LOG("cannot find such node in this vocabulary."));
    }
    NodeId ret = m_aWords[_Wid]; // node id
    while(_LevelsUp--) {
        ret = (*m_pNodes)[ret].parent;
    }
    return ret;
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::getWordsFromNode(
        NodeId _Nid, std::vector<WordId>& _Words) const noexcept(false) {
    _Words.clear();
    _Words.shrink_to_fit();
    if((*m_pNodes)[_Nid].isLeaf()) {
        _Words.emplace_back((*m_pNodes)[_Nid].word_id);
        return;
    }
    assert(m_uiK > 0);
    _Words.reserve(static_cast<size_t>(m_uiK)); // ^1, ^2, ...
    std::vector<NodeId> parents(1, _Nid);
    while(!parents.empty()) {
        auto pid = parents.front();
        parents.erase(parents.begin());
        if(_Words.size() == _Words.capacity()) {
            _Words.reserve(_Words.size() * m_uiK);
        }
        for(const auto& cid : (*m_pNodes)[pid].children) {
            const auto& node = (*m_pNodes)[cid];
            if(node.isLeaf()) {
                _Words.emplace_back(node.word_id);
            } else {
                parents.emplace_back(cid);
            }
        } // for each child
    } // while !parents.empty
}

template <typename TScalar, size_t DescL>
float TemplatedVocabulary<TScalar, DescL>::getEffectiveLevels() const {
    size_t count = 0;
    for(auto nodeId : m_aWords) {
        for(; (*m_pNodes)[nodeId].id != 0; count++) {
            nodeId = (*m_pNodes)[nodeId].parent;
        }
    }
    return static_cast<float>((double)count / size());
}

template <typename TScalar, size_t DescL>
typename TemplatedVocabulary<TScalar, DescL>::Descriptor
TemplatedVocabulary<TScalar, DescL>::getWord(WordId _Wid) const noexcept(false) {
    if(_Wid < 0 || _Wid >= m_aWords.size()) {
        throw OutOfRangeException(TDBOW_LOG("Required word ID(" << _Wid << ") "
                                 "is out of range(0..." << m_aWords.size() - 1 << ")."));
    }
    return (*m_pNodes)[m_aWords[_Wid]].descriptor;
}

template <typename TScalar, size_t DescL>
WordValue TemplatedVocabulary<TScalar, DescL>::getWordWeight(
        WordId _Wid) const noexcept(false) {
    if(_Wid < 0 || _Wid >= m_aWords.size()) {
        throw OutOfRangeException(TDBOW_LOG("Required word ID(" << _Wid << ") "
                                 "is out of range(0..." << m_aWords.size() - 1 << ")."));
    }
    return (*m_pNodes)[m_aWords[_Wid]].weight;
}

template <typename TScalar, size_t DescL>
TemplatedVocabulary<TScalar, DescL>&
TemplatedVocabulary<TScalar, DescL>::setScoringType(ScoringType _Type) noexcept(false) {
    if(!m_pScoringObj || m_eScoring != _Type) {
        if(m_bInit) {
            throw LogicException(TDBOW_LOG("The vocabulary is "
                    "already created, must recreated to alter the scoring type"));
        }
        m_eScoring = _Type;
        _createScoringObject();
    }
    return *this;
}

template <typename TScalar, size_t DescL>
TemplatedVocabulary<TScalar, DescL>&
TemplatedVocabulary<TScalar, DescL>::setWeightingType(WeightingType _Type) noexcept(false) {
    if(m_eWeighting != _Type) {
        if(m_bInit) {
            throw LogicException(TDBOW_LOG("The vocabulary is "
                    "already created, must recreated to alter the weight type"));
        }
        m_eWeighting = _Type;
    }
    return *this;
}

/* ********************************************************************************
 *                             INPUT/OUTPUT METHODS                               *
 ******************************************************************************** */

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::save(
        const std::string& _Filename, const bool _Binary) const noexcept(false) {
    using namespace boost::filesystem;
    path file(_Filename);
    if(file.is_relative()) {
#ifdef PKG_DIR
        file = path(PKG_DIR)/"vocab"/_Filename;
#else
        file = temp_directory_path()/"vocab"/_Filename;
#endif
    }
    std::cout << "Trying to save vocabulary at: " << file.native() << std::endl;
    // Generate the directory
    if(!exists(file.parent_path())) {
        create_directories(file.parent_path());
    }
    // Remove the existed file, since remove all contents of directory is
    // dangerous, will throw an exception when the input is a non-empty
    // directory.
    if(exists(file)) {
        remove(file);
    }
    if(_Binary) {
        saveBinary(file);
    } else {
        saveYAML(file);
    }
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::load(
        const std::string& _Filename, LoadMode _Mode) noexcept(false) {
    boost::filesystem::path file(_Filename);
    if(file.is_relative()) {
#ifdef PKG_DIR
        file = boost::filesystem::path(PKG_DIR)/"vocab"/_Filename;
#else
        file = boost::filesystem::temp_directory_path()/"vocab"/_Filename;
#endif
    }
    if(!boost::filesystem::exists(file) || boost::filesystem::is_directory(file)) {
        throw FileNotExistException(TDBOW_LOG(file.native() << " not exist or is a directory!"));
    }
    switch(_Mode) {
        case LoadMode::BINARY:
            loadBinary(file);
            break;

        case LoadMode::YAML:
            loadYAML(file);
            break;

        case LoadMode::AUTO: default:
            auto extension = file.extension().native();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if(extension == ".yaml" || extension == ".yml") {
                loadYAML(file);
//            } else if(extension == ".bin" || extension == ".lz4") {
//                loadBinary(file);
            } else {
                loadBinary(file);
            }
            break;
    }
}

/**
 * @brief
 * Format YAML:
 * vocabulary
 * {
 *   k:
 *   L:
 *   scoringType:
 *   weightingType:
 *   initialized:
 *   nodes
 *   [
 *     {
 *       nodeId:
 *       parentId:
 *       weight:
 *       weight_backup:
 *       descriptor:
 *     }
 *   ]
 *   words
 *   [
 *     {
 *       wordId:
 *       nodeId:
 *     }
 *   ]
 * }
 *
 * The root node (index 0) is not included in the node vector
 * @author smallchimney
 */
template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::saveYAML(
        const boost::filesystem::path& _File,
        const std::function<std::string(Descriptor)>& _F) const noexcept(false) {
    if(m_pNodes == nullptr) {
        throw EmptyDataException(TDBOW_LOG("No data to save."));
    }
    std::ofstream out(_File.native());
    if(!out) {
        throw FileNotOpenException(TDBOW_LOG("Cannot open the output stream."));
    }
    YAML::Emitter yaml(out);
    yaml << YAML::BeginDoc << YAML::BeginMap
         << YAML::Key << "vocabulary" << YAML::Value << YAML::BeginMap
         << YAML::Key << "k" << YAML::Value << m_uiK
         << YAML::Key << "L" << YAML::Value << m_uiL
         << YAML::Key << "scoringType" << YAML::Value << m_eScoring
         << YAML::Key << "weightingType" << YAML::Value << m_eWeighting
         << YAML::Key << "initialized" << YAML::Value << m_bInit
         << YAML::Key << "nodes" << YAML::Value << YAML::BeginSeq;

    std::vector<NodeId> parents;
    parents.emplace_back(0); // root
    while(!parents.empty()) {
        // Note that the children will writen in inverse order
        NodeId pid = parents.front();
        parents.erase(parents.begin());

        const auto& parent = (*m_pNodes)[pid];
        for(const auto& cid : parent.children) {
            const auto& child = (*m_pNodes)[cid];
            yaml << YAML::BeginMap
                 << YAML::Key << "nodeId" << YAML::Value << cid
                 << YAML::Key << "parentId" << YAML::Value << pid
                 << YAML::Key << "weight" << YAML::Value << child.weight
                 << YAML::Key << "weight_backup" << YAML::Value << child.weightBackup
                 << YAML::Key << "descriptor" << YAML::Value
                 << (_F == nullptr ? util::toString(child.descriptor) : _F(child.descriptor))
                 << YAML::EndMap;

            // add to parent list
            if(!child.isLeaf()) {
                parents.emplace_back(cid);
            }
        }
    }
    yaml << YAML::EndSeq
         << YAML::Key << "words" << YAML::Value << YAML::BeginSeq;
    for(const auto& nodeId : m_aWords) {
        const auto& word = (*m_pNodes)[nodeId];
        yaml << YAML::BeginMap
             << YAML::Key << "wordId" << YAML::Value << word.word_id
             << YAML::Key << "nodeId" << YAML::Value << nodeId
             << YAML::EndMap;
    }
    // Doc {vocabulary {} } EndDoc
    yaml << YAML::EndMap << YAML::EndMap << YAML::EndDoc;
    out.close();
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::saveBinary(
        const boost::filesystem::path& _File, bool _Compressed) const noexcept(false) {
    if(m_pNodes == nullptr) {
        throw EmptyDataException(TDBOW_LOG("No data to save."));
    }
    std::ofstream out(_File.native(), std::ios::out|std::ios::binary);
    if(!out) {
        throw FileNotOpenException(TDBOW_LOG(
                "Could not open file :" << _File.native() << " for writing."));
    }
    write(out, _Compressed);
    out.close();
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::loadYAML(
        const boost::filesystem::path& _File,
        const std::function<Descriptor(std::string)>& _F) noexcept(false) {
    try {
        auto yaml = YAML::LoadFile(_File.native());
        const auto& vocab = yaml["vocabulary"];
        clear(true);
        // Load parameters
        m_uiK << vocab["k"];
        m_uiL << vocab["L"];
        setScoringType((ScoringType)vocab["scoringType"].as<int>());
        setWeightingType((WeightingType)vocab["weightingType"].as<int>());
        if(vocab["initialized"].IsDefined()) {
            m_bInit << vocab["initialized"];
        } else {
            m_bInit = true;
        }
        // Load nodes
        m_pNodes -> resize(vocab["nodes"].size() + 1); // +1 to include root
        (*m_pNodes)[0].id = 0;
        NodeId nodeId;
        for(const auto& node : vocab["nodes"]) {
            nodeId << node["nodeId"];
            auto& nd = (*m_pNodes)[nodeId];
            nd.id = nodeId;
            nd.parent << node["parentId"];
            (*m_pNodes)[nd.parent].children.emplace_back(nodeId);
            nd.weight << node["weight"];
            if(node["weight_backup"].IsDefined()) {
                nd.weightBackup << node["weight_backup"];
            } else {
                nd.weightBackup = nd.weight;
            }
            std::string buf;
            buf << node["descriptor"];
            if(_F == nullptr) {
                util::fromString(buf, nd.descriptor);
            } else {
                nd.descriptor = _F(buf);
            }
            nd.weight << node["weight"];
        }
        // Load words
        m_aWords.resize(vocab["words"].size());
        WordId wordId;
        for(const auto& word : vocab["words"]) {
            wordId << word["wordId"];
            nodeId << word["nodeId"];
            m_aWords[wordId] = nodeId;
            (*m_pNodes)[nodeId].word_id = wordId;
        }
        _buildSearchTree();
    } catch(YAML::Exception& e) {
        throw FormatException(TDBOW_LOG(
                "Vocabulary file may be invalid: " << e.what()));
    }
}


template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::loadBinary(
        const boost::filesystem::path& _File) {
    std::ifstream in(_File.native(), std::ios::in|std::ios::binary);
    if(!in) {
        throw FileNotOpenException(TDBOW_LOG(
                "Could not open file :" << _File.native() << " for reading."));
    }
    read(in);
    in.close();
}

#define CHUNK_SIZE 1e4

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::write(
        std::ostream& _Out, bool _Compressed) const noexcept(false) {
    // Confirm the descriptor signature is same.
    using traits::type_trait::type_id;
    static constexpr type_id idScalar = traits::type_traits<TScalar>::id();
    static constexpr auto valL = DescL;
    _Out.write((char*)&idScalar, sizeof(type_id));
    _Out.write((char*)&valL, sizeof(valL));
    // Save into a stream for compress
    _Out.write((char*)&_Compressed, sizeof(bool));
    std::stringstream streamBuf;
    // Fix the length of nodes
    uint64_t nodesNum = m_pNodes -> size(), wordsNum = m_aWords.size();
    streamBuf.write((char*)&nodesNum       , sizeof(uint64_t));
    streamBuf.write((char*)&wordsNum       , sizeof(uint64_t));
    streamBuf.write((char*)&m_uiK          , sizeof(m_uiK));
    streamBuf.write((char*)&m_uiL          , sizeof(m_uiL));
    streamBuf.write((char*)&m_eScoring     , sizeof(m_eScoring));
    streamBuf.write((char*)&m_eWeighting   , sizeof(m_eWeighting));
    streamBuf.write((char*)&m_bInit        , sizeof(m_bInit));
    // Save nodes
    std::vector<NodeId> parents={0};// root
    while(!parents.empty()) {
        // Note that the children will writen in inverse order
        const NodeId pid = parents.back();
        const Node& parent = (*m_pNodes)[pid];
        parents.pop_back();
        for(const auto cid : parent.children) {
            const Node& child = (*m_pNodes)[cid];
            streamBuf.write((char*)&cid         , sizeof(cid));
            streamBuf.write((char*)&pid         , sizeof(pid));
            streamBuf.write((char*)&child.weight, sizeof(child.weight));
            streamBuf.write((char*)&child.weightBackup, sizeof(child.weightBackup));
            // Since the words will save later, no need for redundant
            util::toBinary(child.descriptor, streamBuf);
            // Add to parent list
            if(!child.isLeaf())parents.emplace_back(cid);
        }
    }
    // Save words
    for(const auto nodeId : m_aWords) {
        // Since the wordId is consecutive, the redundant data can be saved.
        // If you change the data structure from `std::vector` to others, be careful.
        // const auto& word = (*m_pNodes)[nodeId];
        // streamBuf.write((char*)&word.word_id, sizeof(word.word_id));
        streamBuf.write((char*)&nodeId, sizeof(nodeId));
    }

    if(_Compressed) {
        LZ4_streamHC_t lz4Stream = {0};
        auto chunkSize = static_cast<size_t>(CHUNK_SIZE);
        std::vector<char> compressed(chunkSize + 512, 0);
        std::vector<char> chunkBuf(chunkSize, 0);
        auto bufferSize = static_cast<size_t>(streamBuf.tellp());
        size_t chunkNum = bufferSize / chunkSize;
        if(bufferSize % chunkSize)chunkNum++;
        _Out.write((char*)&chunkNum, sizeof(chunkNum));
        auto leftBufSize = bufferSize;
        while(leftBufSize) {
            chunkSize = std::min(chunkSize, leftBufSize);
            streamBuf.read(&chunkBuf[0], chunkSize);
            const auto len = static_cast<int32_t>(LZ4_compress_HC_continue(&lz4Stream,
                    &chunkBuf[0], &compressed[0], static_cast<int>(chunkSize),
                    static_cast<int>(compressed.capacity())));
            leftBufSize -= chunkSize;
            _Out.write((char*)&len, sizeof(len));
            _Out.write(&compressed[0], len);
        }
    } else {
        _Out << streamBuf.rdbuf();
    }
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::read(std::istream& _In) noexcept(false) {
    // Confirm the descriptor signature is same.
    using traits::type_trait::type_id;
    static constexpr type_id ID_SCALAR = traits::type_traits<TScalar>::id();
    static constexpr auto VALUE_L = DescL;
    type_id idScalar;
    _In.read((char*)&idScalar, sizeof(type_id));
    auto valL = DescL;  // generate the var with the same type of L
    _In.read((char*)&valL, sizeof(valL));
    if(idScalar != ID_SCALAR || valL != VALUE_L) {
        throw FormatException(TDBOW_LOG("The descriptor's format cannot be matched."));
    }
    // Whether compressed
    bool compressed;
    _In.read((char*)&compressed, sizeof(bool));
    clear(true);
    // Read and decompress the stream (if was compressed)
    std::istream* in = &_In;
    std::stringstream decompressStream; // used only in compressed mode
    if(compressed) {
        LZ4_streamDecode_t lz4Stream = {0};
        auto chunkSize = static_cast<int32_t>(CHUNK_SIZE);
        std::vector<char> decompressed(static_cast<size_t>(chunkSize));
        std::vector<char> chunkBuf(static_cast<size_t>(chunkSize + 512));
        size_t chunkNum;
        in -> read((char*)&chunkNum, sizeof(chunkNum));
        for(size_t i = 0; i < chunkNum; i++) {
            in -> read((char*)&chunkSize, sizeof(chunkSize));
            in -> read(&chunkBuf[0], chunkSize);
            const int len = LZ4_decompress_safe_continue(&lz4Stream,
                    &chunkBuf[0], &decompressed[0], chunkSize,
                    static_cast<int>(decompressed.capacity()));
            decompressStream.write(&decompressed[0], len);
        }
        in = &decompressStream;
    }
    // Reset the nodes size
    uint64_t nodesNum, wordsNum;
    in -> read((char*)&nodesNum    , sizeof(uint64_t));
    in -> read((char*)&wordsNum    , sizeof(uint64_t));
    // Init all nodes, and reset the value after load
    m_pNodes -> resize(nodesNum);
    m_aWords.resize(wordsNum);
    (*m_pNodes)[0].id = 0;
    // Reset the parameters
    in -> read((char*)&m_uiK       , sizeof(m_uiK));
    in -> read((char*)&m_uiL       , sizeof(m_uiL));
    in -> read((char*)&m_eScoring  , sizeof(m_eScoring));
    in -> read((char*)&m_eWeighting, sizeof(m_eWeighting));
    in -> read((char*)&m_bInit     , sizeof(m_bInit));
    _createScoringObject();
    // Load nodes
    for(size_t i = 1; i < nodesNum; i++) {
        NodeId id;
        in -> read((char*)&id, sizeof(id));
        auto& node = (*m_pNodes)[id];
        node.id = id;
        in -> read((char*)&node.parent, sizeof(node.parent));
        in -> read((char*)&node.weight, sizeof(node.weight));
        in -> read((char*)&node.weightBackup, sizeof(node.weightBackup));
        // Since the children of a single parent is processed consecutively,
        // the original order of children will be kept, nothing will be changed.
        (*m_pNodes)[node.parent].children.emplace_back(id);
        util::fromBinary(*in, node.descriptor);
    }
    // Load words
    for(size_t i = 0; i < wordsNum; i++) {
        // Since the wordId is consecutive, the redundant data can be saved.
        // If you change the data structure from `std::vector` to others, be careful.
        // WordId wid;
        // in -> read((char*)&wid, sizeof(wid));
        // auto& word = m_aWords[wid];
        auto& word = m_aWords[i];
        in -> read((char*)&word, sizeof(word));
        (*m_pNodes)[word].word_id = i;
    }
    _buildSearchTree();
}

/**
 * @breif Writes printable information of the vocabulary
 * @param _Out    Output stream
 * @param _Vocab  Vocabulary data
 */
template <typename TScalar, size_t DescL>
std::ostream& operator<<(std::ostream &_Out, const TemplatedVocabulary<TScalar, DescL>& _Vocab) {
    _Out << "Vocabulary: k = " << _Vocab.getBranchingFactor()
         << ", L = " << _Vocab.getDepthLevels()
         << ", Weighting = ";

    switch(_Vocab.getWeightingType()) {
        case WeightingType::TF_IDF: _Out << "tf-idf"; break;
        case WeightingType::TF: _Out << "tf"; break;
        case WeightingType::IDF: _Out << "idf"; break;
        case WeightingType::BINARY: _Out << "binary"; break;
    }

    _Out << ", Scoring = ";
    switch(_Vocab.getScoringType()) {
        case L1_NORM: _Out << "L1-norm"; break;
        case L2_NORM: _Out << "L2-norm"; break;
        case CHI_SQUARE: _Out << "Chi square distance"; break;
        case KL: _Out << "KL-divergence"; break;
        case BHATTACHARYYA: _Out << "Bhattacharyya coefficient"; break;
        case DOT_PRODUCT: _Out << "Dot product"; break;
    }
    return _Out << ", Number of words = " << _Vocab.size();
}

/* ********************************************************************************
 *                                INNER METHODS                                   *
 ******************************************************************************** */

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::_getFeatures(
        const ConstDataSet& _TrainingData, std::vector<DescriptorConstPtr>& _Features) const noexcept(false) {
    if(_TrainingData.empty()) {
        throw EmptyDataException(TDBOW_LOG("Empty dataset."));
    }
    _Features.clear();
    _Features.shrink_to_fit();
    const auto reserveSize = std::min(_Features.max_size(),
            _TrainingData.size() * _TrainingData[0].size());
    _Features.reserve(reserveSize);
    for(const auto& image : _TrainingData) {
        for(const auto& feature : image) {
            _Features.emplace_back(feature);
        }
    }
    if(_Features.empty()) {
        throw EmptyDataException(TDBOW_LOG("Empty dataset."));
    }
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::_transform(
        const Descriptor& _Feature, WordId& _ID) const noexcept(false) {
    WordValue weight = 0;
    _transform(_Feature, _ID, weight);
}

template <typename TScalar, size_t DescL>
NodeId TemplatedVocabulary<TScalar, DescL>::_transform(
        const Descriptor& _Feature, WordId& _WordId, WordValue& _Weight,
        const unsigned int _LevelsUp) const noexcept(false) {
    if(!ready()) {
        throw NotInitailizedException(TDBOW_LOG(
                "The vocabulary is empty, must be created before transform."));
    }
    // In TDBoW's recommend case
    if(m_uiL == 1 || !std::is_same<typename util::binary_type, TScalar>()) {
        Descriptor query = _Feature;
        size_t index;
        typename FLANNDist::ResultType distance;
        FLANNMatrix<decltype(distance)> dis(&distance, 1, 1);
        FLANNMatrix<size_t> idx(&index, 1, 1);
        m_pWordsSearchTree -> knnSearch(FLANNMatrix<TScalar>(query.data(), 1, DescL),
                                        idx, dis, 1, ::flann::SearchParams(-1));
        _WordId = static_cast<WordId>(index);
        _Weight = (*m_pNodes)[m_aWords[index]].weight;
        return getParentNode(_WordId, _LevelsUp);
    } else {  // In DBoW2's original case
        // Propagate the feature down the tree
        auto requiredLevel = m_uiL >= _LevelsUp ? m_uiL - _LevelsUp : 0;
        NodeId selected = 0, ret = 0; // start from root
        unsigned currentLevel = 0;
        do {
            auto min = std::numeric_limits<distance_type>::max();
            for(const auto& cid : (*m_pNodes)[selected].children) {
                const auto& child = (*m_pNodes)[cid];
                const auto d = util::distance(_Feature, child.descriptor);
                if(d < min) {
                    min = d;
                    selected = cid;
                }
            }
            if(currentLevel++ == requiredLevel) {
                ret = selected;
            }
        } while(!(*m_pNodes)[selected].isLeaf());
        _WordId = (*m_pNodes)[selected].word_id;
        _Weight = (*m_pNodes)[selected].weight;
        return ret;
    }
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::_buildTree(
        const std::vector<DescriptorConstPtr>& _Descriptors) {
    bool loadRequired = m_pCheckPoint ?
            m_pCheckPoint -> completed(CheckPointUtil::CLUSTER_BEG) : false;

    std::queue<Task> tasks;
    // Prepare for the init tasks
    if(!loadRequired) {
        std::vector<size_t> originIndices(_Descriptors.size());
#ifdef FOUND_OPENMP
        #pragma omp parallel for
#endif
        for(size_t i = 1; i < _Descriptors.size(); i++) {
            originIndices[i] = i;
        }
        tasks.push(std::make_tuple(0, _Descriptors, 1, originIndices));
    } else {
        m_pCheckPoint -> ckptLoadInfo();
        m_pCheckPoint -> ckptLoadNodes(m_pNodes);
        m_pCheckPoint -> ckptLoadTasks(tasks, _Descriptors);
        if(m_pCheckPoint -> getStatus() == CheckPointUtil::CLUSTER_BEG) {
            m_pCheckPoint -> ckptLoaded();
        }
    }

    while(!tasks.empty()) {
        if(m_pCheckPoint) {
            if(!loadRequired) {
                m_pCheckPoint -> setStatus(CheckPointUtil::CLUSTER_BEG);
                m_pCheckPoint -> ckptSaveInfo();
                m_pCheckPoint -> ckptSaveNodes(m_pNodes);
                m_pCheckPoint -> ckptSaveTasks(tasks);
                m_pCheckPoint -> ckptDone();
                m_pCheckPoint -> output("New task start.\n");
            } else {
                loadRequired = false;
            }
        }

        const auto& task = tasks.front();
        const auto& descriptors = std::get<1>(task);
        const auto parentId = std::get<0>(task);
        const auto level = std::get<2>(task);
        const auto indices = std::get<3>(task);
        if(descriptors.empty()) {
            tasks.pop();
            continue;
        }

        // Features associated to each cluster using K-means
        DescriptorArray centers(0);
        std::vector<std::vector<DescriptorConstPtr>> clusters(0);
        const auto belong = KMeansUtil(m_uiK)
                .process(descriptors, centers, clusters, m_pCheckPoint);
        assert(descriptors.size() == belong.size());
        assert(centers.size() == clusters.size());
        std::vector<std::vector<size_t>> childIndices(centers.size());
#ifdef FOUND_OPENMP
        #pragma omp parallel for
#endif
        for(size_t i = 0; i < centers.size(); i++) {
            childIndices[i].reserve(clusters[i].size());
        }
        for(size_t i = 0; i < descriptors.size(); i++) {
            childIndices[belong[i]].emplace_back(indices[i]);
        }
        tasks.pop();    // no need for descriptors later

        // Create children nodes
        for(size_t i = 0; i < clusters.size(); ++i) {
            auto nextId = m_pNodes -> back().id + 1;
            m_pNodes -> emplace_back(nextId);
            m_pNodes -> back().descriptor = centers[i];
            m_pNodes -> back().parent = parentId;
            (*m_pNodes)[parentId].children.emplace_back(nextId);
        }

        // Add tasks to the queue
        if(level < m_uiL) {
            for(size_t i = 0; i < clusters.size(); i++) {
                const auto& childId = (*m_pNodes)[parentId].children[i];
                tasks.push(std::make_tuple(
                        childId, clusters[i], level + 1, childIndices[i]));
            }
        }
    }

    // Updating checkpoint
    if(m_pCheckPoint) {
        m_pCheckPoint -> setStatus(CheckPointUtil::CLUSTER_END);
        m_pCheckPoint -> ckptSaveInfo();
        m_pCheckPoint -> ckptSaveNodes(m_pNodes);
        m_pCheckPoint -> ckptDone();
        m_pCheckPoint -> output("All nodes clustered.\n");
    }
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::_buildSearchTree() {
    if(!m_pNodes || m_pNodes -> size() <= 1 || m_aWords.empty())return;
    m_aSearchTreeLeaves.reserve(m_aWords.size());
    for(const auto& word : m_aWords) {
        m_aSearchTreeLeaves.emplace_back((*m_pNodes)[word].descriptor);
    }
    m_pWordsSearchTree = FLANNIndexPtr(new FLANNIndex(
            FLANNMatrix<TScalar>(m_aSearchTreeLeaves.data() -> data(),
                        m_aSearchTreeLeaves.size(), DescL),
            ::flann::KDTreeIndexParams(1)));
    m_pWordsSearchTree -> buildIndex();
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::_createWords() {
    if(!m_pNodes || m_pNodes -> empty())return;
    if(m_pNodes -> size() == 1) {
        if(!m_pCheckPoint)return;
        if(m_pCheckPoint -> getStatus() == CheckPointUtil::CLUSTER_END) {
            // Load the cluster results
            m_pCheckPoint -> ckptLoadInfo();
            m_pCheckPoint -> ckptLoadNodes(m_pNodes);
            m_pCheckPoint -> ckptLoaded();
        }
    }
    m_aWords.reserve(static_cast<size_t>(pow(m_uiK, m_uiL)));
    for(auto& node : *m_pNodes) {
        if(node.isLeaf()) {
            node.word_id = m_aWords.size();
            m_aWords.emplace_back(node.id);
        }
    }
}

template <typename TScalar, size_t DescL>
void TemplatedVocabulary<TScalar, DescL>::_setNodeWeights(const ConstDataSet& _TrainingData) {
    if(empty())return;

    if(m_eWeighting == WeightingType::TF ||
       m_eWeighting == WeightingType::BINARY) {
        // idf part must be 1 always
        for(const auto& nodeId : m_aWords) {
            auto& node = (*m_pNodes)[nodeId];
            node.weight = node.weightBackup = 1;
        }
        return;
    }
    // IDF and TF-IDF: we calculate only the idf path now

    // Note: this actually calculates the idf part of the tf-idf score.
    // The complete tf-idf score is calculated in ::transform
    const size_t& NWords = size();
    const size_t& NDocs  = _TrainingData.size();
    std::vector<size_t> Ni(NWords, 0);

    boost::dynamic_bitset<> counted(NWords);
    for(const auto& image : _TrainingData) {
        counted.reset();
        for(const auto& descriptor : image) {
            WordId wordId = transform(*descriptor);
            if(!counted.test(wordId)) {
                counted.set(wordId);
                Ni[wordId]++;
            }
        }
    }

    // set ln(N/Ni)
    for(size_t i = 0; i < NWords; i++) {
        if(Ni[i] > 0) {
            auto& node = (*m_pNodes)[m_aWords[i]];
            node.weight = node.weightBackup =
                    static_cast<WordValue>(std::log((double)NDocs / (double)Ni[i]));
        }// else // This cannot occur if using k-means++
    }
}

/* ********************************************************************************
 *                                LOCAL METHODS                                   *
 ******************************************************************************** */

template<typename T>
void operator << (T& i, const YAML::Node& node) {
    i = node.as<T>();
}

} // namespace TDBoW

#endif  // __ROCKAUTO_TDBOW_TEMPLATED_VOCABULARY_HPP__
