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
   * File Name     : TemplatedCheckPoint.hpp
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2020-04-02 16:39:56
   * Last Modified : smallchimney
   * Modified Time : 2020-04-07 00:00:15
************************************************************************* */

#ifndef __ROCKAUTO_TDBOW_TEMPLATED_CHECK_POINT_HPP__
#define __ROCKAUTO_TDBOW_TEMPLATED_CHECK_POINT_HPP__

#include <TDBoW/elements/TemplatedDescriptor.hpp>
#include <TDBoW/elements/TemplatedNode.hpp>
#include <TDBoW/elements/TemplatedTask.hpp>

#include <queue>
#include <cstdlib>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm/replace_if.hpp>

namespace TDBoW {

constexpr uint8_t TDBOW_CKPT_HEADER = 'T' ^ 'D' ^ 'B' ^ 'o' ^ 'W';
constexpr uint8_t TDBOW_CKPT_VERSION_MAJOR = 0;
constexpr uint8_t TDBOW_CKPT_VERSION_MINOR = 1;

template <typename TScalar, size_t DescL>
class TemplatedCheckPoint {
protected:
    // Load the typedef from Vocabulary
    typedef TemplatedDescriptorUtil<TScalar, DescL> util;
    TDBOW_DESCRIPTOR_DEF(util)
    typedef sNode<TScalar, DescL> Node;
    typedef std::vector<Node> Nodes;
    typedef std::unique_ptr<Nodes> NodesPtr;
    typedef typename sTask<TScalar, DescL>::Task Task;
    // Use boost::filesystem
    typedef boost::filesystem::path path;

    // Const variable define by standard (Ver 0.1)
    static constexpr size_t TDBOW_DESCRIPTOR_SIZE = DescL * sizeof(TScalar);

public:
    /** @brief Create processing stage */
    typedef enum eStatus {
        UNKNOWN = 0,
        LOADING,     // initialization
        CLUSTER_BEG, // right before cluster
        CLUSTER_INIT,// during cluster init
        CLUSTER_ITER,// during cluster init
        CLUSTER_END, // clustering nodes
        COLLECT_BEG, // collecting words
        COLLECT_END, // collecting words
        WEIGHT_BEG,  // calculating weight
        WEIGHT_END,  // calculating weight
        FINISHED     // finished
    } Status;

    typedef enum eLevel {
        NONE = 0,
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        FATAL
    } Level;

    typedef std::unique_ptr<TemplatedCheckPoint> Ptr;

    TemplatedCheckPoint() = delete;
    TemplatedCheckPoint(const std::string& _Log, size_t _Scale,
            unsigned _K, unsigned _L, bool _Console = true);
    virtual ~TemplatedCheckPoint();

    /** @brief Whether this checkpoint is valid for file */
    bool valid() const {
        return !m_strTag.empty();
    }

    /** @brief Whether this checkpoint is valid for console */
    bool console() const {
        return valid() && m_bOutputConsole;
    }

    /** @brief Return the current processing stage */
    Status getStatus() const {
        return m_eStatus;
    }

    /** @brief Update the status */
    void setStatus(Status&& _Stage) {
        m_eStatus = _Stage;
    }

    /**
     * @brief  Check whether this stage has been done.
     * @param _Status  The stage to be checked.
     * @return         Whether the stage was completed.
     */
    bool completed(const Status&& _Stage) const {
        if(m_eStatus == UNKNOWN)return false;
        return m_eStatus >= _Stage;
    }

    /**
     * @brief  Touch temp checkpoint file and save the header,
     *         will clear the content from old temp checkpoint.
     * @author smallchimney
     */
    void ckptSaveInfo() noexcept(false);

    /**
     * @brief  Try to load the checkpoint header
     * @author smallchimney
     * @throws If user set the wrong file, we don't want to remove
     *         the file when error, so simply throw exception in
     *         this case.
     */
    void ckptLoadInfo() noexcept(false);

    /**
     * @brief  Saving the nodes into temp checkpoint.
     * @author smallchimney
     * @param  _Nodes  The existing nodes.
     */
    void ckptSaveNodes(const NodesPtr& _Nodes) noexcept(false);

    /**
     * @brief  Try to load the checkpoint nodes.
     * @author smallchimney
     * @param  _Nodes  The nodes to be filled.
     * @throws         If user set the wrong file, we don't want to remove
     *                 the file when error, so simply throw exception in
     *                 this case.
     */
    void ckptLoadNodes(NodesPtr& _Nodes) noexcept(false);

    /**
     * @brief  Saving the tasks into temp checkpoint.
     * @author smallchimney
     * @param  _Tasks  The existed tasks, including the current one.
     */
    void ckptSaveTasks(const std::queue<Task>& _Tasks) noexcept(false);

    /**
     * @brief  Try to load the checkpoint tasks.
     * @author smallchimney
     * @param  _Tasks   The tasks to be filled.
     * @param  _Dataset Train dataset to get true address.
     * @throws          If user set the wrong file, we don't want to remove
     *                  the file when error, so simply throw exception in
     *                  this case.
     */
    void ckptLoadTasks(std::queue<Task>& _Tasks,
            const std::vector<DescriptorConstPtr>& _Dataset) noexcept(false);

    /**
     * @brief  Save cluster's each iteration.
     * @author smallchimney
     * @param  _IterNum  Current iteration count
     * @param  _Loss     Current loss
     * @param  _Belong   Current iteration result
     */
    void ckptSaveIter(const size_t& _IterNum,
                      const distance_type& _Loss,
                      const std::vector<size_t>& _Belong) noexcept(false);

    /**
     * @brief  Try to load cluster's each iteration.
     * @author smallchimney
     * @param  _IterNum  Previous iteration count
     * @param  _Loss     Previous loss
     * @param  _Len      Scale of the descriptors
     * @param  _Belong   Previous iteration result
     */
    void ckptLoadIter(size_t& _IterNum,
                      distance_type& _Loss,
                      const size_t& _Len,
                      std::vector<size_t>& _Belong) noexcept(false);

    /**
     * @brief  Finish temp checkpoint writing, will close the temp file and
     *         replace the stable checkpoint file.
     *         Especially, if the {@code m_eStatus} equals with {@code FINISHED},
     *         the stable checkpoint file will be removed.
     * @author smallchimney
     */
    void ckptDone() noexcept(false);

    /**
     * @brief  Finish checkpoint reading.
     * @author smallchimney
     */
    void ckptLoaded() noexcept(false);

    /**
     * @brief  Get the text logger file path
     * @author smallchimney
     * @return The path of text logger
     */
    path getLogFile() noexcept(false) {
        return _getLogFile(m_strTag);
    }

    /**
     * @brief  Get the checkpoint logger file path
     * @author smallchimney
     * @return The path of text logger
     */
    path getCkptFile() noexcept(false) {
        return _getCkptFile(m_strTag);
    }

    /**
     * @brief  Output the log to file (maybe console also)
     * @param _Content  The output log content
     */
    void output(const std::string& _Content, const Level _Level = INFO);

    /**
     * @brief  Output the log to file (maybe console also)
     * @return The output log content
     */
    std::string output(const Level _Level = INFO);

    std::stringstream& str() {
        return m_strStream;
    }
    
protected:

    /** @brief Get the instance of file logger stream */
    std::ofstream& _getFileLogger() noexcept(false) {
        if(!valid() || m_pFileLogger == nullptr) {
            throw LogicException(TDBOW_LOG("Forbid logging."));
        }
        return *m_pFileLogger;
    }

    /** @brief Get the instance of checkpoint stream */
    std::ofstream& _getCheckPoint() noexcept(false) {
        if(!valid() || m_pCheckPoint == nullptr) {
            throw LogicException(TDBOW_LOG("Forbid logging."));
        }
        return *m_pCheckPoint;
    }

    /**
     * @brief  Check the size of stable checkpoint is valid
     * @author smallchimney
     * @return Whether the size is legal
     */
    bool _ckptCheckFile() noexcept(false);

    /**
     * @brief  Get the text logger file path
     * @author smallchimney
     * @param  _Log   The filename of the logger, should be fit OS.
     * @return        The path of text logger
     */
    static path _getLogFile(const std::string& _Log) noexcept(false);

    /**
     * @brief  Get the checkpoint logger file path
     * @author smallchimney
     * @param  _Log   The filename of the logger, should be fit OS.
     * @return        The path of text logger
     */
    static path _getCkptFile(const std::string& _Log) noexcept(false);

    /**
     * @brief  Get the temp checkpoint logger file path
     * @author smallchimney
     * @param  _Log   The filename of the logger, should be fit OS.
     * @return        The path of text logger
     */
    static path _getTmpCkptFile(const std::string& _Log) noexcept(false);

    /**
     * @brief  Set the logger workspace, to be honest the best
     *         choice is TMP directory, but since the checkpoints
     *         only valid in case break the normal creating, and
     *         usually work in very large scale dataset.
     *         Obviously, it's worst news the computer just break
     *         down, and OS just friendly delete our checkpoints
     *         for us. So finally we decide to put the logger file
     *         in {@code PKG_DIR} (as default) or "$HOME/.tdbow/"
     * @author smallchimney
     * @return Existed workspace directory's path.
     * @throws Only if workspace existed, but not a directory.
     */
    static path _getLoggerDir() noexcept(false);

    /**
     * @brief  Try to fix the invalid filename.
     * @author smallchimney
     * @param  _Log  The user set logger tag.
     * @return       The fixed logger tag.
     */
    static std::string format(const std::string& _Log) noexcept;

private:

    /** @brief The checkpoint's tag */
    std::string m_strTag;

    /** @brief Record the BoW K & L */
    uint32_t m_uiK, m_uiL;

    /** @brief Record the dataset scale */
    uint64_t m_ulScale;

    /** @brief Whether output into console */
    bool m_bOutputConsole;

    /** @brief Current checkpoint status */
    Status m_eStatus;

    /** @brief Limit for logger, no use in this version */
    Level m_eLevel = INFO;

    /** @brief String stream to be used */
    std::stringstream m_strStream;

    /** @brief The file stream of text logger */
    std::shared_ptr<std::ofstream> m_pFileLogger = nullptr;

    /** @brief The output file stream of temp checkpoint */
    std::shared_ptr<std::ofstream> m_pCheckPoint = nullptr;

    /** @brief The input file stream of temp checkpoint */
    std::shared_ptr<std::ifstream> m_pReader = nullptr;

}; // class TemplatedCheckPoint

/* ********************************************************************************
 *                        CONSTRUCTION && INITIALIZATION                          *
 ******************************************************************************** */

template <typename TScalar, size_t DescL>
TemplatedCheckPoint<TScalar, DescL>::TemplatedCheckPoint(
        const std::string& _Log, const size_t _Scale,
        const unsigned _K, const unsigned _L, bool _Console)
        : m_uiK(_K), m_uiL(_L), m_ulScale(_Scale), m_bOutputConsole(_Console) {
    m_strTag = format(_Log);
    if(m_strTag.empty())return; // ignore the logger
    // Check && prepare the workspace
    const auto logFile = getLogFile();
    const auto ckptFile = getCkptFile();
    if(exists(logFile.parent_path())) {
        create_directories(logFile.parent_path());
    }
    if(exists(ckptFile.parent_path())) {
        create_directories(ckptFile.parent_path());
    }
    // Try to find the existed checkpoint
    bool isExisted = exists(getCkptFile());
    bool logFirst = false;
    if(!isExisted && exists(getLogFile())) {
        logFirst = true;
        remove(getLogFile()); // remove the old text logger
    }
    m_pFileLogger.reset(new std::ofstream(getLogFile().native(), std::ios::app));
    if(!*m_pFileLogger) {
        throw FileNotOpenException(TDBOW_LOG(
                getLogFile().native() << " cannot be open."));
    }
    if(logFirst) {
        output("Remove the stale text logger, should not use static tag.\n");
    }
    if(isExisted) {
        str() << "Detected broken processing, try to continue..." << std::endl;
        output();
        try {
            ckptLoadInfo();
            if(!_ckptCheckFile()) {
                throw FormatException(TDBOW_LOG(
                        "Checkpoint file size illegal."));
            }
            ckptLoaded();
            switch(m_eStatus) {
                case UNKNOWN: default:
                    throw FormatException(TDBOW_LOG("Unsupported stage loaded."));

                case LOADING:
                    output("Continue succeed: Data loaded.\n");
                    break;

                case CLUSTER_BEG:
                    output("Continue succeed: Ready to cluster.\n");
                    break;

                case CLUSTER_ITER:
                    output("Continue succeed: During a clustering.\n");
                    break;

                case CLUSTER_END:
                    output("Continue succeed: Cluster finished.\n");
                    break;
            }
            output();
            return;
        } catch (FormatException& ex) {
            output(ex.what(), ERROR);
            output("Remove the bad checkpoint file.\n");
            ckptLoaded();
            remove(ckptFile);
        }
    }
    // A bit useless, just for check the writable && mark the text logger file valid.
    setStatus(LOADING);
    ckptSaveInfo();
    ckptDone();
}

template <typename TScalar, size_t DescL>
TemplatedCheckPoint<TScalar, DescL>::~TemplatedCheckPoint() {
    ckptLoaded();
    ckptDone();
    if(m_pFileLogger && *m_pFileLogger) {
        m_pFileLogger -> close();
    }
}

/* ********************************************************************************
 *                            INPUT/OUTPUT METHODS                                *
 ******************************************************************************** */

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::ckptSaveInfo() noexcept(false) {
    assert(m_pReader == nullptr);
    if(m_strTag.empty())return;
    const auto tmpFile = _getTmpCkptFile(m_strTag);
    // reset the current temp checkpoint
    if(m_pCheckPoint) {
        m_pCheckPoint -> close();
    }
    m_pCheckPoint.reset(new std::ofstream(tmpFile.native(), std::ios::binary));
    auto& logger = _getCheckPoint();
    if(!logger) {
        throw FileNotOpenException(TDBOW_LOG(
                tmpFile.native()));
    }
    logger.write((char*)&TDBOW_CKPT_HEADER, sizeof(TDBOW_CKPT_HEADER));
    logger.write((char*)&TDBOW_CKPT_VERSION_MAJOR, sizeof(TDBOW_CKPT_VERSION_MAJOR));
    logger.write((char*)&TDBOW_CKPT_VERSION_MINOR, sizeof(TDBOW_CKPT_VERSION_MINOR));
    uint64_t descL = DescL;
    logger.write((char*)&descL, sizeof(descL));
    logger.write((char*)&m_ulScale, sizeof(m_ulScale));
    logger.write((char*)&m_uiL, sizeof(m_uiL));
    logger.write((char*)&m_uiK, sizeof(m_uiK));
    auto status = static_cast<int32_t>(m_eStatus);
    logger.write((char*)&status, sizeof(status));
}

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::ckptLoadInfo() noexcept(false) {
    assert(m_pCheckPoint == nullptr);
    if(m_strTag.empty())return;
    if(m_pReader) {
        m_pReader -> close();
    }
    const auto file = getCkptFile();
    if(!exists(file)) {
        throw FileNotExistException(TDBOW_LOG(file.native()));
    }
    m_pReader.reset(new std::ifstream(file.native(), std::ios::binary));
    auto& in = *m_pReader;
    if(!in) {
        throw FileNotOpenException(TDBOW_LOG(file.native()));
    }
    do {
        uint8_t uint8;
        in.read((char*)&uint8, sizeof(uint8));
        if(in.eof() || uint8 != TDBOW_CKPT_HEADER)break;
        in.read((char*)&uint8, sizeof(uint8));
        if(in.eof() || uint8 != TDBOW_CKPT_VERSION_MAJOR)break;
        in.read((char*)&uint8, sizeof(uint8));
        if(in.eof() || uint8 != TDBOW_CKPT_VERSION_MINOR)break;
        uint64_t uint64;
        in.read((char*)&uint64, sizeof(uint64));
        if(in.eof() || uint64 != DescL)break;
        in.read((char*)&uint64, sizeof(uint64));
        if(in.eof() || uint64 != m_ulScale)break;
        uint32_t uint32;
        in.read((char*)&uint32, sizeof(uint32));
        if(in.eof() || uint32 != m_uiL)break;
        in.read((char*)&uint32, sizeof(uint32));
        if(in.eof() || uint32 != m_uiK)break;
        int32_t int32;
        in.read((char*)&int32, sizeof(int32));
        m_eStatus = static_cast<Status>(int32);
        if(in.eof() || m_eStatus == UNKNOWN)break;
        return;
    } while(false);
    throw FormatException(TDBOW_LOG(
            "Not header format (Ver " << (int)TDBOW_CKPT_VERSION_MAJOR << '.'
            << (int)TDBOW_CKPT_VERSION_MINOR << ", DescL " << DescL
            << "), please check the file or input stream."));
}

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::ckptSaveNodes(
        const NodesPtr& _Nodes) noexcept(false) {
    assert(m_pReader == nullptr);
    if(m_strTag.empty())return;
    assert(m_pCheckPoint != nullptr);
    auto& logger = _getCheckPoint();
    auto size = static_cast<uint32_t>(_Nodes -> size() - 1);
    logger.write((char*)&size, sizeof(size));
    for(size_t i = 0; i < size; i++) {
        const auto& node = _Nodes -> at(i + 1);
        logger.write((char*)&node.id, sizeof(node.id));
        logger.write((char*)&node.parent, sizeof(node.parent));
        logger.write((char*)node.descriptor.data(), TDBOW_DESCRIPTOR_SIZE);
    }
}

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::ckptLoadNodes(
        NodesPtr& _Nodes) noexcept(false) {
    assert(m_pCheckPoint == nullptr);
    if(m_strTag.empty())return;
    assert(m_pReader != nullptr);
    auto& in = *m_pReader;
    do {
        uint32_t size;
        in.read((char*)&size, sizeof(size));
        if(in.eof())break;
        _Nodes -> resize(size + 1);
        _Nodes -> shrink_to_fit();
        for(size_t i = 0; i < size; i++) {
            auto& node = _Nodes -> at(i + 1);
            in.read((char*)&node.id, sizeof(node.id));
            if(in.eof())break;
            in.read((char*)&node.parent, sizeof(node.parent));
            if(in.eof())break;
            in.read((char*)node.descriptor.data(), TDBOW_DESCRIPTOR_SIZE);
            if(in.eof())break;
            (*_Nodes)[node.parent].children.emplace_back(node.id);
        }
        return;
    } while(false);
    throw FormatException(TDBOW_LOG(
            "Not header format (Ver " << (int)TDBOW_CKPT_VERSION_MAJOR << '.'
            << (int)TDBOW_CKPT_VERSION_MINOR << ", DescL " << DescL
            << "), please check the file or input stream."));
}

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::ckptSaveTasks(
        const std::queue<Task>& _Tasks) noexcept(false) {
    assert(m_pReader == nullptr);
    if(m_strTag.empty())return;
    assert(m_pCheckPoint != nullptr);
    auto& logger = _getCheckPoint();
    uint64_t size = _Tasks.size();
    logger.write((char*)&size, sizeof(size));
    auto tasks = _Tasks;
    while(!tasks.empty()) {
        const auto& task = tasks.front();
        uint32_t parentId = std::get<0>(task);
        uint32_t level = std::get<2>(task);
        const auto& indices = std::get<3>(task);
        logger.write((char*)&parentId, sizeof(parentId));
        logger.write((char*)&level, sizeof(level));
        size = indices.size();
        logger.write((char*)&size, sizeof(size));
        // Since standard type is aligned, so we can save/load in block
        logger.write((char*)indices.data(), sizeof(size_t) * size);
        tasks.pop();
    }
}

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::ckptLoadTasks(
        std::queue<Task>& _Tasks, const std::vector<DescriptorConstPtr>& _Dataset) noexcept(false) {
    assert(m_pCheckPoint == nullptr);
    if(m_strTag.empty())return;
    assert(m_pReader != nullptr);
    auto& in = *m_pReader;
    do {
        uint64_t size;
        in.read((char*)&size, sizeof(size));
        if(in.eof())break;
        for(size_t i = 0; i < size; i++) {
            uint32_t parentId, level;
            in.read((char*)&parentId, sizeof(parentId));
            if(in.eof())break;
            in.read((char*)&level, sizeof(level));
            if(in.eof())break;
            uint64_t scale;
            in.read((char*)&scale, sizeof(scale));
            if(in.eof())break;
            std::vector<size_t> indices(scale);
            in.read((char*)indices.data(), sizeof(size_t) * scale);
            if(in.eof())break;
#ifdef FOUND_OPENMP
            std::vector<DescriptorConstPtr> data(scale);
            #pragma omp parallel for
            for(size_t j = 0; j < indices.size(); j++) {
                data[j] = _Dataset[indices[j]];
            }
#else
            std::vector<DescriptorConstPtr> data(0);
            data.reserve(scale);
            for(const auto& idx : indices) {
                data.emplace_back(_Dataset[idx]);
            }
#endif
            _Tasks.push(std::make_tuple(parentId, data, level, indices));
        }
        return;
    } while(false);
    throw FormatException(TDBOW_LOG(
            "Not header format (Ver " << (int)TDBOW_CKPT_VERSION_MAJOR << '.'
            << (int)TDBOW_CKPT_VERSION_MINOR << ", DescL " << DescL
            << "), please check the file or input stream."));
}

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::ckptSaveIter(
        const size_t& _IterNum, const distance_type& _Loss,
        const std::vector<size_t>& _Belong) noexcept(false) {
    assert(m_pReader == nullptr);
    if(m_strTag.empty())return;
    assert(m_pCheckPoint == nullptr);
    assert(_IterNum != 0);
    const auto stableFile = getCkptFile();
    const auto tmpFile = _getTmpCkptFile(m_strTag);
    using boost::filesystem::copy_option;
    copy_file(stableFile, tmpFile, copy_option::overwrite_if_exists);
    m_pCheckPoint.reset(new std::ofstream(
            tmpFile.native(), std::ios::in|std::ios::binary));
    auto& logger = _getCheckPoint();
    if(!logger) {
        throw FileNotOpenException(TDBOW_LOG(tmpFile.native()));
    }
    logger.seekp(27, std::ios::beg);
    auto status = static_cast<int32_t>(m_eStatus);
    logger.write((char*)&status, sizeof(status));
    if(_IterNum != 1) {
        // Erase the previous iteration's checkpoint
        // This assume size_t and double_t keep 64 bits
        size_t len = 8 * (_Belong.size() + 2);
        logger.seekp(-len, std::ios::end);
    } else {
        logger.seekp(0, std::ios::end);
    }
    logger.write((char*)&_IterNum, sizeof(_IterNum));
    double_t loss = _Loss;
    logger.write((char*)&loss, sizeof(loss));
    // since std::vector<size_t> is already aligned, S/L in block
    logger.write((char*)_Belong.data(), sizeof(size_t) * _Belong.size());
}

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::ckptLoadIter(
        size_t& _IterNum, distance_type& _Loss, const size_t& _Len,
        std::vector<size_t>& _Belong) noexcept(false) {
    assert(m_pCheckPoint == nullptr);
    if(m_strTag.empty())return;
    assert(m_pReader != nullptr);
    auto& in = *m_pReader;
    do {
        in.read((char*)&_IterNum, sizeof(_IterNum));
        if(in.eof())break;
        double_t loss;
        in.read((char*)&loss, sizeof(loss));
        if(in.eof())break;
        _Loss = loss;
        // since std::vector<size_t> is already aligned, S/L in block
        _Belong.resize(_Len);
        _Belong.shrink_to_fit();
        in.read((char*)_Belong.data(), sizeof(size_t) * _Belong.size());
        if(in.eof())break;
        return;
    } while(false);
    throw FormatException(TDBOW_LOG(
            "Not header format (Ver " << (int)TDBOW_CKPT_VERSION_MAJOR << '.'
            << (int)TDBOW_CKPT_VERSION_MINOR << ", DescL " << DescL
            << "), please check the file or input stream."));
}

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::ckptLoaded() noexcept(false) {
    if(!m_pReader)return;
    m_pReader -> close();
    m_pReader.reset();
}

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::ckptDone() noexcept(false) {
    if(m_pCheckPoint) {
        m_pCheckPoint -> close();
        m_pCheckPoint.reset();
    }
    if(m_strTag.empty())return;
    const auto stable = getCkptFile();
    const auto temp = _getTmpCkptFile(m_strTag);
    if(m_eStatus != FINISHED) {
        using boost::filesystem::copy_option;
        copy_file(temp, stable, copy_option::overwrite_if_exists);
        output();
    } else if(exists(stable)) {
        output("Creation completed, remove checkpoint.\n");
        remove(stable);
    }
    remove(temp);
}

template <typename TScalar, size_t DescL>
void TemplatedCheckPoint<TScalar, DescL>::output(
        const std::string& _Content, const Level _Level) {
    if(m_strTag.empty() || _Level < m_eLevel)return;
    _getFileLogger() << _Content;
    _getFileLogger().flush();
    if(m_bOutputConsole) {
        if(_Level < ERROR) {
            std::cout << _Content;
            std::cout.flush();
        } else {
            std::cerr << _Content;
        }
    }
}

template <typename TScalar, size_t DescL>
std::string TemplatedCheckPoint<TScalar, DescL>::output(const Level _Level) {
    auto& ss = str();
    if(m_strTag.empty()) {
        ss.clear();
        ss.str("");
        return "";
    }
    const auto content = ss.str();
    if(content.empty())return content;
    ss.clear();
    ss.str("");
    output(content, _Level);
    return content;
}

/* ********************************************************************************
 *                              FUNCTIONAL METHODS                                *
 ******************************************************************************** */

/* ********************************************************************************
 *                                INNER METHODS                                   *
 ******************************************************************************** */

template <typename TScalar, size_t DescL>
bool TemplatedCheckPoint<TScalar, DescL>::_ckptCheckFile() noexcept(false) {
    assert(m_pReader != nullptr);
    if(m_eStatus == UNKNOWN)return false;
    auto& in = *m_pReader;
    constexpr size_t HEADER_LEN = 31;
    assert(in.tellg() == HEADER_LEN);
    uint32_t uint32;
    uint64_t uint64;
    switch(m_eStatus) {
        default:
            output(TDBOW_LOG("This sentence should not be executed "
                    "in this version. (" << (int)TDBOW_CKPT_VERSION_MAJOR << '.'
                    << (int)TDBOW_CKPT_VERSION_MINOR << ")"), FATAL);
            assert(false);
            break;

        case LOADING:
            in.seekg(0, std::ios::end);
            return in.tellg() == HEADER_LEN;

        case CLUSTER_BEG:
        case CLUSTER_ITER:
        case CLUSTER_END:
            in.read((char*)&uint32, sizeof(uint32));
            if(in.eof())break;
            const auto nodesLen = (DescL * sizeof(TScalar) + 8) * uint32;
            if(m_eStatus == CLUSTER_END) {
                in.seekg(0, std::ios::end);
                const auto pos = static_cast<size_t>(in.tellg());
                const size_t expect = HEADER_LEN + nodesLen + 4;
                str() << "Found " << pos << " bytes, expect "
                      << expect << " bytes." << std::endl;
                output(DEBUG);
                return pos == expect;
            }
            in.seekg(nodesLen, std::ios::cur);
            in.read((char*)&uint64, sizeof(uint64));
            if(in.eof())break;
            const auto tasksLen = 16 * uint64 + m_ulScale * 8;
            if(m_eStatus == CLUSTER_BEG) {
                in.seekg(0, std::ios::end);
                const auto pos = static_cast<size_t>(in.tellg());
                const size_t expect = HEADER_LEN + 12 + nodesLen + tasksLen;
                str() << "Found " << pos << " bytes, expect "
                      << expect << " bytes." << std::endl;
                output(DEBUG);
                return pos == expect;
            }
            in.seekg(8, std::ios::cur);
            in.read((char*)&uint64, sizeof(uint64));
            if(in.eof())break;
            in.seekg(0, std::ios::end);
            const auto pos = static_cast<size_t>(in.tellg());
            const size_t expect = HEADER_LEN + 28 + nodesLen + tasksLen + 8 * uint64;
            str() << "Found " << pos << " bytes, expect "
                  << expect << " bytes." << std::endl;
            output(DEBUG);
            return pos == expect;
    }
    return false;
}

template <typename TScalar, size_t DescL>
typename TemplatedCheckPoint<TScalar, DescL>::path
TemplatedCheckPoint<TScalar, DescL>::_getLogFile(
        const std::string& _Log) noexcept(false) {
    const auto ret = _getLoggerDir() / (_Log + ".txt");
    if(!exists(ret.parent_path())) {
        create_directories(ret.parent_path());
    }
    return ret;
}

template <typename TScalar, size_t DescL>
typename TemplatedCheckPoint<TScalar, DescL>::path
TemplatedCheckPoint<TScalar, DescL>::_getCkptFile(
        const std::string& _Log) noexcept(false) {
    const auto ret = _getLoggerDir() / ("." + _Log + ".ckpt");
    if(!exists(ret.parent_path())) {
        create_directories(ret.parent_path());
    }
    return ret;
}

template <typename TScalar, size_t DescL>
typename TemplatedCheckPoint<TScalar, DescL>::path
TemplatedCheckPoint<TScalar, DescL>::_getTmpCkptFile(
        const std::string& _Log) noexcept(false) {
    const auto ret = _getLoggerDir() / ("." + _Log + ".tmp.ckpt");
    if(!exists(ret.parent_path())) {
        create_directories(ret.parent_path());
    }
    return ret;
}

template <typename TScalar, size_t DescL>
typename TemplatedCheckPoint<TScalar, DescL>::path
TemplatedCheckPoint<TScalar, DescL>::_getLoggerDir() noexcept(false) {
    path workspace;
#ifdef PKG_DIR
    workspace = path(PKG_DIR)/".logger";
#else
    auto homeDir = getenv("HOME");
    if(homeDir != nullptr) {
        workspace = path(homeDir)/".tdbow"/".logger";
    } else {
        workspace = boost::filesystem::temp_directory_path()/".tdbow"/".logger";
    }
#endif
    if(!exists(workspace)) {
        create_directories(workspace);
    }
    if(!is_directory(workspace)) {
        throw IOException(TDBOW_LOG(
                workspace.native() << " exists, but not a directory."));
    }
    return workspace;
}

template <typename TScalar, size_t DescL>
std::string TemplatedCheckPoint<TScalar, DescL>::format(
        const std::string& _Log) noexcept {
    std::string tag = _Log;
    boost::replace_if(tag, boost::is_any_of(",，、。:：？?!！@#$%^&*"), ' ');
    boost::trim_if(tag, boost::is_any_of(" ."));
    boost::replace_if(tag, boost::is_any_of(".- "), '_');
    boost::trim_if(tag, boost::is_any_of("_"));
    return tag;
}

} // namespace TDBoW

#endif //__ROCKAUTO_TDBOW_TEMPLATED_CHECK_POINT_HPP__
