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
 * File: QueryResults.h
 * Date: March, November 2011
 * Author: Dorian Galvez-Lopez
 * Description: structure to store results of database queries
 * License: see the LICENSE.txt file
 *
 */

#ifndef __ROCKAUTO_TDBOW_QUERY_RESULTS__
#define __ROCKAUTO_TDBOW_QUERY_RESULTS__

#include "BowVector.h"

namespace TDBoW {

/// Id of entries of the database
typedef unsigned int EntryId;

/// Single result of a query
class Result {
public:

    /// Entry id
    EntryId Id{};

    /// Score obtained
    double Score{};

    /**
     * Empty constructors
     */
    Result() = default;

    /**
     * @brief Creates a result with the given data
     * @param _id    entry id
     * @param _score score
     */
    Result(EntryId _id, WordValue _score): Id(_id), Score(_score) {}

    virtual ~Result() = default;

    /**
     * @brief  Compares the scores of two results
     * @return {@code true} if this.score < r.score
     */
    bool operator<(const Result& _R) const {
        return this->Score < _R.Score;
    }

    /**
     * @brief  Compares the scores of two results
     * @return {@code true} if this.score <= r.score
     */
    bool operator<=(const Result& _R) const {
        return this->Score <= _R.Score;
    }

    /**
     * @brief  Compares the scores of two results
     * @return {@code true} if this.score > r.score
     */
    bool operator>(const Result& _R) const {
        return this->Score > _R.Score;
    }

    /**
     * @brief  Compares the scores of two results
     * @return {@code true} if this.score > r.score
     */
    bool operator>=(const Result& _R) const {
        return this->Score >= _R.Score;
    }

    /**
     * @brief  Compares the entry id of the result
     * @return {@code true} if this.id == id
     */
    bool operator==(const EntryId _Id) const {
        return this->Id == _Id;
    }

    /**
     * @brief Compares the score of this entry with a given one
     * @param s score to compare with
     * @return {@code true} if this score < s
     */
    bool operator<(const double _S) const {
        return this->Score < _S;
    }

    /**
     * @brief Compares the score of this entry with a given one
     * @param _S score to compare with
     * @return {@code true} if this score <= s
     */
    bool operator<=(const double _S) const {
        return this->Score <= _S;
    }

    /**
     * @brief Compares the score of this entry with a given one
     * @param _S score to compare with
     * @return {@code true} if this score > s
     */
    bool operator>(const double _S) const {
        return this->Score > _S;
    }

    /**
     * @brief Compares the score of this entry with a given one
     * @param s score to compare with
     * @return {@code true} if this score >= s
     */
    bool operator>=(const double _S) const {
        return this->Score >= _S;
    }

    /**
     * @brief Prints a string version of the result
     * @param _Out output stream
     * @param _Ret Result to print
     */
    friend std::ostream & operator<<(std::ostream& _Out, const Result& _Ret);
};

/// Multiple results from a query
class QueryResults: public std::vector<Result> {
public:

    /**
     * @brief Prints a string version of the results
     * @param os ostream
     * @param ret QueryResults to print
     */
    friend std::ostream& operator<<(std::ostream& os, const QueryResults& ret );

    /**
     * @brief Saves a matlab file with the results
     * @param filename
     */
    void saveM(const std::string &filename) const;

};

// --------------------------------------------------------------------------

} // namespace TemplatedBoW
  
#endif

