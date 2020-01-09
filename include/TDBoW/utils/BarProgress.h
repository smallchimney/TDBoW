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
/* Download progress.
   Copyright (C) 2001, 2002 Free Software Foundation, Inc.

This file is part of GNU Wget.

GNU Wget is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

GNU Wget is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Wget; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

In addition, as a special exception, the Free Software Foundation
gives permission to link the code of its release of Wget with the
OpenSSL project's "OpenSSL" library (or with modified versions of it
that use the same license as the "OpenSSL" library), and distribute
the linked executables.  You must obey the GNU General Public License
in all respects for all of the code used other than "OpenSSL".  If you
modify this file, you may extend this exception to your version of the
file, but you are not obligated to do so.  If you do not wish to do
so, delete this exception statement from your version.  */
/* *************************************************************************
   * File Name     : BarProgress.h
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-04-17 20:25:54
   * Last Modified : smallchimney
   * Modified Time : 2020-01-12 20:59:24
************************************************************************* */
#ifndef __CHIMNEY_UTILS_TDBOW_PROGRESS_H__
#define __CHIMNEY_UTILS_TDBOW_PROGRESS_H__

#include <cstddef>
#include <cfloat>
#include <string>
#include <cmath>
#include <chrono>
#include <memory>

#include <signal.h>
#include <sys/ioctl.h>
#include <boost/lexical_cast.hpp>

namespace TDBoW {

static std::string duration2string(const double& _DurationInSec) {
    static std::stringstream buffer;
    unsigned int day, hour, min;
    double sec;
    long long tmp;
    tmp = static_cast<long long>(_DurationInSec / 60);
    sec = _DurationInSec - tmp * 60;
    min = static_cast<unsigned int>(tmp % 60);
    tmp /= 60;
    hour = static_cast<unsigned int>(tmp % 24);
    day = static_cast<unsigned int>(tmp / 24);
    if(day)buffer << day << " days ";
    if(hour)buffer << hour << " hours ";
    if(min)buffer << min << " minutes ";
    buffer << sec << " seconds";
    std::string str = buffer.str();
    buffer.clear();
    buffer.str("");
    return str;
}

class BarProgress {
public:
    typedef std::shared_ptr<BarProgress> Ptr;
    typedef std::shared_ptr<BarProgress const> ConstPtr;

    explicit BarProgress(const size_t& _Total);
    virtual ~BarProgress() = default;
    BarProgress() = delete;

    void update(const size_t& _ProcessCount = 1) {
        update(_ProcessCount, std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now() - m_sStartStamp).count());
    }

    void finished() {
        if(m_bFinished)return;
        if(m_ulTotal - m_ulProgress) {
            update(m_ulTotal - m_ulProgress);
            return;
        }
        free(m_pBuffer);
        using namespace std::chrono;
        std::string timeStr = duration2string(duration_cast<seconds>(
                system_clock::now() - m_sStartStamp).count());
        printf("\nprocessing %s.\n", timeStr.c_str());
        m_bFinished = true;
    }

protected:

    /* Size of the speed history ring. */
    static const size_t SPEED_HISTORY_SIZE = 20;

    /* The minimum time length of a history sample.  By default, each
       sample is at least 150ms long, which means that, over the course of
       20 samples, "current" speed spans at least 3s into the
       past.  */
    static const size_t SPEED_SAMPLE_MIN = 150;

    struct Hist {
        int pos;
        double times[SPEED_HISTORY_SIZE];
        size_t counts[SPEED_HISTORY_SIZE];

        /* The sum of times and bytes respectively, maintained for
           efficiency. */
        double total_time;
        size_t total_counts;

        double recent_start;		/* timestamp of beginning of current
                                       position. */
        size_t recent_counts;		/* counts processed so far. */
    } m_sHist{};

private:
    std::chrono::system_clock::time_point m_sStartStamp;
    bool m_bFinished;
    unsigned int m_uiWidth;
    size_t m_ulProgress, m_ulTotal;
    char* m_pBuffer;

    double m_dLastEtaTime;		/* time of the last update to download
                                   speed and ETA, measured since the
                                   beginning of download. */
    long m_lLastEtaValue;

    void update(const size_t& _ProcessCount, const double& _TimeStamp);

    void updateHist(const size_t& _ProcessCount, const double& _TimeStamp);

    void updateImage(const double& _TimeStamp);

}; // class BarProgress

}  // namespace TDBoW

#endif //__CHIMNEY_UTILS_TDBOW_PROGRESS_H__
