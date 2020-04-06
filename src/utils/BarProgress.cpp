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
   * File Name     : progress.cpp
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-04-17 20:25:54
   * Last Modified : smallchimney
   * Modified Time : 2020-01-12 20:48:08
************************************************************************* */
#include <TDBoW/utils/BarProgress.h>

namespace TDBoW {

static unsigned int determineScreenWidth() {
    /* If there's a way to get the terminal size using POSIX
       tcgetattr(), somebody please tell me.  */
#ifndef TIOCGWINSZ
    return 0;
#else  /* TIOCGWINSZ */
    winsize wsz{};
    int fd = fileno (stderr);
    if(ioctl(fd, TIOCGWINSZ, &wsz) < 0)return 0;	/* most likely ENOTTY */
    return wsz.ws_col;
#endif /* TIOCGWINSZ */
}

#define APPEND_LITERAL(p, s) do {       \
  memcpy (p, s, sizeof (s) - 1);        \
  p += sizeof (s) - 1;                  \
} while (0)

#define DEFAULT_SCREEN_WIDTH 80
#define MINIMUM_SCREEN_WIDTH 45

static unsigned int l_uiScreenWidth = DEFAULT_SCREEN_WIDTH;

#ifdef SIGWINCH
static void handleSigwinch(int sig) {
    auto sw = determineScreenWidth();
    if(sw != 0 && sw >= MINIMUM_SCREEN_WIDTH)
        l_uiScreenWidth = sw;
    signal(SIGWINCH, handleSigwinch);
}
#endif

BarProgress::BarProgress(const size_t& _Total) : m_sStartStamp(std::chrono::system_clock::now()),
        m_bFinished(false), m_ulProgress(0), m_ulTotal(_Total), m_dLastEtaTime(0), m_lLastEtaValue(0) {
#ifdef SIGWINCH
    handleSigwinch(SIGWINCH);
#endif
    m_uiWidth = l_uiScreenWidth - 1;
    m_pBuffer = (char*)malloc(sizeof(char) * l_uiScreenWidth + 100);
    m_pBuffer[m_uiWidth] = '\0';
}

void BarProgress::update(const size_t& _ProcessCount, const double& _TimeStamp) {
    assert(!m_bFinished);
    m_ulProgress += _ProcessCount;
    updateHist(_ProcessCount, _TimeStamp);
    static auto previousStamp = DBL_MIN;

    bool forceScreenFlush = false;
    if(l_uiScreenWidth - 1 != m_uiWidth) {
        m_uiWidth = l_uiScreenWidth - 1;
        free(m_pBuffer);
        m_pBuffer = (char*)malloc(sizeof(char) * l_uiScreenWidth + 100);
        forceScreenFlush = true;
    }
    if(m_ulProgress == m_ulTotal) {
        forceScreenFlush = true;
    }

    /* Don't update more often than five times per second. */
    if(_TimeStamp - previousStamp < 200 && !forceScreenFlush)return;
    updateImage(_TimeStamp);

    printf("\r%s", m_pBuffer);
    fflush(stdout);
    previousStamp = _TimeStamp;
    if(m_ulProgress == m_ulTotal)finished();
}

void BarProgress::updateHist(const size_t& _ProcessCount, const double& _TimeStamp) {
    double duration = _TimeStamp - m_sHist.recent_start;
    m_sHist.recent_counts += _ProcessCount;
    if(duration < SPEED_SAMPLE_MIN)return;

    /* Store "recent" bytes and download time to history ring at the
       position POS.  */

    /* To correctly maintain the totals, first invalidate existing data
       (least recent in time) at this position. */

    m_sHist.total_time   -= m_sHist.times[m_sHist.pos];
    m_sHist.total_counts -= m_sHist.counts[m_sHist.pos];

    /* Now store the new data and update the totals. */
    m_sHist.times[m_sHist.pos] = duration;
    m_sHist.counts[m_sHist.pos] = m_sHist.recent_counts;
    m_sHist.total_time   += duration;
    m_sHist.total_counts += m_sHist.recent_counts;
    m_sHist.recent_start = _TimeStamp;
    m_sHist.recent_counts = 0;

    /* Advance the current ring position. */
    if(++m_sHist.pos == SPEED_HISTORY_SIZE)m_sHist.pos = 0;
}

void BarProgress::updateImage(const double& _TimeStamp) {
    char* p = m_pBuffer;

    std::string processed = boost::lexical_cast<std::string>(m_ulProgress);

    /* The progress bar should look like this:
       xx% [=======>             ] nn,nnn 12.34K/s ETA 00:00

       Calculate the geometry.  The idea is to assign as much room as
       possible to the progress bar.  The other idea is to never let
       things "jitter", i.e. pad elements that vary in size so that
       their variance does not affect the placement of other elements.
       It would be especially bad for the progress bar to be resized
       randomly.

       "xx% " or "100%"  - percentage               - 4 chars
       "[]"              - progress bar decorations - 2 chars
       " nnn,nnn,nnn"    - downloaded bytes         - 12 chars or very rarely more
       " 12.56s"         - average span per count   - 7 chars
       " ETA xx:xx:xx"   - ETA                      - 13 chars

       "=====>..."       - progress bar             - the rest
    */
    size_t countLen = 1 + std::max((size_t)11, processed.length());
    size_t progressLen = m_uiWidth - (4 + 2 + countLen + 7 + 13);
    if(progressLen < 5)progressLen = 5;

    /* "xx% " */
    if(m_ulTotal) {
        assert(m_ulProgress <= m_ulTotal);
        int percentage = static_cast<int>(100.0 * m_ulProgress / m_ulTotal);
        if(percentage < 100)sprintf(p, "%2d%% ", percentage);
        else strcpy (p, "100%");
        p += 4;
    } else APPEND_LITERAL(p, "    ");

    /* The progress bar: "[====>      ]" or "[++==>      ]". */
    if (progressLen && m_ulTotal) {
        /* Size of the initial portion. */
        int barLen = static_cast<int>((double)m_ulProgress / m_ulTotal * progressLen);
        auto progressSize = static_cast<int>(progressLen);

        assert(barLen <= progressSize);

        *p++ = '[';
        const char* begin = p;

        /* Print the bar portion with '=' and one '>'.  */
        for(int i = 0; i < barLen - 1; i++)*p++ = '=';
        if(barLen)*p++ = '>';

        while(p - begin < progressSize)*p++ = ' ';
        *p++ = ']';
    }

    /* " 234,567,890" */
    sprintf(p, " %-11s", processed.c_str());
    p += strlen(p);

    /* " 12.56s" */
    if(m_sHist.total_time != 0. && m_sHist.total_counts) {

        /* Calculate the speed using the history ring and
           recent data that hasn't made it to the ring yet.  */
        auto recentCounts = m_sHist.total_counts + m_sHist.recent_counts;
        auto recentDuration = m_sHist.total_time + _TimeStamp - m_sHist.recent_start;
        if(recentCounts == 0)goto no_speed;
        double speed = recentDuration / (recentCounts * 1000);
        if(speed >= 1000)goto no_speed;
        sprintf(p, " %5.2fs", speed);
        p += strlen(p);
    }
    else {
        no_speed:
        APPEND_LITERAL (p, " --.--s");
    }

    /* " ETA xx:xx:xx"; wait for three seconds before displaying the ETA.
       That's because the ETA value needs a while to become
       reliable.  */
    if (m_ulTotal && _TimeStamp > 3000) {
        /* Don't change the value of ETA more than approximately once
           per second; doing so would cause flashing without providing
           any value to the user. */
        if(m_ulTotal == m_ulProgress ||
           m_lLastEtaValue == 0 ||
           _TimeStamp - m_dLastEtaTime >= 900) {
            /* Calculate ETA using the average download speed to predict
               the future speed.  If you want to use a speed averaged
               over a more recent period, replace _TimeStamp with
               m_sHist.totalTimes and m_ulProgress with m_sHist.totalCounts.
               I found that doing that results in a very jerky and
               ultimately unreliable ETA.  */
            auto remainCounts = m_ulTotal - m_ulProgress;
            m_lLastEtaValue = (long)(_TimeStamp / 1000 * remainCounts / m_ulProgress);
            m_dLastEtaTime = _TimeStamp;
        }

        long eta = m_lLastEtaValue, etaHour, etaMin, etaSec;
        etaHour = eta / 3600, eta %= 3600;
        etaMin  = eta / 60,   eta %= 60;
        etaSec  = eta;

        if(etaHour > 99)goto no_eta;
        if(etaHour) {
            /* Hours printed with one digit: pad with one space. */
            if(etaHour < 10)APPEND_LITERAL (p, " ");
            sprintf (p, " ETA %ld:%02ld:%02ld", etaHour, etaMin, etaSec);
        } else {
            /* Hours not printed: pad with three spaces. */
            APPEND_LITERAL (p, "   ");
            sprintf (p, " ETA %02ld:%02ld", etaMin, etaSec);
        }
        p += strlen (p);
    } else if (m_ulTotal) {
        no_eta:
        APPEND_LITERAL(p, "             ");
    }

    assert (p - m_pBuffer <= m_uiWidth);

    while(p < m_pBuffer + m_uiWidth)*p++ = ' ';
    *p = '\0';
}

}  // namespace TDBoW
