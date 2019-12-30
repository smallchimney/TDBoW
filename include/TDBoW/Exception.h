/**************************************************************************
 * Copyright (c) 2019 Chimney Xu. All Rights Reserve.
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
   * File Name     : exception.h
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-12-27 18:10:47
   * Last Modified : smallchimney
   * Modified Time : 2019-12-29 14:39:34
************************************************************************* */
#ifndef __ROCKAUTO_TDBOW_EXCEPTION_H__
#define __ROCKAUTO_TDBOW_EXCEPTION_H__

#include <stdexcept>

namespace TDBoW {

#define __DEF_EX(NAME, BASE) \
class NAME: public BASE {\
public:\
    NAME () = delete;\
    explicit NAME (const std::string& _What) : BASE (_What) {}\
    explicit NAME (const char* _What) : BASE (_What) {}\
    ~NAME () override = default;\
    NAME(NAME&&) = default;\
    NAME& operator=(const NAME&) = default;\
};

__DEF_EX(Exception, std::runtime_error)
// Program design exceptions
__DEF_EX(LogicException, Exception)
__DEF_EX(NotInitailizedException, LogicException)
__DEF_EX(ParametersException, LogicException)
__DEF_EX(OutOfRangeException, LogicException)
__DEF_EX(MethodNotMatchException, LogicException)
// IO Exceptions
__DEF_EX(IOException, Exception)
__DEF_EX(FileNotExistException, IOException)
__DEF_EX(FileNotOpenException, IOException)
// Data Exceptions
__DEF_EX(DataException, Exception)
__DEF_EX(FormatException, DataException)
__DEF_EX(EmptyDataException, DataException)
__DEF_EX(NanDataException, DataException)

#undef __DEF_EX

}

#endif //__ROCKAUTO_TDBOW_EXCEPTION_H__
