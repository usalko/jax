# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def _version_as_tuple(version_str):
  return tuple(int(i) for i in version_str.split(".") if i.isdigit())

__version__ = "0.3.5"
__version_info__ = _version_as_tuple(__version__)

_minimum_jaxlib_version = "0.3.2"
_minimum_jaxlib_version_info = _version_as_tuple(_minimum_jaxlib_version)
