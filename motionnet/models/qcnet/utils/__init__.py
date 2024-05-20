# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from motionnet.models.qcnet.utils.geometry import angle_between_2d_vectors
from motionnet.models.qcnet.utils.geometry import angle_between_3d_vectors
from motionnet.models.qcnet.utils.geometry import side_to_directed_lineseg
from motionnet.models.qcnet.utils.geometry import wrap_angle
from motionnet.models.qcnet.utils.graph import add_edges
from motionnet.models.qcnet.utils.graph import bipartite_dense_to_sparse
from motionnet.models.qcnet.utils.graph import complete_graph
from motionnet.models.qcnet.utils.graph import merge_edges
from motionnet.models.qcnet.utils.graph import unbatch
from motionnet.models.qcnet.utils.list import safe_list_index
from motionnet.models.qcnet.utils.weight_init import weight_init