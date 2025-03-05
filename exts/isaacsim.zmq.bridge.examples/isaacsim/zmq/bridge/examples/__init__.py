# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

EXT_NAME = "isaacsim.zmq.bridge.examples"

from .core.proto_util import register_proto_modules

register_proto_modules()

from .core.annotators import *
from .core.client import *
from .extension import *
