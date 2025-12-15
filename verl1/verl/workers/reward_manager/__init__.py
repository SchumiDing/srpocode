# Copyright 2024 PRIME team and/or its affiliates
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

from .registry import get_reward_manager_cls, register  # noqa: I001
from .batch import BatchRewardManager
from .dapo import DAPORewardManager
from .naive import NaiveRewardManager
from .prime import PrimeRewardManager
from .grpo import GRPORewardManager
from .srpo import SRPORewardManager

# Import additional reward managers (they will be registered via @register decorator)
try:
    from . import srpo2  # noqa: F401
    from . import srpo3  # noqa: F401
    from . import srpo4  # noqa: F401
    from . import rflux  # noqa: F401
except ImportError:
    pass  # Optional imports

# Note(haibin.lin): no need to include all reward managers here in case of complicated dependencies
__all__ = [
    "BatchRewardManager",
    "DAPORewardManager",
    "NaiveRewardManager",
    "PrimeRewardManager",
    "GRPORewardManager",
    "SRPORewardManager",
    "register",
    "get_reward_manager_cls",
]

# Import experimental reward managers to ensure they are registered
try:
    from verl.experimental.reward.reward_loop.limited import RateLimitedRewardLoopManager  # noqa: F401

    __all__.append("RateLimitedRewardLoopManager")
except ImportError:
    pass  # Optional dependency, may not be available
