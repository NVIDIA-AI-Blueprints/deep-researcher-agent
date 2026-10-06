# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Retriever REST backend for the Deep Researcher Agent Knowledge Layer."""

from .adapter import NemoRetrieverIngestor
from .adapter import NemoRetrieverRetriever

__all__ = ["NemoRetrieverIngestor", "NemoRetrieverRetriever"]
