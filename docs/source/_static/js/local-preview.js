// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(() => {
  const localHosts = new Set(["localhost", "127.0.0.1", "0.0.0.0", "::1"]);
  if (!localHosts.has(window.location.hostname)) {
    return;
  }

  const currentScript = document.currentScript;
  if (currentScript && typeof DOCUMENTATION_OPTIONS !== "undefined") {
    DOCUMENTATION_OPTIONS.theme_switcher_json_url = new URL(
      "../../versions1.json",
      currentScript.src,
    ).href;
  }

  const localPreviewStyles = document.createElement("style");
  localPreviewStyles.dataset.deepResearcherLocalPreview = "true";
  localPreviewStyles.textContent = `
    #onetrust-consent-sdk,
    #onetrust-banner-sdk,
    #onetrust-pc-sdk,
    .onetrust-pc-dark-filter {
      display: none !important;
      pointer-events: none !important;
    }

    html,
    body {
      overflow: auto !important;
    }
  `;
  document.head.append(localPreviewStyles);
})();
