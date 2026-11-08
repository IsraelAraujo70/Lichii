// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA NGX's Vulkan API as loaded from libnvsdk_ngx.so (the DLSS SDK's loader, built as its own
// library next to the executable). Only for the NGX sources: it pulls in the DLSS SDK's headers.

#pragma once

#include "video_core/vulkan_common/vulkan_wrapper.h"

#include <nvsdk_ngx_vk.h>

namespace Vulkan::Streamline {

struct NgxApi {
    decltype(&NVSDK_NGX_VULKAN_RequiredExtensions) required_extensions{};
    decltype(&NVSDK_NGX_VULKAN_Init_with_ProjectID) init{};
    decltype(&NVSDK_NGX_VULKAN_Shutdown1) shutdown{};
    decltype(&NVSDK_NGX_VULKAN_GetCapabilityParameters) get_capability_parameters{};
    decltype(&NVSDK_NGX_VULKAN_AllocateParameters) allocate_parameters{};
    decltype(&NVSDK_NGX_VULKAN_DestroyParameters) destroy_parameters{};
    decltype(&NVSDK_NGX_VULKAN_CreateFeature1) create_feature{};
    decltype(&NVSDK_NGX_VULKAN_EvaluateFeature_C) evaluate_feature{};
    decltype(&NVSDK_NGX_VULKAN_ReleaseFeature) release_feature{};
};

/// The loaded functions; only valid while IsDlssAvailable().
[[nodiscard]] const NgxApi& GetNgx();

/// The device NGX was initialised with; only valid while IsDlssAvailable().
[[nodiscard]] VkDevice NgxDevice();

} // namespace Vulkan::Streamline
