// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS through NVIDIA NGX (Linux) in place of a game's DLSS evaluation. The parameters
// match what Streamline's DLSS plugin hands NGX for the Windows path (vk_streamline_dlss.cpp).

#include <array>
#include <cstdlib>
#include <string_view>

#include "common/logging.h"
#include "video_core/dlss_evaluation.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/ngx_api.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {
namespace {

constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkImageUsageFlags OUTPUT_USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;

int EnvInt(const char* name, int fallback) {
    const char* const value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

// The same ratio thresholds as the Streamline path.
NVSDK_NGX_PerfQuality_Value ModeFor(VkExtent2D render, VkExtent2D upscale) {
    const f32 ratio = static_cast<f32>(render.width) / static_cast<f32>(upscale.width);
    if (ratio >= 0.99f) {
        return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
    if (ratio >= 0.65f) {
        return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    }
    if (ratio >= 0.57f) {
        return NVSDK_NGX_PerfQuality_Value_Balanced;
    }
    if (ratio >= 0.49f) {
        return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    }
    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

// Preset K for every mode, as on Windows. EDEN_NX2_DLSS_PRESET=J|K|L|M picks another model,
// EDEN_NX2_DLSS_PRESET=default leaves the choice to NVIDIA.
u32 Preset() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_PRESET");
    if (value && value[0] >= 'J' && value[0] <= 'M') {
        return NVSDK_NGX_DLSS_Hint_Render_Preset_J + static_cast<u32>(value[0] - 'J');
    }
    if (value && std::string_view{value} == "default") {
        return NVSDK_NGX_DLSS_Hint_Render_Preset_Default;
    }
    return NVSDK_NGX_DLSS_Hint_Render_Preset_K;
}

VkImageMemoryBarrier Barrier(const StreamlineDlssUpscaler::Image& image, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.image,
        .subresourceRange = range,
    };
}

NVSDK_NGX_Resource_VK Resource(VkImageView view, VkImage image, VkImageSubresourceRange range,
                               VkFormat format, VkExtent2D size, bool read_write) {
    NVSDK_NGX_Resource_VK resource{};
    resource.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
    resource.Resource.ImageViewInfo.ImageView = view;
    resource.Resource.ImageViewInfo.Image = image;
    resource.Resource.ImageViewInfo.SubresourceRange = range;
    resource.Resource.ImageViewInfo.Format = format;
    resource.Resource.ImageViewInfo.Width = size.width;
    resource.Resource.ImageViewInfo.Height = size.height;
    resource.ReadWrite = read_write;
    return resource;
}

NVSDK_NGX_Resource_VK Resource(const StreamlineDlssUpscaler::Image& image, VkExtent2D size) {
    return Resource(image.view, image.image, image.range, image.format, size, false);
}

struct CreateOptions {
    NVSDK_NGX_PerfQuality_Value mode;
    u32 preset;
    int flags;
};

} // Anonymous namespace

StreamlineDlssUpscaler::StreamlineDlssUpscaler(const Device& device_, Scheduler& scheduler_,
                                               MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    if (!Streamline::IsDlssAvailable()) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    if (const NVSDK_NGX_Result result = Streamline::GetNgx().allocate_parameters(&parameters);
        NVSDK_NGX_FAILED(result) || !parameters) {
        LOG_ERROR(Render_Vulkan, "NVIDIA NGX could not allocate DLSS parameters (result {:#x})",
                  static_cast<u32>(result));
        parameters = nullptr;
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    point_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
}

StreamlineDlssUpscaler::~StreamlineDlssUpscaler() {
    // Not scheduler.Finish(): this runs when the renderer goes away, after the game's channels are
    // released, and a submit would touch their state. Nothing more gets recorded by then.
    void(device.GetLogical().WaitIdle());
    if (Streamline::IsDlssAvailable()) {
        const Streamline::NgxApi& api = Streamline::GetNgx();
        if (feature) {
            api.release_feature(feature);
        }
        if (parameters) {
            api.destroy_parameters(parameters);
        }
    }
}

void StreamlineDlssUpscaler::CreateOutput(VkExtent2D upscale_size) {
    if (output_image) {
        scheduler.Finish(); // the old output may still be in use
    }
    output_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .extent = {upscale_size.width, upscale_size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = OUTPUT_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    output_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *output_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               VkImageMemoryBarrier{
                                   .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .pNext = nullptr,
                                   .srcAccessMask = 0,
                                   .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange{
                                       .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .baseMipLevel = 0,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = 1,
                                   },
                               });
    });
}

bool StreamlineDlssUpscaler::Upscale(const Inputs& in) {
    if (failed) {
        return false;
    }
    const bool extent_changed = !output_image || in.render_size.width != render_extent.width ||
                                in.render_size.height != render_extent.height ||
                                in.upscale_size.width != upscale_extent.width ||
                                in.upscale_size.height != upscale_extent.height;
    const bool options_changed = extent_changed || in.hdr != hdr_input;
    const bool reset = in.reset || options_changed;
    if (options_changed && feature) {
        // The DLSS feature is created for one size and set of flags: wait until no frame uses
        // the old one.
        scheduler.Finish();
        Streamline::GetNgx().release_feature(feature);
        feature = nullptr;
    }
    if (extent_changed) {
        CreateOutput(in.upscale_size);
        render_extent = in.render_size;
        upscale_extent = in.upscale_size;
    }
    hdr_input = in.hdr;

    // Streamline asks NGX for auto exposure whenever no exposure texture is tagged, which the
    // Windows path never does.
    int flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    if (in.hdr && EnvInt("EDEN_NX2_DLSS_HDR", 1) != 0) {
        flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    }
    if (in.depth_inverted) {
        flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    }
    const CreateOptions options{
        .mode = ModeFor(in.render_size, in.upscale_size),
        .preset = Preset(),
        .flags = flags,
    };
    if (options_changed) {
        LOG_INFO(Render_Vulkan, "NVIDIA DLSS (NGX): {}x{} -> {}x{}, mode {}, preset {}, {}",
                 in.render_size.width, in.render_size.height, in.upscale_size.width,
                 in.upscale_size.height, static_cast<int>(options.mode), options.preset,
                 (flags & NVSDK_NGX_DLSS_Feature_Flags_IsHDR) != 0 ? "HDR" : "LDR");
    }

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, reset, options, output_image_handle = *output_image,
                      output_view_handle = *output_view](vk::CommandBuffer cmdbuf) {
        const Streamline::NgxApi& api = Streamline::GetNgx();
        // The caller draws the output this frame even when DLSS writes nothing to it: clear it
        // then, rather than show undefined contents (FSR 3.1 takes over from the next frame).
        const auto clear_output = [&] {
            const VkImageSubresourceRange range{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            const auto barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = src_access,
                    .dstAccessMask = dst_access,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = output_image_handle,
                    .subresourceRange = range,
                };
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                   barrier(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                           VK_ACCESS_TRANSFER_WRITE_BIT));
            cmdbuf.ClearColorImage(output_image_handle, VK_IMAGE_LAYOUT_GENERAL,
                                   VkClearColorValue{}, range);
            cmdbuf.PipelineBarrier(
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
        };

        constexpr VkAccessFlags WRITES =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        // The inputs come from the game's draws and dispatches and stay in GENERAL otherwise;
        // DLSS reads them as sampled images.
        const std::array to_read{
            Barrier(in.color, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.depth, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.motion_vectors, WRITES, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        };
        const std::array to_general{
            Barrier(in.color, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.depth, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.motion_vectors, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | WRITES,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        const VkMemoryBarrier output_written{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };

        NVSDK_NGX_Resource_VK color = Resource(in.color, in.render_size);
        NVSDK_NGX_Resource_VK depth = Resource(in.depth, in.render_size);
        NVSDK_NGX_Resource_VK motion = Resource(in.motion_vectors, in.render_size);
        NVSDK_NGX_Resource_VK output =
            Resource(output_view_handle, output_image_handle,
                     VkImageSubresourceRange{
                         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                         .baseMipLevel = 0,
                         .levelCount = 1,
                         .baseArrayLayer = 0,
                         .layerCount = 1,
                     },
                     OUTPUT_FORMAT, in.upscale_size, true);

        NVSDK_NGX_Result api_result = NVSDK_NGX_Result_Success;
        const auto result = VideoCommon::RecordDlssEvaluation(
            [&] {
                if (feature) {
                    return true;
                }
                NVSDK_NGX_Parameter* const p = parameters;
                p->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u);
                p->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
                p->Set(NVSDK_NGX_Parameter_Width, in.render_size.width);
                p->Set(NVSDK_NGX_Parameter_Height, in.render_size.height);
                p->Set(NVSDK_NGX_Parameter_OutWidth, in.upscale_size.width);
                p->Set(NVSDK_NGX_Parameter_OutHeight, in.upscale_size.height);
                p->Set(NVSDK_NGX_Parameter_PerfQualityValue, static_cast<int>(options.mode));
                p->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, options.flags);
                p->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
                p->Set(NVSDK_NGX_Parameter_FreeMemOnReleaseFeature, 1);
                p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, options.preset);
                p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, options.preset);
                p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, options.preset);
                p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, options.preset);
                p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
                       options.preset);
                p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality, options.preset);
                api_result = api.create_feature(Streamline::NgxDevice(), *cmdbuf,
                                                NVSDK_NGX_Feature_SuperSampling, p, &feature);
                if (NVSDK_NGX_FAILED(api_result)) {
                    feature = nullptr;
                }
                return feature != nullptr;
            },
            [] { return true; },
            [&] {
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);
            },
            [&] {
                NVSDK_NGX_Parameter* const p = parameters;
                p->Set(NVSDK_NGX_Parameter_Color, static_cast<void*>(&color));
                p->Set(NVSDK_NGX_Parameter_Output, static_cast<void*>(&output));
                p->Set(NVSDK_NGX_Parameter_Depth, static_cast<void*>(&depth));
                p->Set(NVSDK_NGX_Parameter_MotionVectors, static_cast<void*>(&motion));
                p->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, in.jitter[0]);
                p->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, in.jitter[1]);
                // NGX takes the scale from motion vectors to render pixels, which is ours.
                p->Set(NVSDK_NGX_Parameter_MV_Scale_X, in.motion_vector_scale[0]);
                p->Set(NVSDK_NGX_Parameter_MV_Scale_Y, in.motion_vector_scale[1]);
                p->Set(NVSDK_NGX_Parameter_Reset, reset ? 1 : 0);
                p->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
                p->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
                p->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
                p->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, 0u);
                p->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, 0u);
                p->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, 0u);
                p->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, 0u);
                p->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, 0u);
                p->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, 0u);
                p->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, 0u);
                p->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, 0u);
                p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                       in.render_size.width);
                p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
                       in.render_size.height);
                return true;
            },
            [&] {
                api_result = api.evaluate_feature(*cmdbuf, feature, parameters, nullptr);
                return NVSDK_NGX_SUCCEED(api_result);
            },
            [&] {
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, output_written, {},
                                       to_general);
            });
        if (result != VideoCommon::DlssEvaluationResult::Succeeded && !failed.exchange(true)) {
            LOG_ERROR(Render_Vulkan,
                      "NVIDIA DLSS recording failed at stage {} (NGX result {:#x}); using FSR 3.1",
                      static_cast<int>(result), static_cast<u32>(api_result));
        }
        if (result != VideoCommon::DlssEvaluationResult::Succeeded) {
            clear_output();
        }
    });
    // DLSS binds its own pipelines and descriptors on the command buffer.
    scheduler.InvalidateState();
    return true;
}

} // namespace Vulkan
