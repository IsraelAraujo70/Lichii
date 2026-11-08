// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS through NVIDIA NGX on Linux, behind the same interface as Streamline on Windows
// (Streamline has no Linux runtime). The DLSS SDK's loader is its own library, libnvsdk_ngx.so,
// loaded here at run time; it finds the driver's libnvidia-ngx.so.1, which loads the DLSS model
// (libnvidia-ngx-dlss.so.*) from the folders passed below.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <dlfcn.h>

#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/vulkan_common/ngx_api.h"
#include "video_core/vulkan_common/streamline.h"

namespace Vulkan::Streamline {
namespace {

// Same id as the Windows build: the driver may key behaviour off it.
constexpr const char* PROJECT_ID = "8d3e2f9a-6b41-4c7e-9a52-1f0c3b7d5e64";
constexpr const char* LOADER = "libnvsdk_ngx.so";

struct State {
    void* module{};
    NgxApi api{};
    std::filesystem::path module_path;
    std::vector<std::wstring> model_dirs;
    std::vector<const wchar_t*> model_dir_pointers;
    std::wstring data_dir;
    std::vector<std::string> instance_extensions;
    std::vector<std::string> device_extensions;
    NVSDK_NGX_Parameter* capabilities{};
    VkDevice device{};
    bool started{};
    bool attached{};
};

State& GetState() {
    static State state;
    return state;
}

int LogLevel() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_LOG");
    return value ? std::atoi(value) : 0;
}

void NVSDK_CONV LogFromNgx(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    std::string_view text{message ? message : ""};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    LOG_INFO(Render_Vulkan, "NGX: {}", text);
}

/// Where the loader and the DLSS model may be: EDEN_DLSS_DIR, then next to the executable.
std::vector<std::filesystem::path> SearchDirs() {
    std::vector<std::filesystem::path> dirs;
    if (const char* const dir = std::getenv("EDEN_DLSS_DIR"); dir && *dir) {
        dirs.emplace_back(dir);
    }
    std::error_code ec;
    if (const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
        !ec) {
        dirs.push_back(exe.parent_path());
    }
    return dirs;
}

template <typename T>
bool Resolve(void* module, const char* name, T& function) {
    function = reinterpret_cast<T>(dlsym(module, name));
    if (!function) {
        LOG_ERROR(Render_Vulkan, "{} has no {}", LOADER, name);
    }
    return function != nullptr;
}

bool Load(State& state) {
    std::error_code ec;
    for (const std::filesystem::path& dir : SearchDirs()) {
        if (std::filesystem::is_regular_file(dir / LOADER, ec)) {
            state.module_path = dir / LOADER;
            break;
        }
    }
    if (state.module_path.empty()) {
        LOG_WARNING(Render_Vulkan,
                    "NVIDIA DLSS is on, but {} (NVIDIA NGX) is not next to the executable; using "
                    "FSR 3.1",
                    LOADER);
        return false;
    }
    state.module = dlopen(state.module_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!state.module) {
        LOG_ERROR(Render_Vulkan, "Could not load {}: {}",
                  Common::FS::PathToUTF8String(state.module_path), dlerror());
        state.module_path.clear();
        return false;
    }
    NgxApi& api = state.api;
    void* const m = state.module;
    if (Resolve(m, "NVSDK_NGX_VULKAN_RequiredExtensions", api.required_extensions) &&
        Resolve(m, "NVSDK_NGX_VULKAN_Init_with_ProjectID", api.init) &&
        Resolve(m, "NVSDK_NGX_VULKAN_Shutdown1", api.shutdown) &&
        Resolve(m, "NVSDK_NGX_VULKAN_GetCapabilityParameters", api.get_capability_parameters) &&
        Resolve(m, "NVSDK_NGX_VULKAN_AllocateParameters", api.allocate_parameters) &&
        Resolve(m, "NVSDK_NGX_VULKAN_DestroyParameters", api.destroy_parameters) &&
        Resolve(m, "NVSDK_NGX_VULKAN_CreateFeature1", api.create_feature) &&
        Resolve(m, "NVSDK_NGX_VULKAN_EvaluateFeature_C", api.evaluate_feature) &&
        Resolve(m, "NVSDK_NGX_VULKAN_ReleaseFeature", api.release_feature)) {
        return true;
    }
    dlclose(state.module);
    state.module = nullptr;
    state.module_path.clear();
    state.api = {};
    return false;
}

std::vector<std::string> Names(unsigned int count, const char** names) {
    std::vector<std::string> result;
    for (unsigned int i = 0; i < count; ++i) {
        result.emplace_back(names[i]);
    }
    return result;
}

} // Anonymous namespace

bool Start() {
    State& state = GetState();
    if (state.started) {
        return true;
    }
    if (!Settings::values.nvidia_dlss.GetValue()) {
        return false;
    }
    if (!state.module && !Load(state)) {
        return false;
    }
    unsigned int instance_count = 0;
    unsigned int device_count = 0;
    const char** instance_names{};
    const char** device_names{};
    if (const NVSDK_NGX_Result result = state.api.required_extensions(
            &instance_count, &instance_names, &device_count, &device_names);
        NVSDK_NGX_FAILED(result)) {
        LOG_ERROR(Render_Vulkan, "NVIDIA NGX gave no Vulkan extensions (result {:#x}); using FSR 3.1",
                  static_cast<u32>(result));
        return false;
    }
    state.instance_extensions = Names(instance_count, instance_names);
    state.device_extensions = Names(device_count, device_names);
    // The KHR and EXT buffer device address extensions can't both be enabled.
    if (std::ranges::find(state.device_extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) !=
        state.device_extensions.end()) {
        std::erase(state.device_extensions, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    }
    LOG_INFO(Render_Vulkan, "NVIDIA NGX loaded from {}",
             Common::FS::PathToUTF8String(state.module_path));
    state.started = true;
    return true;
}

bool WantsDevice() {
    const State& state = GetState();
    return state.started && !state.attached;
}

std::vector<std::string> InstanceExtensions() {
    const State& state = GetState();
    return state.started ? state.instance_extensions : std::vector<std::string>{};
}

std::vector<std::string> DeviceExtensions() {
    const State& state = GetState();
    return state.started ? state.device_extensions : std::vector<std::string>{};
}

bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                  [[maybe_unused]] u32 graphics_family, vk::DeviceDispatch& dld) {
    State& state = GetState();
    if (!WantsDevice()) {
        return false;
    }
    state.model_dirs.clear();
    state.model_dir_pointers.clear();
    for (const std::filesystem::path& dir : SearchDirs()) {
        state.model_dirs.push_back(dir.wstring());
    }
    for (const std::wstring& dir : state.model_dirs) {
        state.model_dir_pointers.push_back(dir.c_str());
    }
    // NGX writes its logs and caches here.
    const std::filesystem::path data_dir =
        Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "ngx";
    std::error_code ec;
    std::filesystem::create_directories(data_dir, ec);
    state.data_dir = data_dir.wstring();

    NVSDK_NGX_FeatureCommonInfo info{};
    info.PathListInfo.Path = state.model_dir_pointers.data();
    info.PathListInfo.Length = static_cast<unsigned int>(state.model_dir_pointers.size());
    const int log_level = LogLevel();
    info.LoggingInfo.LoggingCallback = LogFromNgx;
    info.LoggingInfo.MinimumLoggingLevel = log_level >= 2   ? NVSDK_NGX_LOGGING_LEVEL_VERBOSE
                                           : log_level == 1 ? NVSDK_NGX_LOGGING_LEVEL_ON
                                                            : NVSDK_NGX_LOGGING_LEVEL_OFF;
    info.LoggingInfo.DisableOtherLoggingSinks = false;

    const NgxApi& api = state.api;
    if (const NVSDK_NGX_Result result =
            api.init(PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", state.data_dir.c_str(),
                     instance, physical, device, dld.vkGetInstanceProcAddr,
                     dld.vkGetDeviceProcAddr, &info, NVSDK_NGX_Version_API);
        NVSDK_NGX_FAILED(result)) {
        LOG_WARNING(Render_Vulkan,
                    "NVIDIA NGX did not start on this GPU or driver (result {:#x}); using FSR 3.1",
                    static_cast<u32>(result));
        state.started = false;
        return false;
    }
    state.device = device;
    state.attached = true;

    int available = 0;
    int needs_driver = 0;
    int init_result = 0;
    if (NVSDK_NGX_FAILED(api.get_capability_parameters(&state.capabilities)) ||
        !state.capabilities) {
        LOG_WARNING(Render_Vulkan, "NVIDIA NGX gave no capabilities; using FSR 3.1");
        Stop();
        return false;
    }
    state.capabilities->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    state.capabilities->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_driver);
    state.capabilities->Get(NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &init_result);
    if (!available) {
        LOG_WARNING(Render_Vulkan,
                    "NVIDIA DLSS is not available (needs a newer driver: {}, result {:#x}); is "
                    "libnvidia-ngx-dlss.so.* next to the executable? Using FSR 3.1",
                    needs_driver != 0, static_cast<u32>(init_result));
        Stop();
        return false;
    }
    LOG_INFO(Render_Vulkan, "NVIDIA DLSS is available through NGX");
    return true;
}

bool IsDlssAvailable() {
    const State& state = GetState();
    return state.started && state.attached;
}

void Stop() {
    State& state = GetState();
    if (!state.started) {
        return;
    }
    if (state.attached) {
        if (state.capabilities) {
            state.api.destroy_parameters(state.capabilities);
            state.capabilities = nullptr;
        }
        const NVSDK_NGX_Result result = state.api.shutdown(state.device);
        LOG_INFO(Render_Vulkan, "NVIDIA NGX stopped (result {:#x})", static_cast<u32>(result));
    }
    state.device = VK_NULL_HANDLE;
    state.started = false;
    state.attached = false;
}

const NgxApi& GetNgx() {
    return GetState().api;
}

VkDevice NgxDevice() {
    return GetState().device;
}

} // namespace Vulkan::Streamline
