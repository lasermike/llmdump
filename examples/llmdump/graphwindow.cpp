// Dear ImGui: standalone example application for DirectX 12

// Learn about Dear ImGui:
// - FAQ                  https://dearimgui.com/faq
// - Getting Started      https://dearimgui.com/getting-started
// - Documentation        https://dearimgui.com/docs (same as your local docs/ folder).
// - Introduction, links and more at the top of imgui.cpp

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include <tchar.h>
#include <math.h>

#include "llama-model.h"
#include "log.h"
#include "..\src\ggml-impl.h"


#ifdef _DEBUG
#define DX12_ENABLE_DEBUG_LAYER
#endif

#ifdef DX12_ENABLE_DEBUG_LAYER
#include <dxgidebug.h>
#pragma comment(lib, "dxguid.lib")
#endif

#include "implot.h"

// Split
#include <iostream>
#include <sstream>
#include <vector>



// Config for example app
static const int APP_NUM_FRAMES_IN_FLIGHT = 2;
static const int APP_NUM_BACK_BUFFERS = 2;
static const int APP_SRV_HEAP_SIZE = 64;

struct FrameContext
{
    ID3D12CommandAllocator*     CommandAllocator;
    UINT64                      FenceValue;
};

// Simple free list based allocator
struct ExampleDescriptorHeapAllocator
{
    ID3D12DescriptorHeap*       Heap = nullptr;
    D3D12_DESCRIPTOR_HEAP_TYPE  HeapType = D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES;
    D3D12_CPU_DESCRIPTOR_HANDLE HeapStartCpu;
    D3D12_GPU_DESCRIPTOR_HANDLE HeapStartGpu;
    UINT                        HeapHandleIncrement;
    ImVector<int>               FreeIndices;

    void Create(ID3D12Device* device, ID3D12DescriptorHeap* heap)
    {
        IM_ASSERT(Heap == nullptr && FreeIndices.empty());
        Heap = heap;
        D3D12_DESCRIPTOR_HEAP_DESC desc = heap->GetDesc();
        HeapType = desc.Type;
        HeapStartCpu = Heap->GetCPUDescriptorHandleForHeapStart();
        HeapStartGpu = Heap->GetGPUDescriptorHandleForHeapStart();
        HeapHandleIncrement = device->GetDescriptorHandleIncrementSize(HeapType);
        FreeIndices.reserve((int)desc.NumDescriptors);
        for (int n = desc.NumDescriptors; n > 0; n--)
            FreeIndices.push_back(n - 1);
    }
    void Destroy()
    {
        Heap = nullptr;
        FreeIndices.clear();
    }
    void Alloc(D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_desc_handle, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_desc_handle)
    {
        IM_ASSERT(FreeIndices.Size > 0);
        int idx = FreeIndices.back();
        FreeIndices.pop_back();
        out_cpu_desc_handle->ptr = HeapStartCpu.ptr + (idx * HeapHandleIncrement);
        out_gpu_desc_handle->ptr = HeapStartGpu.ptr + (idx * HeapHandleIncrement);
    }
    void Free(D3D12_CPU_DESCRIPTOR_HANDLE out_cpu_desc_handle, D3D12_GPU_DESCRIPTOR_HANDLE out_gpu_desc_handle)
    {
        int cpu_idx = (int)((out_cpu_desc_handle.ptr - HeapStartCpu.ptr) / HeapHandleIncrement);
        int gpu_idx = (int)((out_gpu_desc_handle.ptr - HeapStartGpu.ptr) / HeapHandleIncrement);
        IM_ASSERT(cpu_idx == gpu_idx);
        FreeIndices.push_back(cpu_idx);
    }
};

// Data
static FrameContext                 g_frameContext[APP_NUM_FRAMES_IN_FLIGHT] = {};
static UINT                         g_frameIndex = 0;

static ID3D12Device*                g_pd3dDevice = nullptr;
static ID3D12DescriptorHeap*        g_pd3dRtvDescHeap = nullptr;
static ID3D12DescriptorHeap*        g_pd3dSrvDescHeap = nullptr;
static ExampleDescriptorHeapAllocator g_pd3dSrvDescHeapAlloc;
static ID3D12CommandQueue*          g_pd3dCommandQueue = nullptr;
static ID3D12GraphicsCommandList*   g_pd3dCommandList = nullptr;
static ID3D12Fence*                 g_fence = nullptr;
static HANDLE                       g_fenceEvent = nullptr;
static UINT64                       g_fenceLastSignaledValue = 0;
static IDXGISwapChain3*             g_pSwapChain = nullptr;
static bool                         g_SwapChainOccluded = false;
static HANDLE                       g_hSwapChainWaitableObject = nullptr;
static ID3D12Resource*              g_mainRenderTargetResource[APP_NUM_BACK_BUFFERS] = {};
static D3D12_CPU_DESCRIPTOR_HANDLE  g_mainRenderTargetDescriptor[APP_NUM_BACK_BUFFERS] = {};

// Forward declarations of helper functions
bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
void WaitForLastSubmittedFrame();
FrameContext* WaitForNextFrameResources();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);


static const char* GGML_OP_NAME[83] = { //GGML_OP_COUNT
    "NONE",

    "DUP",
    "ADD",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "LOG",
    "SIN",
    "COS",
    "SUM",
    "SUM_ROWS",
    "MEAN",
    "ARGMAX",
    "COUNT_EQUAL",
    "REPEAT",
    "REPEAT_BACK",
    "CONCAT",
    "SILU_BACK",
    "NORM",
    "RMS_NORM",
    "RMS_NORM_BACK",
    "GROUP_NORM",

    "MUL_MAT",
    "MUL_MAT_ID",
    "OUT_PROD",

    "SCALE",
    "SET",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "GET_ROWS_BACK",
    "DIAG",
    "DIAG_MASK_INF",
    "DIAG_MASK_ZERO",
    "SOFT_MAX",
    "SOFT_MAX_BACK",
    "ROPE",
    "ROPE_BACK",
    "CLAMP",
    "CONV_TRANSPOSE_1D",
    "IM2COL",
    "IM2COL_BACK",
    "CONV_TRANSPOSE_2D",
    "POOL_1D",
    "POOL_2D",
    "POOL_2D_BACK",
    "UPSCALE",
    "PAD",
    "PAD_REFLECT_1D",
    "ARANGE",
    "TIMESTEP_EMBEDDING",
    "ARGSORT",
    "LEAKY_RELU",

    "FLASH_ATTN_EXT",
    "FLASH_ATTN_BACK",
    "SSM_CONV",
    "SSM_SCAN",
    "WIN_PART",
    "WIN_UNPART",
    "GET_REL_POS",
    "ADD_REL_POS",
    "RWKV_WKV6",
    "GATED_LINEAR_ATTN",

    "UNARY",

    "MAP_UNARY",
    "MAP_BINARY",

    "MAP_CUSTOM1_F32",
    "MAP_CUSTOM2_F32",
    "MAP_CUSTOM3_F32",

    "MAP_CUSTOM1",
    "MAP_CUSTOM2",
    "MAP_CUSTOM3",

    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK",
    "OPT_STEP_ADAMW",
};


const int maxSamples = 1024;
const int maxGraphs = 256;
const int maxLayerTensors = 512;
const int maxTensorsPerLayer = 32;

const int maxLayers = 28;
const int maxLayersIncludingInputOutput = 30;
const int maxTensorNameLen = 24;

// Raw graphs
__int64 rawTensorsToGraph[maxGraphs][maxSamples];
int showInGraphNum[maxGraphs] = { 0 };
int numSamples = 0;
int currentUISample = 0;
int lastNumRawTensors = 0;

__int64 xs[maxSamples];

// By Layer 
int numModelLayerTensors = 0;
int numGraphLayerTensors = 0;
//__int64 layerSamples[maxSamples][maxLayersIncludingInputOutput][maxTensorsPerLayer];
_int64* layerSamples = nullptr;

int layerStride = 0;
int sampleStride = 0;


#define LAYER_INDEX_UNDEFINED -1
#define LAYER_INDEX_INPUT maxLayers
#define LAYER_INDEX_OUTPUT maxLayers + 1

struct LayerTensorInfo
{
    std::string name;
    int layerIndex;
    int tensorWithinLayerIndex;
    int tensorIndex;
};

std::vector<LayerTensorInfo> modelLayerTensorsInfo;  // TODO: better allocation management, remove usage of std
std::vector<LayerTensorInfo> graphLayerTensorsInfo;  // TODO: better allocation management, remove usage of std

int numTensorsPerLayer = 0;

int graphwindow_addData(__int64* rawTensorValues, int numRawTensors, __int64* layerTensorValues)
{
    // Raw tensor viz
    if (numRawTensors < maxGraphs)
    {
        for (int i = 0; i < numRawTensors; i++)
        {
            rawTensorsToGraph[i][numSamples] = rawTensorValues[i];   // Poor cache...
            if (rawTensorValues[i] > 4000000 && showInGraphNum[i] < 3)
            {
                showInGraphNum[i] = 3;
            }
            else if (rawTensorValues[i] > 1000 && showInGraphNum[i] < 2)
            {
                showInGraphNum[i] = 2;
            }
            else if (rawTensorValues[i] > 0 && showInGraphNum[i] < 1)
            {
                showInGraphNum[i] = 1;
            }

        }

        lastNumRawTensors = numRawTensors;
    }

    // Tensors by layer viz
    //for (int i = 0; i < numModelLayerTensors; i++)
    //{
    //    int layerIndex = modelLayerTensorsInfo[i].layerIndex;
    //    int tensorWithinLayerIndex = modelLayerTensorsInfo[i].tensorWithinLayerIndex;

    //    if (tensorWithinLayerIndex >= 0)
    //    {
    //        layerSamples[numSamples][layerIndex][tensorWithinLayerIndex] = layerTensorValues[i];
    //    }

    //    //TODO Input and ouput tensors
    //}

    // Graph tensors by layer viz
    for (int i = 0; i < numGraphLayerTensors; i++)
    {
        int layerIndex = graphLayerTensorsInfo[i].layerIndex;
        int tensorWithinLayerIndex = graphLayerTensorsInfo[i].tensorWithinLayerIndex;

        if (tensorWithinLayerIndex >= 0 && layerIndex >= 0)
        {
            int sampleIndex = numSamples * sampleStride + layerIndex * layerStride + tensorWithinLayerIndex;
            layerSamples[sampleIndex] = layerTensorValues[i];
        }

        //TODO Input and ouput tensors
    }


    currentUISample = numSamples;

    numSamples++; // TODO: atomic

    return 0;
}

char tensorNamesPerLayer[maxTensorsPerLayer][maxTensorNameLen];

char* tensorNamePtrsPerLayer[maxTensorsPerLayer];

int graphWindow_initLayerOps(struct ggml_cgraph* graph)
{
    numTensorsPerLayer = 0;

    for (int i = 0; i < maxTensorsPerLayer; i++)
    {
        tensorNamesPerLayer[i][0] = '\0';
        tensorNamePtrsPerLayer[i] = tensorNamesPerLayer[i];
    }

    numGraphLayerTensors = graph->n_nodes;

    for (int i = 0; i < graph->n_nodes; i++) {
        const ggml_tensor* node = graph->nodes[i];
        graphLayerTensorsInfo.push_back({ node->name, node->layerNumber, node->layerOpNumber, i});

        numTensorsPerLayer = __max(node->layerOpNumber, numTensorsPerLayer);

        if (tensorNamesPerLayer[node->layerOpNumber][0] == '\0')
        {
            strncpy(tensorNamesPerLayer[node->layerOpNumber], node->name, maxTensorNameLen);

            char* dashPtr = strchr(tensorNamesPerLayer[node->layerOpNumber], '-');
            if (dashPtr != nullptr)
            {
                dashPtr[0] = '\0';
            }
        }

    }

    assert(!layerSamples);

    layerStride = numTensorsPerLayer;
    sampleStride = layerStride * maxLayers;

    int sampleBufferSize = maxSamples * maxLayers * numTensorsPerLayer;
    layerSamples = new __int64[sampleBufferSize];

    memset(layerSamples, 0, sizeof(__int64) * sampleBufferSize);


    //ggml_graph_print(graph);

    // Uncomment to dump graph
    //static bool layerOpNamesInited = false;
    //if (!layerOpNamesInited) {
    //    layerOpNamesInited = true;

    //    for (int i = 0; i < graph->n_nodes; i++) {
    //        LOG_INF("%d;\t%d;\t%s\n", graph->nodes[i]->layerNumber,
    //            graph->nodes[i]->layerOpNumber,
    //            graph->nodes[i]->name);
    //    }
    //}

    return 0;
}



std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Main code
int graphwindow_main(llama_model* model)
{
    /// Parse model info

    numModelLayerTensors = model->tensors_by_name.size();

    int curBlk = 0;
    int curTensorWithinLayer = -1;
    int i = 0;

    for (auto& tensor : model->tensors_by_name)
    {
        LOG_INF("---- %s\n", tensor.first.c_str());

        std::vector<std::string> result = split(tensor.first, '.' /*delimiter*/);

        if (result[0].compare("token_embd") == 0)
        {
            modelLayerTensorsInfo.push_back({ result[0], LAYER_INDEX_INPUT, 0, i});
        }
        else if (result[0].compare("output_norm") == 0)
        {
            modelLayerTensorsInfo.push_back({ result[0], LAYER_INDEX_OUTPUT, 0, i });
        }
        else if (result[0].compare("blk") == 0)
        {
            int layerIndex = strtol(result[1].c_str(), nullptr, 10);

            if (layerIndex == curBlk)
            {
                curTensorWithinLayer++;
            }
            else
            {
                curBlk = layerIndex;
                curTensorWithinLayer = 0;
            }

            modelLayerTensorsInfo.push_back({ result[2], layerIndex, curTensorWithinLayer, i });
        }
        else
        {
            assert(false);
        }

        i++;
    }

    for (int i = 0; i < maxSamples; i++)
    {
        xs[i] = i;
    }

    static const char* xlabels[] = { "C0","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10", "C11","C12","C13","C14","C15","C16","C17","C18","C19","C20", "C21","C22","C23","C24","C25","C26","C27","C28","C29","C30", };
    static const char* ylabels[] = { "0", "1","2","3","4","5","6","7", "8", "9", "10","11","12","13","14","15","16","17", "18", "19", "20", "21","22","23","24","25","26","27", "28", "29", "30", };


    ////////////


    // Create application window
    //ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"ImGui Example", nullptr };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"Tensor time", WS_OVERLAPPEDWINDOW, 100, 100, 1600, 1000, nullptr, nullptr, wc.hInstance, nullptr);

    // Initialize Direct3D
    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    // Show the window
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(hwnd);

    ImGui_ImplDX12_InitInfo init_info = {};
    init_info.Device = g_pd3dDevice;
    init_info.CommandQueue = g_pd3dCommandQueue;
    init_info.NumFramesInFlight = APP_NUM_FRAMES_IN_FLIGHT;
    init_info.RTVFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    init_info.DSVFormat = DXGI_FORMAT_UNKNOWN;
    // Allocating SRV descriptors (for textures) is up to the application, so we provide callbacks.
    // (current version of the backend will only allocate one descriptor, future versions will need to allocate more)
    init_info.SrvDescriptorHeap = g_pd3dSrvDescHeap;
    init_info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE* out_cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE* out_gpu_handle) { return g_pd3dSrvDescHeapAlloc.Alloc(out_cpu_handle, out_gpu_handle); };
    init_info.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)            { return g_pd3dSrvDescHeapAlloc.Free(cpu_handle, gpu_handle); };
    ImGui_ImplDX12_Init(&init_info);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return a nullptr. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != nullptr);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    ImPlot::GetStyle().Colormap = ImPlotColormap_Spectral;

    // Main loop
    bool done = false;
    while (!done)
    {
        // Poll and handle messages (inputs, window resize, etc.)
        // See the WndProc() function below for our to dispatch events to the Win32 backend.
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done)
            break;

        // Handle window screen locked
        if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED)
        {
            ::Sleep(10);
            continue;
        }
        g_SwapChainOccluded = false;

        // Start the Dear ImGui frame
        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        bool itsTrue = true;
        ImGui::Begin("##LLM Analysis", &itsTrue, ImGuiWindowFlags_NoTitleBar);
        {
            static int uiMode = 0;
            {
                ImGui::RadioButton("Layers", &uiMode, 0);
                ImGui::SameLine();
                ImGui::RadioButton("Raw Tensors", &uiMode, 1);
                ImGui::SameLine();
                ImGui::LabelText("##ModeSelect", "%s", "View mode");
            }

            if (uiMode == 0)
            {
                //// Layer mode
                //static ImPlotColormap map = ImPlotColormap_Viridis;
                //if (ImPlot::ColormapButton(ImPlot::GetColormapName(map), ImVec2(225, 0), map)) {
                //    map = (map + 1) % ImPlot::GetColormapCount();
                //    // We bust the color cache of our plots so that item colors will
                //    // resample the new colormap in the event that they have already
                //    // been created. See documentation in implot.h.
                //    ImPlot::BustColorCache("##Heatmap1");
                //    ImPlot::BustColorCache("##Heatmap2");
                //}

                //ImGui::SameLine();
                //ImGui::LabelText("##Colormap Index", "%s", "Change Colormap");

                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Time(ms) per tensor op");


                ImGui::SetNextItemWidth(225);
                static float mn = 0, mx = 1;
                ImGui::DragFloatRange2("Range", &mn, &mx, 0.1f, 0, 1000000);
                ImGui::SetNextItemWidth(225);
                ImGui::DragInt("Sample", &currentUISample, 1, 0, numSamples);

                // ImPlot::ShowColormapSelector("Colormap");
                //if (ImGui::DragInt2("Size", &rows, 10, 0, 10000))
                //    generate_noise();
                //if (ImGui::DragFloat("Scrub", &z, 0.1f, 0, 10))
                //    generate_noise();

                const ImPlotAxisFlags axes_flags = ImPlotAxisFlags_None; // ImPlotAxisFlags_Lock; // ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks;

                if (ImPlot::BeginPlot("##Heatmap1", ImVec2(1500, 800), ImPlotFlags_NoMouseText)) {

                    ImPlot::SetupAxes("Op", "Layer", ImPlotAxisFlags_NoTickLabels, axes_flags);
                    ImPlot::SetupAxisTicks(ImAxis_X1, 0, numTensorsPerLayer, numTensorsPerLayer, tensorNamePtrsPerLayer);
                    ImPlot::SetupAxisTicks(ImAxis_Y1, 0, 28, maxLayers, ylabels);
                    ImPlot::PlotHeatmap("##T", (__int64*) &(layerSamples[currentUISample * sampleStride]), maxLayers, numTensorsPerLayer, mn, mx, "%d", ImPlotPoint(0.0, 0.0), ImPlotPoint(numTensorsPerLayer, maxLayers), 0);

                    for (int i = 0; i < numTensorsPerLayer; ++i) {
                        // Adjust the y coordinate (here, 0.0) to position the text as needed.
                        ImPlot::PlotText(tensorNamePtrsPerLayer[i], (double)i + 0.5, -3.0, ImVec2(0,0), ImPlotTextFlags_Vertical);
                    }

                    ImPlot::EndPlot();
                }
            }
            else if (uiMode == 1)
            {
                // Raw tensor mode
                if (ImPlot::BeginPlot("Tensor op time < 1000 us", ImVec2(1280, 300))) {
                    ImPlot::SetupAxes("Samples", "Time (us)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    for (int i = 0; i < lastNumRawTensors; i++)
                    {
                        if (showInGraphNum[i] == 1)
                        {
                            ImPlot::PlotLine(GGML_OP_NAME[i], xs, &rawTensorsToGraph[i][0], numSamples);
                        }
                    }
                    ImPlot::EndPlot();
                }

                // Larger graph scale
                if (ImPlot::BeginPlot("Tensor ops time scale > 1000 us", ImVec2(1280, 300))) {
                    ImPlot::SetupAxes("Samples", "Time (us)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    for (int i = 0; i < lastNumRawTensors; i++)
                    {
                        if (showInGraphNum[i] == 2)
                        {
                            ImPlot::PlotLine(GGML_OP_NAME[i], xs, &rawTensorsToGraph[i][0], numSamples);
                        }
                    }
                    ImPlot::EndPlot();
                }

                // Larger graph scale
                if (ImPlot::BeginPlot("Tensor ops time scale > 3000000 us", ImVec2(1280, 300))) {
                    ImPlot::SetupAxes("Samples", "Time (us)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    for (int i = 0; i < lastNumRawTensors; i++)
                    {
                        if (showInGraphNum[i] == 3)
                        {
                            ImPlot::PlotLine(GGML_OP_NAME[i], xs, &rawTensorsToGraph[i][0], numSamples);
                        }
                    }
                    ImPlot::EndPlot();
                }
            }
        }
        ImGui::End();

        // Rendering
        ImGui::Render();

        FrameContext* frameCtx = WaitForNextFrameResources();
        UINT backBufferIdx = g_pSwapChain->GetCurrentBackBufferIndex();
        frameCtx->CommandAllocator->Reset();

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource   = g_mainRenderTargetResource[backBufferIdx];
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
        g_pd3dCommandList->Reset(frameCtx->CommandAllocator, nullptr);
        g_pd3dCommandList->ResourceBarrier(1, &barrier);

        // Render Dear ImGui graphics
        const float clear_color_with_alpha[4] = { clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w };
        g_pd3dCommandList->ClearRenderTargetView(g_mainRenderTargetDescriptor[backBufferIdx], clear_color_with_alpha, 0, nullptr);
        g_pd3dCommandList->OMSetRenderTargets(1, &g_mainRenderTargetDescriptor[backBufferIdx], FALSE, nullptr);
        g_pd3dCommandList->SetDescriptorHeaps(1, &g_pd3dSrvDescHeap);
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), g_pd3dCommandList);
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
        g_pd3dCommandList->ResourceBarrier(1, &barrier);
        g_pd3dCommandList->Close();

        g_pd3dCommandQueue->ExecuteCommandLists(1, (ID3D12CommandList* const*)&g_pd3dCommandList);

        // Present
        HRESULT hr = g_pSwapChain->Present(1, 0);   // Present with vsync
        //HRESULT hr = g_pSwapChain->Present(0, 0); // Present without vsync
        g_SwapChainOccluded = (hr == DXGI_STATUS_OCCLUDED);

        UINT64 fenceValue = g_fenceLastSignaledValue + 1;
        g_pd3dCommandQueue->Signal(g_fence, fenceValue);
        g_fenceLastSignaledValue = fenceValue;
        frameCtx->FenceValue = fenceValue;
    }

    WaitForLastSubmittedFrame();

    // Cleanup
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}

// Helper functions

bool CreateDeviceD3D(HWND hWnd)
{
    // Setup swap chain
    DXGI_SWAP_CHAIN_DESC1 sd;
    {
        ZeroMemory(&sd, sizeof(sd));
        sd.BufferCount = APP_NUM_BACK_BUFFERS;
        sd.Width = 0;
        sd.Height = 0;
        sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.SampleDesc.Count = 1;
        sd.SampleDesc.Quality = 0;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
        sd.Scaling = DXGI_SCALING_STRETCH;
        sd.Stereo = FALSE;
    }

    // [DEBUG] Enable debug interface
#ifdef DX12_ENABLE_DEBUG_LAYER
    ID3D12Debug* pdx12Debug = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pdx12Debug))))
        pdx12Debug->EnableDebugLayer();
#endif

    // Create device
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    if (D3D12CreateDevice(nullptr, featureLevel, IID_PPV_ARGS(&g_pd3dDevice)) != S_OK)
        return false;

    // [DEBUG] Setup debug interface to break on any warnings/errors
#ifdef DX12_ENABLE_DEBUG_LAYER
    if (pdx12Debug != nullptr)
    {
        ID3D12InfoQueue* pInfoQueue = nullptr;
        g_pd3dDevice->QueryInterface(IID_PPV_ARGS(&pInfoQueue));
        pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
        pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
        pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, true);
        pInfoQueue->Release();
        pdx12Debug->Release();
    }
#endif

    {
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        desc.NumDescriptors = APP_NUM_BACK_BUFFERS;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        desc.NodeMask = 1;
        if (g_pd3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&g_pd3dRtvDescHeap)) != S_OK)
            return false;

        SIZE_T rtvDescriptorSize = g_pd3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = g_pd3dRtvDescHeap->GetCPUDescriptorHandleForHeapStart();
        for (UINT i = 0; i < APP_NUM_BACK_BUFFERS; i++)
        {
            g_mainRenderTargetDescriptor[i] = rtvHandle;
            rtvHandle.ptr += rtvDescriptorSize;
        }
    }

    {
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = APP_SRV_HEAP_SIZE;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (g_pd3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&g_pd3dSrvDescHeap)) != S_OK)
            return false;
        g_pd3dSrvDescHeapAlloc.Create(g_pd3dDevice, g_pd3dSrvDescHeap);
    }

    {
        D3D12_COMMAND_QUEUE_DESC desc = {};
        desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        desc.NodeMask = 1;
        if (g_pd3dDevice->CreateCommandQueue(&desc, IID_PPV_ARGS(&g_pd3dCommandQueue)) != S_OK)
            return false;
    }

    for (UINT i = 0; i < APP_NUM_FRAMES_IN_FLIGHT; i++)
        if (g_pd3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_frameContext[i].CommandAllocator)) != S_OK)
            return false;

    if (g_pd3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_frameContext[0].CommandAllocator, nullptr, IID_PPV_ARGS(&g_pd3dCommandList)) != S_OK ||
        g_pd3dCommandList->Close() != S_OK)
        return false;

    if (g_pd3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)) != S_OK)
        return false;

    g_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (g_fenceEvent == nullptr)
        return false;

    {
        IDXGIFactory4* dxgiFactory = nullptr;
        IDXGISwapChain1* swapChain1 = nullptr;
        if (CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)) != S_OK)
            return false;
        if (dxgiFactory->CreateSwapChainForHwnd(g_pd3dCommandQueue, hWnd, &sd, nullptr, nullptr, &swapChain1) != S_OK)
            return false;
        if (swapChain1->QueryInterface(IID_PPV_ARGS(&g_pSwapChain)) != S_OK)
            return false;
        swapChain1->Release();
        dxgiFactory->Release();
        g_pSwapChain->SetMaximumFrameLatency(APP_NUM_BACK_BUFFERS);
        g_hSwapChainWaitableObject = g_pSwapChain->GetFrameLatencyWaitableObject();
    }

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->SetFullscreenState(false, nullptr); g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_hSwapChainWaitableObject != nullptr) { CloseHandle(g_hSwapChainWaitableObject); }
    for (UINT i = 0; i < APP_NUM_FRAMES_IN_FLIGHT; i++)
        if (g_frameContext[i].CommandAllocator) { g_frameContext[i].CommandAllocator->Release(); g_frameContext[i].CommandAllocator = nullptr; }
    if (g_pd3dCommandQueue) { g_pd3dCommandQueue->Release(); g_pd3dCommandQueue = nullptr; }
    if (g_pd3dCommandList) { g_pd3dCommandList->Release(); g_pd3dCommandList = nullptr; }
    if (g_pd3dRtvDescHeap) { g_pd3dRtvDescHeap->Release(); g_pd3dRtvDescHeap = nullptr; }
    if (g_pd3dSrvDescHeap) { g_pd3dSrvDescHeap->Release(); g_pd3dSrvDescHeap = nullptr; }
    if (g_fence) { g_fence->Release(); g_fence = nullptr; }
    if (g_fenceEvent) { CloseHandle(g_fenceEvent); g_fenceEvent = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }

#ifdef DX12_ENABLE_DEBUG_LAYER
    IDXGIDebug1* pDebug = nullptr;
    if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&pDebug))))
    {
        pDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_SUMMARY);
        pDebug->Release();
    }
#endif
}

void CreateRenderTarget()
{
    for (UINT i = 0; i < APP_NUM_BACK_BUFFERS; i++)
    {
        ID3D12Resource* pBackBuffer = nullptr;
        g_pSwapChain->GetBuffer(i, IID_PPV_ARGS(&pBackBuffer));
        g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, g_mainRenderTargetDescriptor[i]);
        g_mainRenderTargetResource[i] = pBackBuffer;
    }
}

void CleanupRenderTarget()
{
    WaitForLastSubmittedFrame();

    for (UINT i = 0; i < APP_NUM_BACK_BUFFERS; i++)
        if (g_mainRenderTargetResource[i]) { g_mainRenderTargetResource[i]->Release(); g_mainRenderTargetResource[i] = nullptr; }
}

void WaitForLastSubmittedFrame()
{
    FrameContext* frameCtx = &g_frameContext[g_frameIndex % APP_NUM_FRAMES_IN_FLIGHT];

    UINT64 fenceValue = frameCtx->FenceValue;
    if (fenceValue == 0)
        return; // No fence was signaled

    frameCtx->FenceValue = 0;
    if (g_fence->GetCompletedValue() >= fenceValue)
        return;

    g_fence->SetEventOnCompletion(fenceValue, g_fenceEvent);
    WaitForSingleObject(g_fenceEvent, INFINITE);
}

FrameContext* WaitForNextFrameResources()
{
    UINT nextFrameIndex = g_frameIndex + 1;
    g_frameIndex = nextFrameIndex;

    HANDLE waitableObjects[] = { g_hSwapChainWaitableObject, nullptr };
    DWORD numWaitableObjects = 1;

    FrameContext* frameCtx = &g_frameContext[nextFrameIndex % APP_NUM_FRAMES_IN_FLIGHT];
    UINT64 fenceValue = frameCtx->FenceValue;
    if (fenceValue != 0) // means no fence was signaled
    {
        frameCtx->FenceValue = 0;
        g_fence->SetEventOnCompletion(fenceValue, g_fenceEvent);
        waitableObjects[1] = g_fenceEvent;
        numWaitableObjects = 2;
    }

    WaitForMultipleObjects(numWaitableObjects, waitableObjects, TRUE, INFINITE);

    return frameCtx;
}

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (g_pd3dDevice != nullptr && wParam != SIZE_MINIMIZED)
        {
            WaitForLastSubmittedFrame();
            CleanupRenderTarget();
            HRESULT result = g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
            assert(SUCCEEDED(result) && "Failed to resize swapchain.");
            CreateRenderTarget();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}
