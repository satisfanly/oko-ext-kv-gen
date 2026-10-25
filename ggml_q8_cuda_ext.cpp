#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <ggml.h>
#include <ggml-backend.h>

#include <mutex>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <stdexcept>

namespace {

std::mutex g_mutex;
ggml_backend_t g_backend = nullptr;
int g_backend_device = -1;

// CPU-only metadata workspace. GGML's old path allocated/freed 16 MiB of
// host metadata for EVERY individual projection. It is safe to reuse because
// the global mutex serializes GGML calls, and no ggml_context escapes a call.
constexpr size_t METADATA_BYTES = 16 * 1024 * 1024;
std::vector<std::max_align_t> g_metadata;

uint64_t g_input_cpu_bytes = 0;
uint64_t g_input_cuda_bytes = 0;
uint64_t g_weight_upload_bytes = 0;
uint64_t g_output_download_bytes = 0;

bool enable_direct_input() {
    const char * flag = std::getenv("GGML_Q8_DIRECT_INPUT");
    return flag == nullptr || (std::string(flag) != "0" && std::string(flag) != "false");
}

void ensure_backend() {
    if (g_backend == nullptr) {
        throw std::runtime_error("GGML backend is not initialized; call init_backend() first");
    }
}

struct CtxBuffer {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ~CtxBuffer() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

CtxBuffer make_context() {
    // no_alloc=true => this memory is only GGML graph/tensor metadata.
    if (g_metadata.empty()) {
        constexpr size_t n = (METADATA_BYTES + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
        g_metadata.resize(n);
    }
    ggml_init_params params = {
        /* .mem_size   = */ METADATA_BYTES,
        /* .mem_buffer = */ g_metadata.data(),
        /* .no_alloc   = */ true,
    };

    CtxBuffer ret;
    ret.ctx = ggml_init(params);
    if (ret.ctx == nullptr) {
        throw std::runtime_error("ggml_init() failed");
    }
    return ret;
}

void validate_packed(const torch::Tensor & packed) {
    TORCH_CHECK(packed.device().is_cpu(), "packed Q8 tensor must remain on CPU");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed Q8 tensor must be torch.uint8");
    TORCH_CHECK(packed.is_contiguous(), "packed Q8 tensor must be contiguous");
}

} // namespace

void init_backend(const std::string & backend_dir, int device) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_backend != nullptr) {
        return;
    }

    // Load libggml-cuda.so and the other backend plugins built by llama.cpp.
    ggml_backend_load_all_from_path(backend_dir.c_str());

    const std::string cuda_name = "CUDA" + std::to_string(device);
    g_backend = ggml_backend_init_by_name(cuda_name.c_str(), nullptr);

    if (g_backend == nullptr && device == 0) {
        // Useful fallback for builds where the device is registered as the first GPU
        // but has a backend-specific name.
        g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }

    if (g_backend == nullptr) {
        throw std::runtime_error(
            "Unable to initialize GGML CUDA backend " + cuda_name +
            ". Check that llama.cpp was built with -DGGML_CUDA=ON and that " +
            backend_dir + " contains libggml-cuda.so"
        );
    }
    g_backend_device = device;
}

std::string backend_name() {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_backend();
    return ggml_backend_name(g_backend);
}

torch::Tensor q8_mul_mat(
    torch::Tensor packed,
    int64_t in_features,
    int64_t out_features,
    torch::Tensor x
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_backend();
    validate_packed(packed);

    const bool direct_cuda = x.is_cuda() && enable_direct_input();
    TORCH_CHECK(x.device().is_cpu() || direct_cuda,
                "x must be CPU F32 or CUDA with direct input enabled");
    if (direct_cuda) {
        TORCH_CHECK(x.get_device() == g_backend_device, "x CUDA device does not match the GGML backend");
        TORCH_CHECK(x.scalar_type() == torch::kFloat32 ||
                    x.scalar_type() == torch::kFloat16 ||
                    x.scalar_type() == torch::kBFloat16,
                    "direct CUDA input requires F32, F16, or BF16");
    } else {
        TORCH_CHECK(x.scalar_type() == torch::kFloat32, "CPU x must be float32");
    }
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be [rows, in_features]");
    TORCH_CHECK(x.size(1) == in_features, "x.size(1) does not match in_features");
    TORCH_CHECK(in_features % 32 == 0, "Q8_0 requires in_features divisible by 32");

    const int64_t rows = x.size(0);

    auto cb = make_context();

    // GGML matrix dimensions are ne0 x ne1. A linear weight [out, in] in
    // PyTorch is stored in GGUF as GGML [in, out].
    ggml_tensor * w = ggml_new_tensor_2d(cb.ctx, GGML_TYPE_Q8_0, in_features, out_features);
    ggml_tensor * a = ggml_new_tensor_2d(cb.ctx, GGML_TYPE_F32, in_features, rows);
    ggml_tensor * y = ggml_mul_mat(cb.ctx, w, a); // [out_features, rows]

    TORCH_CHECK(
        static_cast<int64_t>(ggml_nbytes(w)) == packed.numel(),
        "packed Q8 byte count mismatch: expected ", ggml_nbytes(w),
        ", got ", packed.numel()
    );

    ggml_cgraph * graph = ggml_new_graph(cb.ctx);
    ggml_build_forward_expand(graph, y);

    cb.buffer = ggml_backend_alloc_ctx_tensors(cb.ctx, g_backend);
    TORCH_CHECK(cb.buffer != nullptr, "ggml_backend_alloc_ctx_tensors() failed");
    ggml_backend_buffer_set_usage(cb.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_tensor_set(w, packed.data_ptr<uint8_t>(), 0, packed.numel());
    g_weight_upload_bytes += packed.numel();

    if (direct_cuda) {
        // GGML already allocated a F32 CUDA input tensor `a`. Wrap that exact
        // device allocation as a NON-OWNING Torch view and copy/cast in-place.
        // Unlike x.float() or x.to("cpu"), this creates no additional CUDA
        // tensor allocation. This handles F16/BF16/F32 using ATen copy_ kernels.
        auto a_view = torch::from_blob(
            a->data, {rows, in_features}, [](void *) {},
            torch::TensorOptions().device(x.device()).dtype(torch::kFloat32));
        {
            torch::NoGradGuard guard;
            a_view.copy_(x);
        }
        // GGML CUDA runs on an independent stream. Ensure Torch has completed
        // the copy/cast before GGML starts reading `a` on its own stream.
        c10::cuda::getCurrentCUDAStream(x.get_device()).synchronize();
        g_input_cuda_bytes += x.numel() * x.element_size();
    } else {
        ggml_backend_tensor_set(a, x.data_ptr<float>(), 0, x.numel() * sizeof(float));
        g_input_cpu_bytes += x.numel() * sizeof(float);
    }

    const ggml_status status = ggml_backend_graph_compute(g_backend, graph);
    TORCH_CHECK(status == GGML_STATUS_SUCCESS, "GGML CUDA graph failed: ", ggml_status_to_string(status));

    auto out = torch::empty({rows, out_features}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    TORCH_CHECK(y->type == GGML_TYPE_F32, "unexpected GGML output type for q8_mul_mat");
    ggml_backend_tensor_get(y, out.data_ptr<float>(), 0, out.numel() * sizeof(float));
    g_output_download_bytes += out.numel() * sizeof(float);

    return out;
}

torch::Tensor q8_get_rows(
    torch::Tensor packed,
    int64_t embedding_dim,
    int64_t num_embeddings,
    torch::Tensor ids
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_backend();
    validate_packed(packed);

    TORCH_CHECK(ids.device().is_cpu(), "ids must be CPU");
    TORCH_CHECK(ids.scalar_type() == torch::kInt32, "ids must be int32");
    TORCH_CHECK(ids.is_contiguous(), "ids must be contiguous");
    TORCH_CHECK(ids.dim() == 1, "ids must be flattened");
    TORCH_CHECK(embedding_dim % 32 == 0, "Q8_0 requires embedding_dim divisible by 32");

    const int64_t rows = ids.numel();

    auto cb = make_context();

    ggml_tensor * w = ggml_new_tensor_2d(cb.ctx, GGML_TYPE_Q8_0, embedding_dim, num_embeddings);
    ggml_tensor * idx = ggml_new_tensor_1d(cb.ctx, GGML_TYPE_I32, rows);
    ggml_tensor * y = ggml_get_rows(cb.ctx, w, idx); // [embedding_dim, rows]

    TORCH_CHECK(
        static_cast<int64_t>(ggml_nbytes(w)) == packed.numel(),
        "packed Q8 byte count mismatch: expected ", ggml_nbytes(w),
        ", got ", packed.numel()
    );

    ggml_cgraph * graph = ggml_new_graph(cb.ctx);
    ggml_build_forward_expand(graph, y);

    cb.buffer = ggml_backend_alloc_ctx_tensors(cb.ctx, g_backend);
    TORCH_CHECK(cb.buffer != nullptr, "ggml_backend_alloc_ctx_tensors() failed");
    ggml_backend_buffer_set_usage(cb.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_tensor_set(w, packed.data_ptr<uint8_t>(), 0, packed.numel());
    ggml_backend_tensor_set(idx, ids.data_ptr<int32_t>(), 0, ids.numel() * sizeof(int32_t));

    const ggml_status status = ggml_backend_graph_compute(g_backend, graph);
    TORCH_CHECK(status == GGML_STATUS_SUCCESS, "GGML CUDA graph failed: ", ggml_status_to_string(status));

    auto out = torch::empty({rows, embedding_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    TORCH_CHECK(y->type == GGML_TYPE_F32, "unexpected GGML output type for q8_get_rows");
    ggml_backend_tensor_get(y, out.data_ptr<float>(), 0, out.numel() * sizeof(float));

    return out;
}

pybind11::dict io_stats() {
    std::lock_guard<std::mutex> lock(g_mutex);
    pybind11::dict d;
    d["input_cpu_bytes"] = g_input_cpu_bytes;
    d["input_cuda_bytes"] = g_input_cuda_bytes;
    d["weight_upload_bytes"] = g_weight_upload_bytes;
    d["output_download_bytes"] = g_output_download_bytes;
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_backend", &init_backend, "Load llama.cpp GGML CUDA backend");
    m.def("backend_name", &backend_name, "Return active GGML backend name");
    m.def("q8_mul_mat", &q8_mul_mat, "GGML CUDA Q8_0 mul_mat");
    m.def("q8_get_rows", &q8_get_rows, "GGML CUDA Q8_0 get_rows");
    m.def("io_stats", &io_stats, "Return cumulative GGML bridge traffic counts");
}
