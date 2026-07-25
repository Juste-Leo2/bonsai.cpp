#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-webgpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static float cosine(const float *a, const float *b, int n) {
    double dot=0,na=0,nb=0;
    for(int i=0;i<n;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return (float)(dot/(sqrt(fmax(na,1e-30))*sqrt(fmax(nb,1e-30))));
}

int main() {
    fprintf(stdout,"=== ggml_add WebGPU test ===\n"); fflush(stdout);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    fprintf(stdout,"CPU init OK\n"); fflush(stdout);

    fprintf(stdout,"GPU init...\n"); fflush(stdout);
    ggml_backend_t gpu = ggml_backend_webgpu_init();
    if(!gpu){ fprintf(stdout,"NO GPU\n"); ggml_backend_free(cpu); return 0; }
    fprintf(stdout,"GPU OK\n"); fflush(stdout);

    int n = 1024;
    std::vector<float> a(n), b(n);
    for(int i=0;i<n;i++){ a[i]=(float)rand()/RAND_MAX; b[i]=(float)rand()/RAND_MAX; }

    struct ggml_init_params gp = { 256*1024*1024, NULL, true };
    ggml_context *ctx = ggml_init(gp);
    ggml_tensor *ta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor *tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor *out = ggml_add(ctx, ta, tb);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, out);

    // CPU
    fprintf(stdout,"Alloc CPU buf...\n"); fflush(stdout);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    fprintf(stdout,"Set CPU data...\n"); fflush(stdout);
    ggml_backend_tensor_set(ta, a.data(), 0, n*sizeof(float));
    ggml_backend_tensor_set(tb, b.data(), 0, n*sizeof(float));
    fprintf(stdout,"Compute CPU...\n"); fflush(stdout);
    ggml_backend_graph_compute(cpu, graph);
    std::vector<float> ref(n);
    ggml_backend_tensor_get(out, ref.data(), 0, n*sizeof(float));
    fprintf(stdout,"CPU done μ=%.4f\n", ref[0]); fflush(stdout);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx);

    // GPU (new context)
    ctx = ggml_init(gp);
    ta   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    tb   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    out  = ggml_add(ctx, ta, tb);
    graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, out);

    fprintf(stdout,"Alloc GPU buf...\n"); fflush(stdout);
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    fprintf(stdout,"Set GPU data...\n"); fflush(stdout);
    ggml_backend_tensor_set(ta, a.data(), 0, n*sizeof(float));
    ggml_backend_tensor_set(tb, b.data(), 0, n*sizeof(float));
    fprintf(stdout,"Compute GPU...\n"); fflush(stdout);
    ggml_backend_graph_compute(gpu, graph);
    std::vector<float> gpu_res(n);
    ggml_backend_tensor_get(out, gpu_res.data(), 0, n*sizeof(float));
    fprintf(stdout,"GPU done μ=%.4f\n", gpu_res[0]); fflush(stdout);

    float cos = cosine(ref.data(), gpu_res.data(), n);
    float maxd = 0;
    for(int i=0;i<n;i++) maxd = fmaxf(maxd, fabsf(ref[i]-gpu_res[i]));
    fprintf(stdout,"cos=%.6f maxd=%.6f %s\n", cos, maxd, cos>0.9999f?"PASS":"FAIL"); fflush(stdout);

    ggml_backend_buffer_free(buf_gpu);
    ggml_free(ctx);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return cos>0.9999f?0:1;
}
