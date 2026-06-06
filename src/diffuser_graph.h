#pragma once

#include "ggml.h"

#include <vector>

#include "diffuser_types.h"
#include "b1_0_kernel.h"

namespace bonsai {

struct DiffuserGraph {
    ggml_tensor * img_in;
    ggml_tensor * txt_in;
    ggml_tensor * timestep;
    ggml_tensor * out;
    ggml_cgraph * graph;
    std::vector<B1LinearUserData> b1_ud;
};

DiffuserGraph build_diffuser_graph(
    ggml_context * ctx,
    const DiffuserParams & params,
    const DiffuserWeights & weights,
    int img_tokens,
    int txt_tokens,
    int batch,
    int n_threads);

} // namespace bonsai
