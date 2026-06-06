#pragma once

#include "ggml.h"
#include "vae_types.h"

namespace bonsai {

// Build the full Flux VAE decoder graph.
// x: input latent in NCHW (ggml: [W, H, 32, 1]) with values in latent space
// returns: output RGB image in NCHW (ggml: [W, H, 3, 1]) with raw decoder values
//          (matches Python's flux2_vae.Decoder.forward, no post-processing)
ggml_tensor * build_decoder_graph(ggml_context * ctx, const VAEWeights & w, ggml_tensor * x);

}  // namespace bonsai
