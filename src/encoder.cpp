#include "llama.h"
#include "ggml.h"

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <fstream>
#include <iomanip>
#include <sstream>

struct ExtractionState {
    std::set<int> target_layers;
    std::unordered_map<int, std::vector<float>> extracted_states;
    std::unordered_map<int, bool> layer_extracted;
};

bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;
    
    auto* state = static_cast<ExtractionState*>(user_data);
    if (!state) return true;

    std::string name = t->name;
    
    for (int layer : state->target_layers) {
        if (!state->layer_extracted[layer]) {
            std::string target_name_1 = "ffn_out-" + std::to_string(layer);
            std::string target_name_2 = "l_out-" + std::to_string(layer);
            
            if (name == target_name_1 || name == target_name_2) {
                size_t n_elements = ggml_nelements(t);
                state->extracted_states[layer].resize(n_elements);
                
                // Copy data from device (e.g., CUDA) to host
                ggml_backend_tensor_get(t, state->extracted_states[layer].data(), 0, n_elements * sizeof(float));
                state->layer_extracted[layer] = true;
                
                std::cout << "[INFO] Successfully extracted hidden state for layer " << layer 
                          << " (elements: " << n_elements << ")" << std::endl;
            }
        }
    }
    return true;
}

void save_tensor_to_bin(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Failed to open file for writing: " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    std::cout << "[INFO] Saved " << filename << " (" << data.size() * sizeof(float) << " bytes)" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [prompt] [--layers 9,18,27]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = (argc > 2 && argv[2][0] != '-') ? argv[2] : "Extract hidden states for bonsai.cpp pipeline.";
    
    std::set<int> target_layers = {9, 18, 27}; // Default for Qwen3-4B

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--layers" && i + 1 < argc) {
            target_layers.clear();
            std::string layers_str = argv[++i];
            std::stringstream ss(layers_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    target_layers.insert(std::stoi(token));
                } catch (...) {
                    std::cerr << "[WARNING] Invalid layer number: " << token << std::endl;
                }
            }
        } else if (arg[0] != '-' && prompt == "Extract hidden states for bonsai.cpp pipeline.") {
            prompt = arg;
        }
    }

    std::cout << "[INFO] Initializing llama backend..." << std::endl;
    llama_backend_init();

    std::cout << "[INFO] Loading model: " << model_path << std::endl;
    llama_model_params model_params = llama_model_default_params();
    // Force CUDA if available, otherwise CPU
    model_params.n_gpu_layers = 999; 
    
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "[ERROR] Failed to load model" << std::endl;
        llama_backend_free();
        return 1;
    }

    std::cout << "[INFO] Model loaded successfully. Creating context..." << std::endl;
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512; // Minimal context for single pass
    ctx_params.n_batch = 512;
    ctx_params.embeddings = true; // Ensure embeddings are computed
    
    // Set up the evaluation callback to intercept intermediate layers
    ExtractionState ext_state;
    ext_state.target_layers = target_layers;
    for (int layer : target_layers) {
        ext_state.layer_extracted[layer] = false;
    }
    
    ctx_params.cb_eval = eval_callback;
    ctx_params.cb_eval_user_data = &ext_state;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "[ERROR] Failed to create llama context" << std::endl;
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::cout << "[INFO] Tokenizing prompt..." << std::endl;
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(prompt.size() + 8, 0);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) {
        std::cerr << "[ERROR] Tokenization failed" << std::endl;
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    tokens.resize(n_tokens);

    std::cout << "[INFO] Running forward pass (" << n_tokens << " tokens)..." << std::endl;
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    
    int32_t res = llama_decode(ctx, batch);
    if (res != 0) {
        std::cerr << "[ERROR] llama_decode failed with code: " << res << std::endl;
    }

    std::cout << "[INFO] Forward pass complete. Checking extracted layers..." << std::endl;
    for (int layer : target_layers) {
        if (ext_state.layer_extracted[layer]) {
            std::string filename = "layer_" + std::to_string(layer) + "_hidden_state.bin";
            save_tensor_to_bin(filename, ext_state.extracted_states[layer]);
        } else {
            std::cerr << "[WARNING] Failed to extract hidden state for layer " << layer 
                      << ". The tensor naming might differ for this specific model architecture." << std::endl;
        }
    }

    std::cout << "[INFO] Destroying context and model to free memory immediately..." << std::endl;
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    std::cout << "[INFO] Extraction pipeline finished successfully." << std::endl;
    return 0;
}
