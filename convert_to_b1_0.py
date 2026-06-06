import os
import struct
import numpy as np
import torch
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType
import gguf.quants

GGML_TYPE_B1_0 = 100
BLOCK_SIZE = 32

gguf.quants.GGML_QUANT_SIZES[GGML_TYPE_B1_0] = (BLOCK_SIZE, 6)


def quantize_b1_0(tensor_np):
    """
    Convert a numpy tensor (FP32/FP16) into custom B1_0 blocks.
    B1_0 structure: [scale (float16: 2 bytes), bits (uint32: 4 bytes)] = 6 bytes per block of 32 weights.
    """
    assert tensor_np.shape[-1] % BLOCK_SIZE == 0, f"Dimension {tensor_np.shape[-1]} not divisible by {BLOCK_SIZE}"

    reshaped = tensor_np.reshape(*tensor_np.shape[:-1], -1, BLOCK_SIZE)

    scales = np.mean(np.abs(reshaped), axis=-1).astype(np.float16)

    bits = (reshaped >= 0).astype(np.uint8)

    packed_bits = np.packbits(bits, axis=-1, bitorder='little')

    num_blocks = reshaped.shape[1]
    out_shape = tensor_np.shape[:-1] + (num_blocks,)

    raw_blocks = np.empty(out_shape + (6,), dtype=np.uint8)

    scales_bytes = scales.view(np.uint8).reshape(out_shape + (2,))
    raw_blocks[..., 0:2] = scales_bytes

    raw_blocks[..., 2:6] = packed_bits

    return raw_blocks.flatten()

def map_tensor_name(pt_name):
    """Basic mapping from PyTorch names to GGML/GGUF naming convention"""
    name = pt_name
    name = name.replace("single_transformer_blocks.", "blk.")
    name = name.replace("double_stream_modulation", "double_stream")
    return name

def main():
    safetensors_path = "diffusion_pytorch_model.safetensors"
    output_gguf = "models/flux2_4b_1bit.gguf"
    
    os.makedirs("models", exist_ok=True)
    
    print(f"Opening {safetensors_path}...")
    
    writer = GGUFWriter(output_gguf, "flux.diffusion")
    writer.add_uint32("flux.vision.hidden_size", 3072)
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        total = len(keys)
        print(f"Converting {total} tensors...")
        
        for i, key in enumerate(keys):
            gguf_name = map_tensor_name(key)
            
            tensor_pt = f.get_tensor(key)
            tensor = tensor_pt.to(torch.float32).numpy()
            
            if len(tensor.shape) == 1:
                tensor_fp32 = tensor.astype(np.float32)
                writer.add_tensor(gguf_name, tensor_fp32, raw_dtype=GGMLQuantizationType.F32)
                print(f"[{i+1}/{total}] {gguf_name} -> F32 (1D)")
            else:
                raw_bytes = quantize_b1_0(tensor)
                
                byte_shape = tensor.shape[:-1] + (tensor.shape[-1] // BLOCK_SIZE * 6,)
                
                writer.add_tensor(gguf_name, raw_bytes, raw_shape=byte_shape, raw_dtype=GGML_TYPE_B1_0)
                print(f"[{i+1}/{total}] {gguf_name} -> B1_0 (2D) | shape {tensor.shape}")
                
    print("\nWriting GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"Done! GGUF saved to: {output_gguf}")

if __name__ == "__main__":
    main()
