#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "diffuser_types.h"
#include "b1_0_kernel.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace bonsai;
bool bonsai::g_bonsai_debug = false;

static void f32_write(const char *path, const ggml_tensor *t) {
    FILE *fp = fopen(path, "wb"); fwrite(t->data, 1, ggml_nbytes(t), fp); fclose(fp);
}
static ggml_tensor * column_1d(ggml_context * ctx, ggml_tensor * b) {
    ggml_tensor * v = ggml_reshape_4d(ctx, b, b->ne[0], 1, 1, 1);
    return ggml_cont(ctx, v);
}
static ggml_tensor * load_f32(ggml_context *ctx, const char *path, int ne0, int ne1) {
    ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    t->data = malloc(ne0*ne1*sizeof(float));
    FILE *f=fopen(path,"rb"); if(!f) abort();
    fread(t->data,sizeof(float),ne0*ne1,f); fclose(f);
    return t;
}
static ggml_tensor * load_f32_1d(ggml_context *ctx, const char *path, int n) {
    ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    t->data = malloc(n*sizeof(float));
    FILE *f=fopen(path,"rb"); if(!f) abort();
    fread(t->data,sizeof(float),n,f); fclose(f);
    return t;
}
static ggml_tensor * load_b1(ggml_context *ctx, const char *path, int in_dim, int out_dim) {
    int nbytes = out_dim * (in_dim/32) * 6;
    ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, nbytes, 1);
    t->data = malloc(nbytes);
    FILE *f=fopen(path,"rb"); if(!f) abort();
    fread(t->data,1,nbytes,f); fclose(f);
    return t;
}

struct ModTriple { ggml_tensor *shift, *scale, *gate; };

static ModTriple split_mod_single(ggml_context *ctx, ggml_tensor *mod, int H) {
    ModTriple r;
    r.shift = ggml_view_2d(ctx, mod, H, 1, mod->nb[1], 0);
    r.scale = ggml_view_2d(ctx, mod, H, 1, mod->nb[1], 1*H*sizeof(float));
    r.gate  = ggml_view_2d(ctx, mod, H, 1, mod->nb[1], 2*H*sizeof(float));
    return r;
}
static ggml_tensor * rms_norm_qk(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, int head_dim) {
    int n_heads=x->ne[1], seq=x->ne[2];
    x=ggml_cont(ctx,ggml_reshape_3d(ctx,x,head_dim,n_heads,seq));
    x=ggml_rms_norm(ctx,x,1e-6f);
    ggml_tensor *wv=ggml_reshape_3d(ctx,w,head_dim,1,1);
    x=ggml_mul(ctx,x,ggml_repeat(ctx,wv,x));
    return ggml_cont(ctx,x);
}

int main(int argc, char **argv) {
    int H=3072, C=128, ctx_dim=7680, n_heads=24, head_dim=128, it=4096, tt=512, mlp_hd=H*3, total_t=it+tt;
    int n_threads=4;
    for (int i=1;i<argc;i++) { if(std::string(argv[i])=="--threads"&&i+1<argc) n_threads=std::stoi(argv[++i]); }

    fprintf(stderr,"Init context...\n");
    struct ggml_init_params cparams={256ULL*1024*1024,NULL,true};
    ggml_context *ctx=ggml_init(cparams);

    // Load inputs
    ggml_tensor *combined=load_f32(ctx,"combined.bin",H,total_t);
    ggml_tensor *mod_single=load_f32(ctx,"mod_single.bin",1,3*H);
    ggml_tensor *cos_c=load_f32(ctx,"cos_combined.bin",64,total_t);
    ggml_tensor *sin_c=load_f32(ctx,"sin_combined.bin",64,total_t);

    // Load weights
    B1Weights sb_qkv_mlp={load_b1(ctx,"w_attn.to_qkv_mlp_proj.weight",H,3*H+mlp_hd*2),H,3*H+mlp_hd*2};
    B1Weights sb_to_out={load_b1(ctx,"w_attn.to_out.weight",H+mlp_hd,H),H+mlp_hd,H};
    ggml_tensor *n_q=load_f32_1d(ctx,"w_attn.norm_q.weight",head_dim);
    ggml_tensor *n_k=load_f32_1d(ctx,"w_attn.norm_k.weight",head_dim);

    auto sm=split_mod_single(ctx,mod_single,H);

    std::vector<B1LinearUserData> b1_ud; b1_ud.reserve(4);
    std::vector<Rope2DUserData> rod; rod.reserve(2);
    auto b1=[&](ggml_tensor *a,B1Weights &w){b1_ud.push_back({0x31423142,w.in_dim,w.out_dim});return b1_linear(ctx,a,w,n_threads,b1_ud.back());};

    // ── Build single block ──
    ggml_tensor *one_t=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1);
    ggml_tensor *ones=ggml_fill(ctx,one_t,1.0f);

    ggml_tensor *xn=ggml_norm(ctx,combined,1e-6f);
    xn=ggml_add(ctx,ggml_mul(ctx,xn,ggml_add(ctx,ggml_repeat(ctx,column_1d(ctx,ggml_cont(ctx,ggml_view_1d(ctx,sm.scale,H,0))),xn),ones)),ggml_repeat(ctx,column_1d(ctx,ggml_cont(ctx,ggml_view_1d(ctx,sm.shift,H,0))),xn));

    ggml_tensor *pa=b1(xn,sb_qkv_mlp);
    int qkv_dim=3*H;
    ggml_tensor *qkv_t=ggml_view_2d(ctx,pa,qkv_dim,total_t,pa->nb[1],0);
    ggml_tensor *mlp_t=ggml_view_2d(ctx,pa,mlp_hd*2,total_t,pa->nb[1],qkv_dim*sizeof(float));

    // split_qkv
    auto split_qkv_=[&](ggml_tensor *qkv_tensor){
        auto vh=[&](ggml_tensor *t,int off){return ggml_view_2d(ctx,t,H,total_t,t->nb[1],off*H*sizeof(float));};
        ggml_tensor *qt=vh(qkv_tensor,0), *kt=vh(qkv_tensor,1), *vt=vh(qkv_tensor,2);
        auto rh=[&](ggml_tensor *t){return ggml_cont(ctx,ggml_reshape_3d(ctx,ggml_cont(ctx,t),head_dim,n_heads,total_t));};
        struct QKV { ggml_tensor *q,*k,*v; };
        return QKV{rh(qt),rh(kt),rh(vt)};
    };
    auto qkv=split_qkv_(qkv_t);

    qkv.q=rms_norm_qk(ctx,qkv.q,n_q,head_dim);
    qkv.k=rms_norm_qk(ctx,qkv.k,n_k,head_dim);

    // RoPE
    ggml_tensor *qr=nullptr,*kr=nullptr;
    rod.push_back({0x524F5045,head_dim,n_heads,total_t}); qr=rope_2d_fwd(ctx,qkv.q,cos_c,sin_c,rod.back());
    rod.push_back({0x524F5045,head_dim,n_heads,total_t}); kr=rope_2d_fwd(ctx,qkv.k,cos_c,sin_c,rod.back());
    qkv.q=qr; qkv.k=kr;

    // Attention
    ggml_tensor *q3=ggml_cont(ctx,ggml_permute(ctx,qkv.q,0,2,1,3));
    ggml_tensor *k3=ggml_cont(ctx,ggml_permute(ctx,qkv.k,0,2,1,3));
    ggml_tensor *v3=ggml_cont(ctx,ggml_permute(ctx,qkv.v,0,2,1,3));
    float scale=1.0f/sqrtf((float)head_dim);

    ggml_tensor *s_attn;
#if 1
    ggml_tensor *k3f=k3, *v3f=v3;
    if(k3f->type==GGML_TYPE_F32) k3f=ggml_cast(ctx,k3f,GGML_TYPE_F16);
    if(v3f->type==GGML_TYPE_F32) v3f=ggml_cast(ctx,v3f,GGML_TYPE_F16);
    s_attn=ggml_flash_attn_ext(ctx,q3,k3f,v3f,nullptr,scale,0.0f,0.0f);
    ggml_flash_attn_ext_set_prec(s_attn,GGML_PREC_F32);
    s_attn=ggml_reshape_2d(ctx,s_attn,H,total_t);
#else
    ggml_tensor *sc=ggml_mul_mat(ctx,k3,q3); sc=ggml_scale_inplace(ctx,sc,scale);
    sc=ggml_soft_max_inplace(ctx,sc);
    ggml_tensor *v3t=ggml_cont(ctx,ggml_permute(ctx,v3,1,0,2,3));
    s_attn=ggml_mul_mat(ctx,v3t,sc);
    s_attn=ggml_cont(ctx,ggml_permute(ctx,s_attn,0,2,1,3));
    s_attn=ggml_reshape_2d(ctx,s_attn,H,total_t);
#endif

    // MLP: swiglu
    ggml_tensor *smlp=ggml_swiglu(ctx,mlp_t);
    ggml_tensor *scat=ggml_concat(ctx,s_attn,smlp,0);
    ggml_tensor *sout=b1(scat,sb_to_out);

    combined=ggml_add(ctx,combined,ggml_mul(ctx,ggml_repeat(ctx,column_1d(ctx,ggml_cont(ctx,ggml_view_1d(ctx,sm.gate,H,0))),sout),sout));

    ggml_set_output(combined);

    // Compute
    fprintf(stderr,"Building graph...\n");
    ggml_cgraph *graph=ggml_new_graph_custom(ctx,65536,false);
    ggml_build_forward_expand(graph,combined);

    ggml_backend_t backend=ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend,n_threads);
    ggml_gallocr_t galloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(galloc,graph);
    ggml_gallocr_alloc_graph(galloc,graph);

    fprintf(stderr,"Computing...\n");
    ggml_backend_graph_compute(backend,graph);
    fprintf(stderr,"Done!\n");

    fprintf(stderr,"combined: [%lld,%lld] %lld bytes\n",(long long)combined->ne[0],(long long)combined->ne[1],(long long)ggml_nbytes(combined));
    f32_write("combined_out.bin",combined);

    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
