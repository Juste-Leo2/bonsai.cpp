#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void f32_write(const char *path, const ggml_tensor *t) {
    FILE *fp=fopen(path,"wb"); fwrite(t->data,1,ggml_nbytes(t),fp); fclose(fp);
}
static ggml_tensor * col(ggml_context *ctx, ggml_tensor *b) {
    return ggml_cont(ctx, ggml_reshape_4d(ctx,b,b->ne[0],1,1,1));
}
static ggml_tensor * load_f32(ggml_context *ctx, const char *path, int n0, int n1) {
    ggml_tensor *t=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,n0,n1);
    t->data=malloc(n0*n1*sizeof(float));
    FILE *f=fopen(path,"rb"); fread(t->data,sizeof(float),n0*n1,f); fclose(f);
    return t;
}

int main(int argc, char **argv) {
    int H=3072, C=128, it=4096, tt=512, n_threads=4;
    for (int i=1;i<argc;i++) if(std::string(argv[i])=="--threads"&&i+1<argc) n_threads=std::stoi(argv[++i]);

    fprintf(stderr,"Init...\n");
    struct ggml_init_params cparams={256ULL*1024*1024,NULL,true};
    ggml_context *ctx=ggml_init(cparams);

    ggml_tensor *combined=load_f32(ctx,"combined.bin",H,it+tt);
    ggml_tensor *vec_n=load_f32(ctx,"vec_norm.bin",H,1);
    ggml_tensor *w_no=load_f32(ctx,"w_norm_out.bin",H,2*H);
    ggml_tensor *w_po=load_f32(ctx,"w_proj_out.bin",H,C);

    ggml_tensor *one_t=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1);
    ggml_tensor *ones=ggml_fill(ctx,one_t,1.0f);

    // Extract img tokens
    ggml_tensor *final_img=ggml_view_2d(ctx,combined,H,it,combined->nb[1],tt*H*sizeof(float));
    ggml_tensor *fn=ggml_norm(ctx,final_img,1e-6f);

    ggml_tensor *w_no2=ggml_reshape_2d(ctx,w_no,H,2*H);
    ggml_tensor *mod_raw=ggml_mul_mat(ctx,w_no2,vec_n);
    ggml_tensor *mod_s=ggml_view_2d(ctx,mod_raw,H,1,mod_raw->nb[1],0);
    ggml_tensor *mod_h=ggml_view_2d(ctx,mod_raw,H,1,mod_raw->nb[1],H*sizeof(float));

    fn=ggml_add(ctx,ggml_mul(ctx,fn,ggml_add(ctx,ggml_repeat(ctx,col(ctx,mod_s),fn),ones)),ggml_repeat(ctx,col(ctx,mod_h),fn));

    ggml_tensor *w_po2=ggml_reshape_2d(ctx,w_po,H,C);
    ggml_tensor *out=ggml_mul_mat(ctx,w_po2,fn);
    ggml_set_output(out);

    ggml_cgraph *graph=ggml_new_graph_custom(ctx,4096,false);
    ggml_build_forward_expand(graph,out);

    ggml_backend_t backend=ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend,n_threads);
    ggml_gallocr_t galloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(galloc,graph);
    ggml_gallocr_alloc_graph(galloc,graph);

    fprintf(stderr,"Computing...\n");
    ggml_backend_graph_compute(backend,graph);
    f32_write("out.bin",out);
    fprintf(stderr,"Done!\n");

    ggml_gallocr_free(galloc); ggml_backend_free(backend); ggml_free(ctx);
    return 0;
}
