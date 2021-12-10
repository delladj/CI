__kernel void mat_mul(
const int N, const int K,
  __global float *A,
  __global float *B,
  __global float *C,
  __local float *Awork,
  __local float *Bwork) {
  
  int i = get_global_id(0);
  int j = get_global_id(1);
  int ib = get_group_id(0);
  int jb = get_group_id(1);
  int il = get_local_id(0);
  int jl = get_local_id(1);
  int BS = get_local_size(0);
  int Abase = ib * BS * K;
  int Bbase = jb * BS;
  int Binc = N * BS;
  float acc = 0.0f;
  for (int k = 0; k < K; k += BS) {
    Awork[il*BS+jl] = A[Abase+il*K+jl];
    Bwork[il*BS+jl] = B[Bbase+il*N+jl];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kc = 0; kc < BS; kc++)
      acc += Awork[il*BS+kc] * Bwork[kc*BS+jl];
    barrier(CLK_LOCAL_MEM_FENCE);
    Abase += BS; Bbase += Binc;
  }
  C[i*N+j] = acc;
}
