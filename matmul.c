#include "matmul.h"
#include "device_picker.h"
#include "err_code.h"
#include "matrix_lib.h"

#define CL_KERNEL_FILE "matmul.cl"

// cette fonction trouve le plus grand diviseur de X dans l'intervalle [1;16]
int div(int X) {
  int div = 1;
  for (int i = 2; i <= 16; i++) {
    if (X % i == 0) {
      div = i;
    }
  }
  return div;
}

// Load an OpenCL kernel from file
char *readKernelFile(const char *filename, long *_size) {

  // Open the file
  FILE *file = fopen(filename, "r");
  if (!file) {
    printf("-- Error opening file %s\n", filename);
    exit(1);
  }

  // Get its size
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  rewind(file);

  // Read the kernel code as a string
  char *source = (char *)malloc((size + 1) * sizeof(char));
  fread(source, 1, size * sizeof(char), file);
  source[size] = '\0';
  fclose(file);

  // Save the size and return the source string
  *_size = (size + 1);
  return source;
}



int main(int argc, char *argv[]) {
  float *h_A;  // A matrix
  float *h_B;  // B matrix
  float *h_C;  // C = A*B matrix
  int M, N, K; // A[M][K], B[K][N], C[M][N]
  int A_size;  // number of elements in matrix A
  int B_size;  // number of elements in matrix B
  int C_size;  // number of elements in matrix C

  cl_mem d_a, d_b, d_c; // Matrices in device memory

  double start_time; // Starting time
  double run_time;   // timing data

  cl_int err;                // error code returned from OpenCL calls
  cl_device_id device;       // compute device id
  cl_context context;        // compute context
  cl_command_queue commands; // compute command queue
  cl_program program;        // compute program
  cl_kernel kernel;          // compute kernel

  // TODO-BLOC-DEBUT : Afin de tester votre programme, modifier les valeurs de
  // M,N et K pour des matrices non carrées
  M = 1030;
  N = 1028;
  K = 1032;
  // TODO-BLOC-FIN.
  
  A_size = M * K;
  B_size = K * N;
  C_size = M * N;

  h_A = (float *)malloc(A_size * sizeof(float));
  h_B = (float *)malloc(B_size * sizeof(float));
  h_C = (float *)malloc(C_size * sizeof(float));

  //--------------------------------------------------------------------------------
  // Create a context, queue and device.
  //--------------------------------------------------------------------------------

  cl_uint deviceIndex = 0;
  parseArguments(argc, argv, &deviceIndex);

  // Get list of devices
  cl_device_id devices[MAX_DEVICES];
  unsigned numDevices = getDeviceList(devices);

  // Check device index in range
  if (deviceIndex >= numDevices) {
    printf("Invalid device index (try '--list')\n");
    return EXIT_FAILURE;
  }

  device = devices[deviceIndex];

  char name[MAX_INFO_STRING];
  getDeviceName(device, name);
  printf("\nUsing OpenCL device: %s\n", name);

  // Create a compute context
  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  checkError(err, "Creating context");

  // Create a command queue
  commands = clCreateCommandQueue(context, device, 0, &err);
  checkError(err, "Creating command queue");

  //--------------------------------------------------------------------------------
  // Run sequential version on the host
  //--------------------------------------------------------------------------------

  initmat(M, N, K, h_A, h_B, h_C);

  printf("\n===== OpenMP, matrix mult (dot prod), order (%d,%d)x(%d,%d) on "
         "host CPU ======\n",
         M, K, K, N);
  for (int i = 0; i < COUNT; i++) {
    zero_mat(M, N, h_C);
    start_time = wtime();

    omp_mat_mul_sdot(M, N, K, h_A, h_B, h_C);

    run_time = wtime() - start_time;
    results(M, N, K, h_C, run_time);
  }

  //--------------------------------------------------------------------------------
  // Setup the buffers, initialize matrices, and write them into global memory
  //--------------------------------------------------------------------------------

  //  Reset A, B and C matrices (just to play it safe)
  initmat(M, N, K, h_A, h_B, h_C);

  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * A_size, h_A, &err);
  checkError(err, "Creating buffer d_a");

  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * B_size, h_B, &err);
  checkError(err, "Creating buffer d_b");

  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * C_size, NULL,
                       &err);
  checkError(err, "Creating buffer d_c");

  //--------------------------------------------------------------------------------
  // OpenCL matrix multiplication ... Naive
  //--------------------------------------------------------------------------------

  // Read the kernel file from disk
  long sizeSource;
  char *source = readKernelFile(CL_KERNEL_FILE, &sizeSource);

  // Create the comput program from the source buffer
  program =
      clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  checkError(err, "Creating program");

  // Build the program
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    printf("Error: Failed to build program executable!\n%s\n", err_code(err));
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
                          buffer, &len);
    printf("%s\n", buffer);
    return EXIT_FAILURE;
  }

  // Create the compute kernel from the program
  kernel = clCreateKernel(program, "mmul", &err);
  checkError(err, "Creating kernel");

  printf("\n===== OpenCL, matrix mult, C(i,j) per work item, order %d ======\n",
         N);

  // calcul de la taille du bloc ( <=16 )
  int BlocN = divisor(N);
  int BlocM = divisor(M);
  // Do the multiplication COUNT times
  for (int i = 0; i < COUNT; i++) {
    zero_mat(M, N, h_C);

    // TODO-BLOC-DEBUT : modifications à faire pour rendre le code compatible 
    // vec des matrices non carrées


    err =  clSetKernelArg(kernel, 0,       sizeof(int),    &N);
    err =  clSetKernelArg(kernel, 1,       sizeof(int),    &K);
    err |= clSetKernelArg(kernel, 2,       sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 3,       sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 4,       sizeof(cl_mem), &d_c);        
	err |= clSetKernelArg(kernel, 5, BlocN*BlocN*sizeof(cl_float), NULL);
    err |= clSetKernelArg(kernel, 6, BlocM*BlocM*sizeof(cl_float), NULL);
    checkError(err, "Setting kernel arguments");

    start_time = wtime();

    // Execute the kernel over the entire range of C matrix elements ...
    // computing a dot product for each element of the product matrix.
    const size_t global[2] = {N, M};
    const size_t local[2] = {BlocN, BlocM};

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0,
                                 NULL, NULL);

    checkError(err, "Enqueuing kernel");

    // TODO-BLOC-FIN.

    err = clFinish(commands);
    checkError(err, "Waiting for commands to finish");

    run_time = wtime() - start_time;

    err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * C_size,
                              h_C, 0, NULL, NULL);
    checkError(err, "Reading back buffer d_c");

    results(M, N, K, h_C, run_time);

  } // end for loop

  //--------------------------------------------------------------------------------
  // Clean up!
  //--------------------------------------------------------------------------------

  free(h_A);
  free(h_B);
  free(h_C);
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return EXIT_SUCCESS;
}
