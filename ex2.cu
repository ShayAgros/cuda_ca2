/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include <assert.h>
#include <string.h>

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////
#define IMG_DIMENSION 32
#define NREQUESTS 30000
#define THREADS_NR SQR(IMG_DIMENSION)

#define SHARED_MEMORY_SZ_PER_TB (256*2)

#define QUEUE_SLOTS_NR 10
#define KERNEL_MAX_REGISTERS 32

#define VALID_BIT_IX 0
#define IMGIX_BIT_IX 1
#define DATA_IX 1 + sizeof(int)

#define CELL_SIZE (1 + sizeof(int) + SQR(IMG_DIMENSION))
// Treat a one-dimensional array as a two dimensional array of type
// queue[QUEUE_SLOTS_NR][(1 + sizeof(int) + SQR(IMG_DIMENSION))]
#define QUEUE_IX(q, first_dim, second_dim) \
	q[(first_dim * CELL_SIZE) + second_dim]

typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

void process_image(uchar *img_in, uchar *img_out) {
    int histogram[256] = { 0 };
    for (int i = 0; i < SQR(IMG_DIMENSION); i++) {
        histogram[img_in[i]]++;
    }

    int cdf[256] = { 0 };
    int hist_sum = 0;
    for (int i = 0; i < 256; i++) {
        hist_sum += histogram[i];
        cdf[i] = hist_sum;
    }

    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    uchar map[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        int map_value = (float)(cdf[i] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[i] = (uchar)map_value;
    }

    for (int i = 0; i < SQR(IMG_DIMENSION); i++) {
        img_out[i] = map[img_in[i]];
    }
}

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

/* we'll use these to rate limit the request load */
struct rate_limit_t {
    double last_checked;
    double lambda;
    unsigned seed;
};

void rate_limit_init(struct rate_limit_t *rate_limit, double lambda, int seed) {
    rate_limit->lambda = lambda;
    rate_limit->seed = (seed == -1) ? 0 : seed;
    rate_limit->last_checked = 0;
}

int rate_limit_can_send(struct rate_limit_t *rate_limit) {
    if (rate_limit->lambda == 0) return 1;
    double now = get_time_msec() * 1e-3;
    double dt = now - rate_limit->last_checked;
    double p = dt * rate_limit->lambda;
    rate_limit->last_checked = now;
    if (p > 1) p = 1;
    double r = (double)rand_r(&rate_limit->seed) / RAND_MAX;
    return (p > r);
}

void rate_limit_wait(struct rate_limit_t *rate_limit) {
    while (!rate_limit_can_send(rate_limit)) {
        struct timespec t = {
            0,
            long(1. / (rate_limit->lambda * 1e-9) * 0.01)
        };
        nanosleep(&t, NULL);
    }
}

double distance_sqr_between_image_arrays(uchar *img_arr1, uchar *img_arr2) {
    double distance_sqr = 0;
    for (int i = 0; i < NREQUESTS * SQR(IMG_DIMENSION); i++) {
        distance_sqr += SQR(img_arr1[i] - img_arr2[i]);
    }
    return distance_sqr;
}

/* we won't load actual files. just fill the images with random bytes */
void load_images(uchar *images) {
    srand(0);
    for (int i = 0; i < NREQUESTS * SQR(IMG_DIMENSION); i++) {
        images[i] = rand() % 256;
    }
}

__device__ int arr_min(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int rhs, lhs;

    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            rhs = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            lhs = arr[tid];
            if (rhs != 0) {
                if (lhs == 0)
                    arr[tid] = rhs;
                else
                    arr[tid] = min(arr[tid], rhs);
            }
        }
        __syncthreads();
    }

    int ret = arr[arr_size - 1];
    return ret;
}

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;

    for (int stride = 1; stride < min(blockDim.x, arr_size); stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__global__ void gpu_process_image(uchar *in, uchar *out) {
    __shared__ int histogram[SHARED_MEMORY_SZ_PER_TB/2];
    __shared__ int hist_min[SHARED_MEMORY_SZ_PER_TB/2];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        hist_min[tid] = histogram[tid];
    }
    __syncthreads();

    int cdf_min = arr_min(hist_min, 256);

    __shared__ uchar map[256];
    if (tid < 256) {
        int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[tid] = (uchar)map_value;
    }

    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
        out[i] = map[in[i]];
    }
    return;
}

__device__ void gpu_queues_process_image(uchar *in, uchar *out) {
    __shared__ int histogram[SHARED_MEMORY_SZ_PER_TB/2];
    __shared__ int hist_min[SHARED_MEMORY_SZ_PER_TB/2];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        hist_min[tid] = histogram[tid];
    }
    __syncthreads();

    int cdf_min = arr_min(hist_min, 256);

    __shared__ uchar map[256];
    if (tid < 256) {
        int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[tid] = (uchar)map_value;
    }

    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
        out[i] = map[in[i]];
    }
    return;
}

// an invalid queue index has VALID_BIT set to 0. Otherwise it's valid
__global__ void gpu_tb_server(uchar *rqs, uchar *sqs, volatile bool *terminate) {
	__shared__ int rq_ix;
	__shared__ int out_ix;
	__shared__ uchar *rq;
	__shared__ uchar *sq;
	int tid = threadIdx.x;

	// find local queue for thread block
	rq = rqs + blockIdx.x * (QUEUE_SLOTS_NR * CELL_SIZE);
	sq = sqs + blockIdx.x * (QUEUE_SLOTS_NR * CELL_SIZE);

	while (true) {

		if (tid == 0) {
			for (int i = (rq_ix +1) % QUEUE_SLOTS_NR; ; i = (i+1) % QUEUE_SLOTS_NR) {
				if (QUEUE_IX(rq,i, VALID_BIT_IX) == 1) {
					rq_ix = i;
					break;
				}
				if (*terminate) {
					rq_ix = -1;
					break;
				}
			}
		}

		__syncthreads();
		// received terminate
		if (rq_ix < 0)
			return;
		
		gpu_queues_process_image(&QUEUE_IX(rq,rq_ix,DATA_IX), &QUEUE_IX(sq,out_ix,DATA_IX));
		__syncthreads();

		// send processed image back
		if (tid == 0) {
			// we move the img ix number from RQ to SQ
			memcpy(&QUEUE_IX(sq,out_ix, IMGIX_BIT_IX), &QUEUE_IX(rq,rq_ix,IMGIX_BIT_IX), sizeof(int));

			__threadfence_system();
			// validate entry
			QUEUE_IX(sq,out_ix,VALID_BIT_IX) = 1;

			// invalidate rq cell
			QUEUE_IX(rq,rq_ix,VALID_BIT_IX) = 0;
			__threadfence_system();

			// choose new out_ix
			for (int i = (out_ix +1) % QUEUE_SLOTS_NR; !(*terminate); i = (i+1) % QUEUE_SLOTS_NR) {
				if (QUEUE_IX(sq,i,0) == 0) {
					out_ix = i;
					break;
					
				}
			}
		}

		__threadfence_system();
		__syncthreads();
	
	}
}

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}

int get_threadblock_number_device(int threads_per_block__nr, int device_number) {
	cudaDeviceProp prop;
	int max_threads_tb_nr;
	int max_shared_mem_tb_nr;
	int max_regs_tb_nr;
	int tb_nr = 0;

	CUDA_CHECK(cudaGetDeviceProperties(&prop, device_number));
	
	max_threads_tb_nr = prop.maxThreadsPerMultiProcessor / threads_per_block__nr;
	max_shared_mem_tb_nr = prop.sharedMemPerMultiprocessor / SHARED_MEMORY_SZ_PER_TB;
	max_regs_tb_nr = prop.regsPerMultiprocessor / (KERNEL_MAX_REGISTERS * threads_per_block__nr);

	tb_nr = (max_threads_tb_nr > max_shared_mem_tb_nr) ? max_shared_mem_tb_nr : max_threads_tb_nr;
	tb_nr = (tb_nr > max_regs_tb_nr) ? max_regs_tb_nr : tb_nr;

	return tb_nr * prop.multiProcessorCount;
}

int get_threadblock_number(int threads_nr) {
	int min, devices_nr;

	CUDA_CHECK(cudaGetDeviceCount(&devices_nr));

	if (devices_nr <= 0)
		return 0;

	min = get_threadblock_number_device(threads_nr, 0);
	for (int i = 1; i < devices_nr; i++) {
		int cur = get_threadblock_number_device(threads_nr, i);
		min = (cur < min) ? cur : min;
	}

	return min;
}

void checkCompletedRequests(cudaStream_t streams[64], double *req_t_end,
							int *free_stream, int stream2img[64])
{
	for (int i = 0; i < 64; i++)
		if ( cudaStreamQuery(streams[i]) == cudaSuccess) {
			if (*free_stream < 0)
				*free_stream = i;

			if (stream2img[i] >= 0) {
				req_t_end[stream2img[i]] = get_time_msec();
				stream2img[i] = -1;
			}
		}
}

void initialize_terminate_variable(bool **cpu_terminate, bool **gpu_terminate)
{
    CUDA_CHECK( cudaHostAlloc(cpu_terminate, sizeof(bool), 0) );

	CUDA_CHECK(cudaHostGetDevicePointer(gpu_terminate, *cpu_terminate, 0 ));
}

void initialize_gpu_tb_server_queues(uchar **cpu_rqs, uchar **cpu_sqs,
                                     uchar **gpu_rqs, uchar **gpu_sqs,
                                     int tb_nr)
{
    CUDA_CHECK( cudaHostAlloc(cpu_rqs, tb_nr * (QUEUE_SLOTS_NR * CELL_SIZE), 0) );
    CUDA_CHECK( cudaHostAlloc(cpu_sqs, tb_nr * (QUEUE_SLOTS_NR * CELL_SIZE), 0) );

    memset(*cpu_rqs, 0, tb_nr * (QUEUE_SLOTS_NR * CELL_SIZE));
    memset(*cpu_sqs, 0, tb_nr * (QUEUE_SLOTS_NR * CELL_SIZE));

	__sync_synchronize();

	CUDA_CHECK(cudaHostGetDevicePointer(gpu_rqs, *cpu_rqs, 0 ));
	CUDA_CHECK(cudaHostGetDevicePointer(gpu_sqs, *cpu_sqs, 0 ));
}

enum {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};
int main(int argc, char *argv[]) {

    int mode = -1;
    int threads_queue_mode = -1; /* valid only when mode = queue */
    double load = 0;
    if (argc < 3) print_usage_and_die(argv[0]);

    if (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_STREAMS;
        load = atof(argv[2]);
    } else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        mode = PROGRAM_MODE_QUEUE;
        threads_queue_mode = atoi(argv[2]);
        load = atof(argv[3]);
    } else {
        print_usage_and_die(argv[0]);
    }

    uchar *images_in; /* we concatenate all images in one huge array */
    uchar *images_out;
    CUDA_CHECK( cudaHostAlloc(&images_in, NREQUESTS * SQR(IMG_DIMENSION), 0) );
    CUDA_CHECK( cudaHostAlloc(&images_out, NREQUESTS * SQR(IMG_DIMENSION), 0) );

    load_images(images_in);
    double t_start, t_finish;

    /* using CPU */
    printf("\n=== CPU ===\n");
    t_start  = get_time_msec();
    for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx)
        process_image(&images_in[img_idx * SQR(IMG_DIMENSION)], &images_out[img_idx * SQR(IMG_DIMENSION)]);
    t_finish = get_time_msec();
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

    double total_distance = 0;

    /* using GPU task-serial.. just to verify the GPU code makes sense */
    printf("\n=== GPU Task Serial ===\n");

    uchar *images_out_from_gpu;
    CUDA_CHECK( cudaHostAlloc(&images_out_from_gpu, NREQUESTS * SQR(IMG_DIMENSION), 0) );

    do {
        uchar *gpu_image_in, *gpu_image_out;
        CUDA_CHECK(cudaMalloc(&gpu_image_in, SQR(IMG_DIMENSION)));
        CUDA_CHECK(cudaMalloc(&gpu_image_out, SQR(IMG_DIMENSION)));

        t_start = get_time_msec();
        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
            CUDA_CHECK(cudaMemcpy(gpu_image_in, &images_in[img_idx * SQR(IMG_DIMENSION)], SQR(IMG_DIMENSION), cudaMemcpyHostToDevice));
            gpu_process_image<<<1, THREADS_NR>>>(gpu_image_in, gpu_image_out);
            CUDA_CHECK(cudaMemcpy(&images_out_from_gpu[img_idx * SQR(IMG_DIMENSION)], gpu_image_out, SQR(IMG_DIMENSION), cudaMemcpyDeviceToHost));
        }
        total_distance += distance_sqr_between_image_arrays(images_out, images_out_from_gpu);
        CUDA_CHECK(cudaDeviceSynchronize());
        t_finish = get_time_msec();
        printf("distance from baseline %lf (should be zero)\n", total_distance);
        printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);

        CUDA_CHECK(cudaFree(gpu_image_in));
        CUDA_CHECK(cudaFree(gpu_image_out));
    } while (0);

    /* now for the client-server part */
    printf("\n=== Client-Server ===\n");
    double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_start, 0, NREQUESTS * sizeof(double));

    double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
    memset(req_t_end, 0, NREQUESTS * sizeof(double));

    struct rate_limit_t rate_limit;
    rate_limit_init(&rate_limit, load, 0);

    int stream2img[64];

    // initialize stream2img
    for (int i = 0; i < 64; i++)
    	stream2img[i] = -1;

    /* TODO allocate / initialize memory, streams, etc... */
	uchar *gpu_image_in, *gpu_image_out;
    CUDA_CHECK(cudaMemset(images_out_from_gpu, 0, NREQUESTS * SQR(IMG_DIMENSION)));

    cudaStream_t streams[64];
    int free_stream;

    for(int i = 0; i < 64; i++) {
    	cudaStreamCreate(&streams[i]);
    }


    double ti = get_time_msec();
    if (mode == PROGRAM_MODE_STREAMS) {

		// allocate the pointers we're working with
		CUDA_CHECK(cudaMalloc(&gpu_image_in, NREQUESTS * SQR(IMG_DIMENSION)));
		CUDA_CHECK(cudaMalloc(&gpu_image_out, NREQUESTS * SQR(IMG_DIMENSION)));

        for (int img_idx = 0; img_idx < NREQUESTS; ++img_idx) {
        	free_stream = -1;

            /* TODO query (don't block) streams for any completed requests.
             * update req_t_end of completed requests
             */
			checkCompletedRequests(streams, req_t_end, &free_stream, stream2img);
			if (free_stream < 0)
				continue;

			if (!rate_limit_can_send(&rate_limit)) {
                  --img_idx;
                  continue;
            }

            req_t_start[img_idx] = get_time_msec();

            stream2img[free_stream] = img_idx;

            /* TODO place memcpy's and kernels in a stream */
            CUDA_CHECK(cudaMemcpyAsync(&gpu_image_in[img_idx * SQR(IMG_DIMENSION)],
									   &images_in[img_idx * SQR(IMG_DIMENSION)],
									   SQR(IMG_DIMENSION), cudaMemcpyHostToDevice,
									   streams[free_stream]));

            gpu_process_image<<<1, THREADS_NR, 0, streams[free_stream]>>>(&gpu_image_in[img_idx * SQR(IMG_DIMENSION)],
            														&gpu_image_out[img_idx * SQR(IMG_DIMENSION)]);

			CUDA_CHECK(cudaMemcpyAsync(&images_out_from_gpu[img_idx * SQR(IMG_DIMENSION)],
									   &gpu_image_out[img_idx * SQR(IMG_DIMENSION)],
									   SQR(IMG_DIMENSION),
									   cudaMemcpyDeviceToHost, streams[free_stream]));

        }

        // freeing memory
        CUDA_CHECK(cudaFree(gpu_image_in));
        CUDA_CHECK(cudaFree(gpu_image_out));

        /* TODO now make sure to wait for all streams to finish */
        int done = 0, ix = 0;

		// count all pending jobs
		for (int j = 0; j < 64; j++) {
			done += stream2img[j] >= 0;
		}

        // busy-wait on jobs till everyone finishes
        while (done > 0) {

        	if (stream2img[ix] >= 0 && cudaStreamQuery(streams[ix]) == cudaSuccess) {
				req_t_end[stream2img[ix]] = get_time_msec();
				done--;
				stream2img[ix] = -1;
        	}
        	ix = (ix + 1) % 64;
        }

    } else if (mode == PROGRAM_MODE_QUEUE) {
        // TODO launch GPU consumer-producer kernel
		int tb_nr = get_threadblock_number(threads_queue_mode);
		uchar *cpu_rqs, *gpu_rqs;
		uchar *cpu_sqs, *gpu_sqs;
		bool *cpu_terminate, *gpu_terminate;
		int next_rq = 0;
		int equalized_img_ix;
		int in_process_imgs = 0;
		bool img_sent = false;

		initialize_gpu_tb_server_queues(&cpu_rqs, &cpu_sqs, &gpu_rqs, &gpu_sqs, tb_nr);
		initialize_terminate_variable(&cpu_terminate,&gpu_terminate);

		// set program running
		*cpu_terminate = false;
		__sync_synchronize();

		// launch  servers
		gpu_tb_server <<< tb_nr, threads_queue_mode >>> (gpu_rqs, gpu_sqs, gpu_terminate);

        for (int img_idx = 0; img_idx < NREQUESTS;) {
            /* TODO check producer consumer queue for any responses.
             * don't block. if no responses are there we'll check again in the next iteration
             * update req_t_end of completed requests
             */
            for (int tb_ix = 0; tb_ix < tb_nr; tb_ix++) {
            	uchar *cpu_sq = cpu_sqs + tb_ix * (QUEUE_SLOTS_NR * CELL_SIZE);

            	for (int sq_ix = 0; sq_ix < QUEUE_SLOTS_NR; sq_ix++)
					// found a processed image
					if (QUEUE_IX(cpu_sq, sq_ix, VALID_BIT_IX) == 1) {
						// get processed image index
						memcpy(&equalized_img_ix, &QUEUE_IX(cpu_sq, sq_ix, IMGIX_BIT_IX), sizeof(int));
						// copy processed image to mode output
						memcpy(&images_out_from_gpu[equalized_img_ix * SQR(IMG_DIMENSION)], &QUEUE_IX(cpu_sq, sq_ix, DATA_IX), SQR(IMG_DIMENSION));
						// invalidate cell
						QUEUE_IX(cpu_sq, sq_ix, VALID_BIT_IX) = 0;

						req_t_end[equalized_img_ix] = get_time_msec();
						in_process_imgs--;
					}
			}
			__sync_synchronize();

            /*rate_limit_wait(&rate_limit);*/
			if (!rate_limit_can_send(&rate_limit))
                  continue;

            req_t_start[img_idx] = get_time_msec();

			img_sent = false;
            /* TODO push task to queue */
			// we start looking for a free spot in the the queue next
			// to the one we filled in the previous iteration
            for (int i = 0, tb_ix = next_rq; i < tb_nr && !img_sent; i++, tb_ix = (tb_ix +1) % tb_nr) {
				uchar *cpu_rq = cpu_rqs + tb_ix * (QUEUE_SLOTS_NR * CELL_SIZE);

            	for (int rq_ix = 0; rq_ix < QUEUE_SLOTS_NR; rq_ix++)
					if (QUEUE_IX(cpu_rq, rq_ix, VALID_BIT_IX) == 0) {

						// copy image to rq for processing
						memcpy(&QUEUE_IX(cpu_rq, rq_ix, DATA_IX), &images_in[img_idx * SQR(IMG_DIMENSION)], SQR(IMG_DIMENSION));
						// copy image's index
						memcpy(&QUEUE_IX(cpu_rq, rq_ix, IMGIX_BIT_IX), &img_idx, sizeof(int));
						// validate rq index
						QUEUE_IX(cpu_rq, rq_ix, VALID_BIT_IX) = 1;
						__sync_synchronize();

						// try a different rq next time
						next_rq = (next_rq + 1) % tb_nr;
						// increase "in-flight" images
						in_process_imgs++;
						img_idx++;
						img_sent = true;
						break;
					}
            }

        }

		// we wait for all images to process
		while (in_process_imgs > 0) {
            for (int tb_ix = 0; tb_ix < tb_nr; tb_ix++) {
            	uchar *cpu_sq = cpu_sqs + tb_ix * (QUEUE_SLOTS_NR * CELL_SIZE);

            	for (int sq_ix = 0; sq_ix < QUEUE_SLOTS_NR; sq_ix ++)
					// found a processed image
					if (QUEUE_IX(cpu_sq, sq_ix, VALID_BIT_IX) == 1) {
						// get processed image index
						memcpy(&equalized_img_ix, &QUEUE_IX(cpu_sq, sq_ix, IMGIX_BIT_IX), sizeof(int));
						// copy processed image to mode output
						memcpy(&images_out_from_gpu[equalized_img_ix * SQR(IMG_DIMENSION)],
							   &QUEUE_IX(cpu_sq, sq_ix, DATA_IX),
							   SQR(IMG_DIMENSION));
						// invalidate cell
						QUEUE_IX(cpu_sq, sq_ix, VALID_BIT_IX) = 0;

						req_t_end[equalized_img_ix] = get_time_msec();
						in_process_imgs--;
					}
			}
			__sync_synchronize();
		}

		// close open threads
		*cpu_terminate = true;
		__sync_synchronize();
        CUDA_CHECK(cudaDeviceSynchronize());


        CUDA_CHECK(cudaFreeHost(cpu_rqs));
        CUDA_CHECK(cudaFreeHost(cpu_sqs));
        CUDA_CHECK(cudaFreeHost(cpu_terminate));
        /* TODO wait until you have responses for all requests */

    } else {
        assert(0);
    }
    double tf = get_time_msec();

    total_distance = distance_sqr_between_image_arrays(images_out, images_out_from_gpu);
    double avg_latency = 0;
    for (int i = 0; i < NREQUESTS; i++) {
        avg_latency += (req_t_end[i] - req_t_start[i]);
    }
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", load);
    if (mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", threads_queue_mode);
    printf("distance from baseline %lf (should be zero)\n", total_distance);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

	CUDA_CHECK(cudaFreeHost(images_in));
	CUDA_CHECK(cudaFreeHost(images_out));
    return 0;
}
