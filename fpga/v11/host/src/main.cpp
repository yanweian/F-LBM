
#include <vector>
#include <cmath>

#include "funcs.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;
using namespace std;

// Kernels.
enum Kernel
{
    KERNEL_LBM = 0,
    KERNEL_COLLISION,
    KERNEL_EDGE,
    KERNEL_WRITEU,
    KERNEL_COUNT
};

// Buffers.
enum Buffer
{
    BUFFER_F = 0,
    BUFFER_U,
    BUFFER_F_NEXT,
    BUFFER_COUNT
};

// OpenCL Runtime Params
cl_platform_id platform;
cl_device_id device; // device used
cl_context context;
cl_command_queue queue[KERNEL_COUNT]; // parallel queue array
cl_program program;                   // compile aocx file to program
cl_kernel kernel[KERNEL_COUNT];       // kernel array
cl_mem buffer[BUFFER_COUNT];          // mem object to exchange data between host and device
// data array
scoped_aligned_ptr<Float> rho;
scoped_aligned_ptr<Float2> u;
scoped_aligned_ptr<Float2> u_fpga;

scoped_aligned_ptr<Float> f;
scoped_aligned_ptr<Float> f_new;
scoped_aligned_ptr<Float> f_new_ref;

// Gnuplot gp; //gnuplot
int looptime = 1;
void output(int i, Float2 *u)
{
    ostringstream name;
    name << "cavity_" << i << ".dat";
    ofstream out(name.str().c_str());
    out << "Title= \"LBM Lid Driven Flow\"\n"
        << "VARIABLES= \"X\",\"Y\",\"U\",\"V\"\n"
        << "ZONE T= \"BOX\",I= " << nx
        << ",J= " << ny << ",F= POINT" << endl;
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            out << double(x) / Lx << " " << double(y) / Lx << " " << u[idx(x, y)].x << " " << u[idx(x, y)].y << endl;
        }
    }
}
void run()
{
    char flag = 0;
    cl_int status;
    // ======Set kernel arguments=========
    // 第一,二个参数是ping-pong buffer
    unsigned argi = 1;
    status = clSetKernelArg(kernel[KERNEL_LBM], argi++, sizeof(nx), &nx);
    status = clSetKernelArg(kernel[KERNEL_LBM], argi++, sizeof(ny), &ny);
    status = clSetKernelArg(kernel[KERNEL_LBM], argi++, sizeof(float), &u1.x);
    status = clSetKernelArg(kernel[KERNEL_LBM], argi++, sizeof(float), &u1.y);

    argi = 1;
    // __kernel void collision (__global float * restrict f_next,float tau,int nx,int ny,char flag)
    status = clSetKernelArg(kernel[KERNEL_COLLISION], argi++, sizeof(tau_f), &tau_f);
    status = clSetKernelArg(kernel[KERNEL_COLLISION], argi++, sizeof(nx), &nx);
    status = clSetKernelArg(kernel[KERNEL_COLLISION], argi++, sizeof(ny), &ny);

    argi = 1;
    // __kernel void edge (__global float * restrict f_next,int nx)
    status = clSetKernelArg(kernel[KERNEL_EDGE], argi++, sizeof(nx), &nx);

    argi = 0;
    //__kernel void writeU (__global Float2 * restrict u,int cell_num)
    status = clSetKernelArg(kernel[KERNEL_WRITEU], argi++, sizeof(cl_mem), &buffer[BUFFER_U]);
    status = clSetKernelArg(kernel[KERNEL_WRITEU], argi++, sizeof(nx), &nx);
    status = clSetKernelArg(kernel[KERNEL_WRITEU], argi++, sizeof(ncells), &ncells);

    // ======write buffer from host to device====
    status = clEnqueueWriteBuffer(queue[KERNEL_LBM], buffer[BUFFER_F], CL_FALSE,
                                  0, ncells * Q * sizeof(Float), f, 0, NULL, NULL);
    // wait for mem write finished
    clFinish(queue[0]);
    // set global size
    const size_t global_work_size_collision[3] = {ncells * Q, 1, 1};
    printf("collision_size:%d\n", ncells);
    const size_t global_work_size_edge[3] = {(nx + nx + ny + ny - 4) * Q, 1, 1};
    printf("edge_size:%d\n", (nx + nx + ny + ny - 4));
    const size_t local_work_size[3] = {Q, 1, 1};

    // define kernel_event
    cl_event kernel_event[KERNEL_COUNT];
    double t_file_elapsed = 0.0;
    int remainder = 0;
    for (double t = 0.0; t < t_end; t += dt, t_file_elapsed += dt)
    {
        //printf("%f \n", t);
        // ping-pong buffer
        if (remainder == 0)
        {
            remainder = 1;
            argi = 0;
            status = clSetKernelArg(kernel[KERNEL_LBM], argi, sizeof(cl_mem), &buffer[BUFFER_F]);
            status = clSetKernelArg(kernel[KERNEL_COLLISION], argi, sizeof(cl_mem), &buffer[BUFFER_F_NEXT]);
            status = clSetKernelArg(kernel[KERNEL_EDGE], argi, sizeof(cl_mem), &buffer[BUFFER_F_NEXT]);
        }
        else
        {
            remainder = 0;
            argi = 0;
            status = clSetKernelArg(kernel[KERNEL_LBM], argi, sizeof(cl_mem), &buffer[BUFFER_F_NEXT]);
            status = clSetKernelArg(kernel[KERNEL_COLLISION], argi, sizeof(cl_mem), &buffer[BUFFER_F]);
            status = clSetKernelArg(kernel[KERNEL_EDGE], argi, sizeof(cl_mem), &buffer[BUFFER_F]);
        }
        // set read u flag when every t_file
        // if (t_file_elapsed >= t_file)
        // {
        // t_file_elapsed = 0;
        // flag = 1;
        // excute writeU kernel
        // status = clEnqueueTask(queue[KERNEL_WRITEU], kernel[KERNEL_WRITEU], 0, NULL, &kernel_event[KERNEL_WRITEU]);
        // checkError(status, "Failed to launch KERNEL_WRITEU");
        // }
        argi = 4;
        status = clSetKernelArg(kernel[KERNEL_COLLISION], argi, sizeof(flag), &flag);
        //printf("%f \n", t);

        // excute LBM kernel
        status = clEnqueueTask(queue[KERNEL_LBM], kernel[KERNEL_LBM], 0, NULL, &kernel_event[KERNEL_LBM]);
        checkError(status, "Failed to launch KERNEL_LBM");
        // excute COLLISION kernel
        status = clEnqueueNDRangeKernel(queue[KERNEL_COLLISION], kernel[KERNEL_COLLISION], 3, NULL,
                                        global_work_size_collision, local_work_size, 0, NULL, &kernel_event[KERNEL_COLLISION]);
        checkError(status, "Failed to launch KERNEL_COLLISION");
        // excute EDGE kernel
        status = clEnqueueNDRangeKernel(queue[KERNEL_EDGE], kernel[KERNEL_EDGE], 3, NULL,
                                        global_work_size_edge, local_work_size, 0, NULL, &kernel_event[KERNEL_EDGE]);
        checkError(status, "Failed to launch KERNEL_EDGE");
        //printf("%f \n", t);

        // wait for kernel finished and release them
        for (int j = 0; j < KERNEL_COUNT - 1; j++)
        {
            // cout << "release event:" << j << endl;
            clWaitForEvents(1, &kernel_event[j]);
            clReleaseEvent(kernel_event[j]);
        }
        // read u from device mem
        // if (flag == 1)
        // {
        //     flag = 0;
        //     clEnqueueReadBuffer(queue[KERNEL_LBM], buffer[BUFFER_U], CL_TRUE,
        //                         0, ncells * sizeof(Float2), u_fpga, 0, NULL, NULL);
        //     clFinish(queue[KERNEL_LBM]);
        //     output((int)t, u_fpga);
        //     // testgnuplot(u_fpga);
        // }
        printf("\rt = %.1fs./%.1fs., %.1f%% done", t, t_end, t / t_end * 100.0);
        // compute kernel excute time
        // cl_ulong time_ns = getStartEndTime(kernel_event[KERNEL_COLLISION]);
        // printf("Kernel time: %0.3f ms\n", double(time_ns) * 1e-6);
        // printf("loop %d finished.\n",i);
    }
    printf("\niterate finished.\n\n");
}
// init data
void init_problem()
{
    printf("1/tau=%f\n", tau_f);
    // Set cell flow vector values
    set_e_values(e);

    // Particle distributions
    // 数组分配内存
    u.reset(ncells);
    u_fpga.reset(ncells);
    rho.reset(ncells);

    f.reset(ncells * Q);
    f_new.reset(ncells * Q);
    f_new_ref.reset(ncells * Q);

    // Set densities, velocities and flow vectors
    init_rho_v(rho, u);
    init_f(f, rho, u, e);
}

// Initializes the OpenCL objects.
bool init_opencl()
{
    cl_int status;

    if (!setCwdToExeDir())
    {
        return false;
    }

    // Get the OpenCL platform.
    platform = findPlatform("Intel(R) FPGA");
    if (platform == NULL)
    {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return false;
    }

    unsigned num_devices;
    scoped_array<cl_device_id> devices;

    // Query the available OpenCL device.
    devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    printf("  -Platform: %s\n", getPlatformName(platform).c_str());
    printf("  -find %d devices\n", num_devices);
    for (unsigned i = 0; i < num_devices; ++i)
    {
        printf("    -%s\n", getDeviceName(devices[i]).c_str());
    }
    device = devices[0];
    printf("  -use the first one(%s)\n", getDeviceName(device).c_str());

    // Create the context.
    context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    // Create the command queues. There is one command queue for each kernel
    // because the kernels run simultaneously.
    for (int i = 0; i < KERNEL_COUNT; i++)
    {
        queue[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
        checkError(status, "Failed to create command queue");
    }

    // Create the program.
    std::string binary_file = getBoardBinaryFile("lbm", device);
    printf("  Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Kernel.
    kernel[KERNEL_LBM] = clCreateKernel(program, "lbm", &status);
    checkError(status, "Failed to create kernel lbm");
    kernel[KERNEL_COLLISION] = clCreateKernel(program, "collision", &status);
    checkError(status, "Failed to create kernel collision");
    kernel[KERNEL_EDGE] = clCreateKernel(program, "edge", &status);
    checkError(status, "Failed to create kernel edge");
    kernel[KERNEL_WRITEU] = clCreateKernel(program, "writeU", &status);
    checkError(status, "Failed to create kernel writeU");

    //mem object
    buffer[BUFFER_F] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      Q * ncells * sizeof(Float), NULL, &status);
    checkError(status, "Failed to create buffer for f");
    buffer[BUFFER_F_NEXT] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           Q * ncells * sizeof(Float), NULL, &status);
    checkError(status, "Failed to create buffer for f_next");
    buffer[BUFFER_U] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      ncells * sizeof(Float2), NULL, &status);
    checkError(status, "Failed to create buffer for u");
    return true;
}

// need not to modify
void cleanup()
{
    cout << "release kernel and queue" << endl;
    for (unsigned i = 0; i < KERNEL_COUNT; i++)
    {
        if (kernel && kernel[i])
        {
            clReleaseKernel(kernel[i]);
        }
        if (queue && queue[i])
        {
            clReleaseCommandQueue(queue[i]);
        }
    }

    cout << "release buffer" << endl;
    for (unsigned i = 0; i < BUFFER_COUNT; i++)
    {
        if (buffer && buffer[i])
        {
            clReleaseMemObject(buffer[i]);
        }
    }

    cout << "release program" << endl;
    if (program)
        clReleaseProgram(program);

    cout << "release context" << endl;
    if (context)
        clReleaseContext(context);
}

// Entry point.
int main(int argc, char **argv)
{
    // read some value from command line
    Options options(argc, argv);
    if (options.has("x"))
    {
        nx = options.get<unsigned>("x");
    }
    if (options.has("y"))
    {
        ny = options.get<unsigned>("y");
    }
    if (options.has("l"))
    {
        t_end = options.get<double>("l");
    }
    if (options.has("t"))
    {
        t_file = options.get<double>("t");
    }

    // recompute some value
    ncells = nx * ny;
    Lx = dx * nx;
    Ly = dx * ny;
    nu = U * Lx / Re;
    tau_f = 1 / tau();
    // print some info
    printf("### Lattice-Boltzman D%dQ%d test ###\n", D, Q);
    printf("enter host program.\n");
    printf("nx=%d,ny=%d\n", nx, ny);
    printf("ncells=%d\n", ncells);

    // Initialize OpenCL.
    printf("==init OpenCL environment.\n");
    if (!init_opencl())
    {
        return -1;
    }
    printf("OpenCL environment init finished.\n");

    // Initialize the problem data.
    printf("==init program data.\n");
    init_problem();
    printf("program data init finished.\n\n");

    // Run the kernel and compute the duration.
    printf("==start run.\n");
    const Float start_time = getCurrentTimestamp();
    run();
    const Float end_time = getCurrentTimestamp();
    const Float total_time = end_time - start_time;
    printf("FPGA total Time: %0.3f ms\n", total_time * 1e3);

    // Free the resources allocated
    cleanup();
    return 0;
}
