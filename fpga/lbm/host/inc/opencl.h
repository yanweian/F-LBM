#ifndef __OPENCL_H__
#define __OPENCL_H__
#include "lbm.h"
/**
 * OPENCL HELPER
*/
// Kernels.
enum Kernel
{
    KERNEL_DATA_READER = 0, //infinite
    KERNEL_BU,              //auto
    KERNEL_CU,
    KERNEL_VELOCITY_WRITER, //infinite
    KERNEL_COUNT
};
// Buffers.
enum Buffer
{
    BUFFER_fSA = 0,
    BUFFER_fSB,
    BUFFER_U,
    BUFFER_COUNT
};
void cleanup() {}
class OpenCLFPGA
{
public:
    void Run(Lattice &lattice);
    cl_command_queue queues[KERNEL_COUNT];
    cl_kernel kernels[KERNEL_COUNT]; // kernel array
    cl_mem buffers[BUFFER_COUNT];    // mem object to exchange data between host and device
    cl_event events[KERNEL_COUNT];

    cl_program program;
    cl_int status;
    cl_platform_id platform;
    cl_context context;
    cl_device_id device;

private:
    bool Init();
    void CleanUp();
    void CreateKernelQueues();
    bool InitStatic();
    void CreateBuffers(Lattice &lattice);
    void SetKernelArgs(Lattice &lattice);
    void TransferData2Device(Lattice &lattice);
    void TransferData2Host(Lattice &lattice, Vector2DReal *u);
    void RunKernels(int interval, int end, Lattice &lattice);
    void Output2Files(int i, Lattice &lattice, Vector2DReal *u);
};
void OpenCLFPGA::Run(Lattice &lattice)
{
    if (Init())
    {
        // Create buffers
        cout << "Create buffers." << endl;
        CreateBuffers(lattice);
        cout << "Set args of kernels." << endl;
        SetKernelArgs(lattice);
        cout << "Transfer Datas to device" << endl;
        TransferData2Device(lattice);
        cout << "Launch Kernels with a specific sequece" << endl;
        RunKernels(lattice.interval, lattice.iterate_end, lattice);
        cout << "Kernel finished and Clear Resources" << endl;
        CleanUp();
        lattice.grid.release();
    }
}
void OpenCLFPGA::RunKernels(int interval, int end, Lattice &lattice)
{
    // set global size
    const size_t global_work_size_dr[3] = {lattice.lbmparameters.size.x, lattice.lbmparameters.size.y, 1};
    const size_t local_work_size_dr[3] = {N_VECTOR, N_VECTOR, 1};
    const size_t global_work_size_cu[3] = {lattice.lbmparameters.size.x * lattice.lbmparameters.size.y * Q_VECTOR, 1, 1};
    const size_t local_work_size_cu[3] = {Q_VECTOR, 1, 1};
    const size_t global_work_size_bu[3] = {2 * (lattice.lbmparameters.size.x + lattice.lbmparameters.size.y - 2) * Q_VECTOR, 1, 1};
    const size_t local_work_size_bu[3] = {Q_VECTOR, 1, 1};

    cl_int status;
    scoped_aligned_ptr<Vector2DReal> u;
    u.reset(lattice.lbmparameters.ncells);
    char iswrite = 0x0;
    cl_mem buffer_dr = buffers[BUFFER_fSA];
    cl_mem buffer_cbu = buffers[BUFFER_fSB];
    // ====== loop signal generator =======
    cout << "===========Loop signal generator...\n";
    for (int i = 0; i < end; i++)
    {
        iswrite = 0x1;
        for (int j = 0; j < interval; j++)
        {
            if (j == interval - 1)
                iswrite = 0x0;
            // set args
            unsigned argi = 0;

            status = clSetKernelArg(kernels[KERNEL_DATA_READER], argi, sizeof(cl_mem), &buffer_dr);
            status = clSetKernelArg(kernels[KERNEL_CU], argi, sizeof(cl_mem), &buffer_cbu);
            status = clSetKernelArg(kernels[KERNEL_BU], argi, sizeof(cl_mem), &buffer_cbu);
            status = clSetKernelArg(kernels[KERNEL_VELOCITY_WRITER], argi, sizeof(char), &iswrite);

            status = clEnqueueNDRangeKernel(queues[KERNEL_DATA_READER], kernels[KERNEL_DATA_READER], 3, NULL,
                                            global_work_size_dr, local_work_size_dr, 0, NULL, &events[KERNEL_DATA_READER]);
            checkError(status, "Failed to launch KERNEL_DATA_READER");

            // status = clEnqueueNDRangeKernel(queues[KERNEL_CU], kernels[KERNEL_CU], 3, NULL,
            //                                 global_work_size_cu, local_work_size_cu, 0, NULL, &events[KERNEL_CU]);
            status = clEnqueueTask(queues[KERNEL_CU], kernels[KERNEL_CU], 0, NULL, &events[KERNEL_CU]);
            checkError(status, "Failed to launch KERNEL_CU");

            // status = clEnqueueNDRangeKernel(queues[KERNEL_BU], kernels[KERNEL_BU], 3, NULL,
            //                                 global_work_size_bu, local_work_size_bu, 0, NULL, &events[KERNEL_BU]);
            status = clEnqueueTask(queues[KERNEL_BU], kernels[KERNEL_BU], 0, NULL, &events[KERNEL_BU]);
            checkError(status, "Failed to launch KERNEL_BU");

            status = clEnqueueTask(queues[KERNEL_VELOCITY_WRITER], kernels[KERNEL_VELOCITY_WRITER], 0, NULL, &events[KERNEL_VELOCITY_WRITER]);
            checkError(status, "Failed to launch KERNEL_VELOCITY_WRITER");

            // release
            for (int k = 0; k < KERNEL_COUNT; k++)
            {
                clWaitForEvents(1, &events[k]);
                clReleaseEvent(events[k]);
            }

            cl_mem mid = buffer_dr;
            buffer_dr = buffer_cbu;
            buffer_cbu = mid;
        }
        cout << "===========================Loop " << i << " ==================================" << endl;
        // 读取v数据
        cout << "transfer data to host\n";
        TransferData2Host(lattice, u);
        cout << "transfer data to file\n";
        Output2Files(i, lattice, u);
    }
}
void OpenCLFPGA::Output2Files(int i, Lattice &lattice, Vector2DReal *u)
{
    ostringstream name;
    int nx = lattice.lbmparameters.size.x;
    int ny = lattice.lbmparameters.size.y;
    int Lx = lattice.lbmparameters.Lxy.x;
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
            LongIndex index = getIndex(MakeVec(x, y), lattice.lbmparameters.size);
            out << double(x) / Lx << " " << double(y) / Lx << " " << u[index].x << " " << u[index].y << endl;
        }
    }
}
void OpenCLFPGA::TransferData2Host(Lattice &lattice, Vector2DReal *u)
{
    // ====== write buffer from host to device ====
    cl_int status = clEnqueueReadBuffer(queues[KERNEL_VELOCITY_WRITER], buffers[BUFFER_U], CL_FALSE,
                                        0, lattice.lbmparameters.ncells * sizeof(Vector2DReal), u, 0, NULL, &events[KERNEL_VELOCITY_WRITER]);
    // wait for mem write finished
    clWaitForEvents(1, &events[KERNEL_VELOCITY_WRITER]);
    clReleaseEvent(events[KERNEL_VELOCITY_WRITER]);
}
void OpenCLFPGA::TransferData2Device(Lattice &lattice)
{
    // ====== write buffer from host to device ====
    cl_int status = clEnqueueWriteBuffer(queues[KERNEL_DATA_READER], buffers[BUFFER_fSA], CL_FALSE,
                                         0, Q_VECTOR * lattice.lbmparameters.ncells * sizeof(real), lattice.grid, 0, NULL, &events[KERNEL_DATA_READER]);
    // wait for mem write finished
    clWaitForEvents(1, &events[KERNEL_DATA_READER]);
    clReleaseEvent(events[KERNEL_DATA_READER]);
}
bool OpenCLFPGA::Init()
{
    // Init some static steps, such as platform,context,device,program
    if (!InitStatic())
        return false;
    // Create the command queues and kernels.
    CreateKernelQueues();
    return true;
}
void OpenCLFPGA::CreateKernelQueues()
{
    printf("Create Kernels and Command_queues...\n");
    // Create the command queues.
    for (int i = 0; i < KERNEL_COUNT; i++)
    {
        queues[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
        checkError(status, "Failed to create command queue");
    }
    // Create Kernels.

    kernels[KERNEL_DATA_READER] = clCreateKernel(program, "data_reader", &status);
    checkError(status, "Failed to create kernel KERNEL_DATA_READER");
    kernels[KERNEL_VELOCITY_WRITER] = clCreateKernel(program, "velocity_writer", &status);
    checkError(status, "Failed to create kernel KERNEL_VELOCITY_WRITER");
    kernels[KERNEL_BU] = clCreateKernel(program, "bu", &status);
    checkError(status, "Failed to create kernel KERNEL_BU");
    kernels[KERNEL_CU] = clCreateKernel(program, "cu", &status);
    checkError(status, "Failed to create kernel KERNEL_CU");
}
void OpenCLFPGA::CreateBuffers(Lattice &lattice)
{
    cl_int status;
    //======== mem object =============
    cout << "create cl_mem objects\n";
    buffers[BUFFER_fSA] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         Q_VECTOR * lattice.lbmparameters.ncells * sizeof(real), NULL, &status);
    checkError(status, "Failed to create buffer for cells");
    buffers[BUFFER_fSB] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         Q_VECTOR * lattice.lbmparameters.ncells * sizeof(real), NULL, &status);
    checkError(status, "Failed to create buffer for cells");
    buffers[BUFFER_U] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       lattice.lbmparameters.ncells * sizeof(Vector2DReal), NULL, &status);
    checkError(status, "Failed to create buffer for u");
}
bool OpenCLFPGA::InitStatic()
{
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
    printf("Platform: %s\n", getPlatformName(platform).c_str());
    printf("Find %d devices\n", num_devices);
    for (unsigned i = 0; i < num_devices; ++i)
    {
        printf("    -%s\n", getDeviceName(devices[i]).c_str());
    }
    device = devices[0];
    if (num_devices > 1)
        printf("Use one of them(%s)\n", getDeviceName(device).c_str());

    // Create the context.
    context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    // Create the program.
    std::string binary_file = getBoardBinaryFile("lbm", device);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");
    return true;
}
void OpenCLFPGA::SetKernelArgs(Lattice &lattice)
{
    int interval = lattice.interval;
    LbmParameters parameters = lattice.lbmparameters;

    cl_int status;
    unsigned argi = 0;
    status = clSetKernelArg(kernels[KERNEL_DATA_READER], argi++, sizeof(cl_mem), &buffers[BUFFER_fSA]);
    status = clSetKernelArg(kernels[KERNEL_DATA_READER], argi++, sizeof(real), &parameters.u0.x);
    status = clSetKernelArg(kernels[KERNEL_DATA_READER], argi++, sizeof(real), &parameters.u0.y);

    argi = 0;
    char iswrite = 0x0;
    status = clSetKernelArg(kernels[KERNEL_VELOCITY_WRITER], argi++, sizeof(char), &iswrite);
    status = clSetKernelArg(kernels[KERNEL_VELOCITY_WRITER], argi++, sizeof(cl_mem), &buffers[BUFFER_U]);
    status = clSetKernelArg(kernels[KERNEL_VELOCITY_WRITER], argi++, sizeof(Index), &parameters.size.x);
    status = clSetKernelArg(kernels[KERNEL_VELOCITY_WRITER], argi++, sizeof(Index), &parameters.size.y);

    argi = 0;
    status = clSetKernelArg(kernels[KERNEL_CU], argi++, sizeof(cl_mem), &buffers[BUFFER_fSB]);
    status = clSetKernelArg(kernels[KERNEL_CU], argi++, sizeof(real), &parameters.tau_f);
    status = clSetKernelArg(kernels[KERNEL_CU], argi++, sizeof(Index), &parameters.size.x);
    status = clSetKernelArg(kernels[KERNEL_CU], argi++, sizeof(Index), &parameters.size.y);

    argi = 0;
    status = clSetKernelArg(kernels[KERNEL_BU], argi++, sizeof(cl_mem), &buffers[BUFFER_fSB]);
    status = clSetKernelArg(kernels[KERNEL_BU], argi++, sizeof(Index), &parameters.size.x);
    status = clSetKernelArg(kernels[KERNEL_BU], argi++, sizeof(Index), &parameters.size.y);
}
void OpenCLFPGA::CleanUp()
{
    for (int i = 0; i < KERNEL_COUNT; i++)
    {
        if (kernels && kernels[i])
        {
            clReleaseKernel(kernels[i]);
        }
        if (queues && queues[i])
        {
            clReleaseCommandQueue(queues[i]);
        }
    }
    for (int i = 0; i < BUFFER_COUNT; i++)
    {
        if (buffers && buffers[i])
        {
            cout << "clean buffer:" << i << endl;
            clReleaseMemObject(buffers[i]);
        }
    }
    if (program)
        clReleaseProgram(program);
    if (context)
        clReleaseContext(context);
    cout << "clear finished\n";
}

#endif