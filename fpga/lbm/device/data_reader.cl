/**
read cell data from global buffer */
__attribute__((reqd_work_group_size(N_VECTOR, N_VECTOR, 1)))
kernel void
data_reader(__global real *restrict fs, real u0x, real u0y)
{
    int global_id_0 = get_global_id(0);
    int global_id_1 = get_global_id(1);

    int global_size_0 = get_global_size(0);
    int global_size_1 = get_global_size(1);

    int item_id_0 = get_local_id(0);
    int item_id_1 = get_local_id(1);

    // 定义local mem
    __local Cell cells[N_VECTOR][N_VECTOR];

    // 判断是否为边界点
    bool isBoundary = false;
    bool isUp = false;
    bool isDown = false;
    bool isRight = false;
    bool isLeft = false;
    if (global_id_1 == (global_size_1 - 1)) //上边界
    {
        isUp = true;
        isBoundary = true;
    }
    else if (global_id_1 == 0) //下边界
    {
        isDown = true;
        isBoundary = true;
    }
    if (global_id_0 == (global_size_0 - 1)) //右边界
    {
        isRight = true;
        isBoundary = true;
    }
    else if (global_id_0 == 0) //左边界
    {
        isLeft = true;
        isBoundary = true;
    }

    // 每个work-item 读取 cell
    Cell cell;
    cell.position = MakeVecIndex(global_id_0, global_id_1);
    // 计算 start global_index:(y*size_x+x)*Q
    size_t index = (global_id_1 * global_size_0 + global_id_0) * Q_VECTOR;
    // 读取f，顺便计算rho和u
    Vector2DReal u = {0.f, 0.f};
    real rho = 0.f;
#pragma unroll
    for (int k = 0; k < Q_VECTOR; k++)
    {
        cell.f[k] = fs[index + k];
        rho += cell.f[k];
        u.x += e[k].x * cell.f[k];
        u.y += e[k].y * cell.f[k];
    }
    cell.rho = rho;
    if (isUp) //上边界
    {
        cell.u = MakeVecReal(u0x, u0y);
    }
    else if (isBoundary) //其他边界，这时不会计算上边界
    {
        cell.u = MakeVecReal(0.f, 0.f);
    }
    else //其他内部格点
    {
        cell.u.x = u.x / rho;
        cell.u.y = u.y / rho;
        // 可以将内部格点送到CU中进行碰撞
        // write_channel_intel(DATAR2CU_CELL_CHANNEL, cell);
    }
    // 内部格点全部计算完毕，边界的rho还未计算
    cells[item_id_1][item_id_0] = cell;

    // 等待所有work-item完成上述步骤
    barrier(CLK_LOCAL_MEM_FENCE);

    // 计算offset
    if (isBoundary)
    {
        int offset_x = 0;
        int offset_y = 0;
        if (isUp)
            offset_y = -1;
        else if (isDown)
            offset_y = 1;
        if (isRight)
            offset_x = -1;
        else if (isLeft)
            offset_x = 1;
        Cell cell_interior = cells[item_id_1 + offset_y][item_id_0 + offset_x];
        cell.rho = cell_interior.rho;
        // 至此，边界格点的rho获得
        // 组装Cell_Edge
        CellEdge cell_edge;
#pragma unroll
        for (int k = 0; k < Q_VECTOR; k++)
        {
            cell_edge.f[k] = cell_interior.f[k];
        }
        cell_edge.u_edge = cell.u;
        cell_edge.u = cell_interior.u;
        cell_edge.rho = cell_interior.rho;
        cell_edge.position = cell.position;
        // 送至BU计算
        write_channel_intel(DATAR2BU_CELLEDGE_CHANNEL, cell_edge);
    }
    // 送至CU计算
    write_channel_intel(DATAR2CU_CELL_CHANNEL, cell);
    if (DEBUG)
        printf("data_reader: global_index=%lu cell = (%d, %d) u = (%f, %f) rho = %f boundary(%d,%d,%d,%d)\n", index / Q_VECTOR, cell.position.x, cell.position.y, cell.u.x, cell.u.y, cell.rho, isUp, isDown, isLeft, isRight);
}