/**
this code is for colliding parallel */
// this function is the prototype code for each cu kernels in N_VECTOR cu arrays
// __attribute__((reqd_work_group_size(Q_VECTOR, 1, 1)))
// kernel void
// cu(global real *restrict fs, real tau_f, Index size_x, Index size_y)
// {
//     int item_id_0 = get_local_id(0);

//     __local Cell local_cell[1];
//     // Cell cell;
//     // 每一组第一个工作项读channel
//     if (item_id_0 == 0)
//     {
//         local_cell[0] = read_channel_intel(DATAR2CU_CELL_CHANNEL);
//         write_channel_intel(CU2VW_CELL_CHANNEL, local_cell[0]);
//         // if (DEBUG)
//         //     printf("cu: get cell = (%d, %d) from dr and transfer it to vw\n", cell.position.x, cell.position.y);
//         // local_cell[0] = cell;
//     }
//     // 等待第一个工作项读取完毕
//     barrier(CLK_LOCAL_MEM_FENCE);
//     Cell cell = local_cell[0];
//     // 开始计算
//     // propagation
//     int new_x = cell.position.x + e[item_id_0].x;
//     int new_y = cell.position.y + e[item_id_0].y;
//     if (new_x > 0 && new_x < (size_x - 1) && new_y > 0 && new_y < (size_y - 1))
//     {
//         size_t new_id = (new_y * size_x + new_x) * Q_VECTOR + item_id_0;
//         // compute new value;
//         fs[new_id] = bgk(cell.f[item_id_0], cell.rho, cell.u, item_id_0, tau_f);
//     }
// }
kernel void
cu(global real *restrict fs, real tau_f, Index size_x, Index size_y)
{
    size_t end = size_x * size_y;
#pragma ivdep
    while ((end--) > 0)
    {
        if (DEBUG)
            printf("cu: end=%lu\n", end);
        Cell cell = read_channel_intel(DATAR2CU_CELL_CHANNEL);
        write_channel_intel(CU2VW_CELL_CHANNEL, cell);
        if (DEBUG)
            printf("cu: get cell = (%d, %d) from dr and transfer it to vw\n", cell.position.x, cell.position.y);
#pragma unroll
        for (int item_id_0 = 0; item_id_0 < Q_VECTOR; item_id_0++)
        {
            // 开始计算
            // propagation
            int new_x = cell.position.x + e[item_id_0].x;
            int new_y = cell.position.y + e[item_id_0].y;
            if (new_x > 0 && new_x < (size_x - 1) && new_y > 0 && new_y < (size_y - 1))
            {
                size_t new_id = (new_y * size_x + new_x) * Q_VECTOR + item_id_0;
                // compute new value;
                fs[new_id] = bgk(cell.f[item_id_0], cell.rho, cell.u, item_id_0, tau_f);
            }
        }
    }
}