/**
this code is for colliding parallel */
// this function is the prototype code for each cu kernels in N_VECTOR cu arrays
TASK kernel void cu(real tau_f, global HostCell *restrict hostcells, Index size_x, Index size_y)
{
    while (true)
    {
        CellVector cellvector = read_channel_intel(BU2CU_CELLVECTOR_CHANNEL);
        write_channel_intel(CU2VW_CELLVECTOR_CHANNEL, cellvector);
        // 确定偏移量
        LongIndex offset = cellvector.offset;
        int i = 0;
        int j = 0;
        if (DEBUG)
            printf("cu: get group offset=%d\n", offset);
#pragma ivdep
        while (j < N_VECTOR)
        {
            Cell cell = cellvector.cells[j][i];
#pragma unroll
            for (int m = 0; m < Q_VECTOR; m++)
            {
                // 计算新的坐标
                int new_x = cell.position.x + e[m].x;
                int new_y = cell.position.y + e[m].y;
                // 判断坐标位置
                if (new_x > 0 && new_x < (size_x - 1) && new_y > 0 && new_y < (size_y - 1))
                {
                    // 计算并传输f至host
                    hostcells[new_y * size_x + new_x + offset].f[m] = bgk(cell.f[m], cell.rho, cell.u, m, tau_f);
                    if (DEBUG)
                    {
                        printf("cu: cell(%d,%d)[%d]->cell(%d,%d)=%f\n", cell.position.x, cell.position.y, m, new_x, new_y, hostcells[cell.position.y * size_x + cell.position.x + offset].f[m]);
                    }
                }
            }
            if (++i == N_VECTOR)
            {
                i = 0;
                j++;
            }
        }
    }
}