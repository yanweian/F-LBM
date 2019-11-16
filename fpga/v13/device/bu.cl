TASK kernel void bu(global HostCell *restrict hostcells, Index size_x)
{
    while (true)
    {
        CellVector cellvector = read_channel_intel(DATAR2BU_CELLVECTOR_CHANNEL);
        write_channel_intel(BU2CU_CELLVECTOR_CHANNEL, cellvector);
        if (cellvector.isboundary == 0x1) //处理边界组
        {
            LongIndex offset = cellvector.offset;
            if (DEBUG)
                printf("bu: get boundary cellvector\n");
            int x = 0;
            int y = 0;
#pragma ivdep
            while (y < N_VECTOR)
            {
                Cell cell = cellvector.cells[y][x];
                if (cell.mask != 0x0) // 处理边界cell
                {
                    // 获取内部cell
                    Cell cell_interior = cellvector.cells[y + cell.offset.y][x + cell.offset.x];
                    // 实例化hostcell
                    HostCell tmpHostCell;
                    // 计算新的f
                    for (int i = 0; i < Q_VECTOR; i++)
                    {
                        tmpHostCell.f[i] = nfeq_boundary(cell_interior.f[i], cell_interior.rho, cell_interior.u, cell.u, i);
                    }
                    // 复制mask和position
                    tmpHostCell.mask = cell.mask;
                    tmpHostCell.position = cell.position;
                    // 传输至global中
                    hostcells[cell.position.y * size_x + cell.position.x + offset] = tmpHostCell;
                }
                //compute index
                x++;
                if (x == N_VECTOR)
                {
                    x = 0;
                    y++;
                }
            }
        }
    }
}