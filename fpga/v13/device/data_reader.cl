/**
read cell data from global buffer */
TASK kernel void data_reader(LongIndex offset, LongIndex nextoffset,
                             global HostCell *restrict hostcells,
                             Index groupsize_x, Index groupsize_y,
                             Index size_x, real u0x, real u0y)
{
    Index group_y = 0;
    Index group_x = 0;
    LongIndex group_index = offset;

    while (group_y < groupsize_y)
    {
        // ================start reader data==================
        CellVector cellvector;
        cellvector.offset = nextoffset;
        cellvector.isboundary = 0x0;
        int i = 0;
        int j = 0;
        LongIndex item_index = group_index;
        while (j < N_VECTOR)
        {
            // read data from global_mem
            HostCell tmpHostCell = hostcells[item_index];
            Cell cell;
            cell.mask = tmpHostCell.mask;
            cell.position = tmpHostCell.position;
            cell.offset = MakeVecIndex(0, 0);
            if (DEBUG)
                printf("data_reader: cells(%d,%d)=%d\n", cell.position.x, cell.position.y, item_index);
            // compute rho and u
            if (cell.mask == 0x0) //normal point
            {
                real rho = 0.f;
                Vector2DReal u = {0.f, 0.f};
#pragma unroll
                for (int m = 0; m < Q_VECTOR; m++)
                {
                    cell.f[m] = tmpHostCell.f[m];
                    rho += cell.f[m];
                    u.x += e[m].x * cell.f[m];
                    u.y += e[m].y * cell.f[m];
                }
                u.x /= rho;
                u.y /= rho;

                cell.rho = rho;
                cell.u = u;
                if (DEBUG)
                    printf("data_reader: rho=%f,\n", rho);
            }
            else
            { // 标记为边界组，边界的cell的速度为0
                cellvector.isboundary = 0x1;
                cell.u = MakeVecReal(0.f, 0.f);
                //计算偏移量
                Vector2DIndex positionOffset = {0, 0};
                if (ISUP(tmpHostCell.mask)) // drive point
                {
                    positionOffset.y = -1;
                    //如果是上边界，速度为驱动速度
                    cell.u = MakeVecReal(u0x, u0y);
                }
                else if (ISDOWN(tmpHostCell.mask))
                {
                    positionOffset.y = 1;
                }
                if (ISRIGHT(tmpHostCell.mask))
                {
                    positionOffset.x = -1;
                }
                else if (ISLEFT(tmpHostCell.mask))
                {
                    positionOffset.x = 1;
                }
                //设置边界点的偏移量
                cell.offset = positionOffset;
            }
            if (DEBUG)
                printf("data_reader: u(%f,%f)\n", cell.u.x, cell.u.y);
            cellvector.cells[j][i] = cell;
            // compute index(j,i)
            item_index++;
            i++;
            if (i == N_VECTOR)
            {
                i = 0;
                j++;
                item_index = group_index + j * size_x;
            }
        }
        // ================end reader data==================
        // compute rho
        if (cellvector.isboundary == 0x1)
        {
            int i2 = 0;
            int j2 = 0;
            while (j2 < N_VECTOR)
            {
                Vector2DIndex positionOffset = cellvector.cells[j2][i2].offset;
                cellvector.cells[j2][i2].rho = cellvector.cells[j2 + positionOffset.y][i2 + positionOffset.x].rho;
                if (DEBUG)
                    printf("data_reader:cell(%d,%d) u(%f,%f) rho=%f\n", j2, i2, cellvector.cells[j2][i2].u.x, cellvector.cells[j2][i2].u.y, cellvector.cells[j2][i2].rho);
                i2++;
                if (i2 == N_VECTOR)
                {
                    i2 = 0;
                    j2++;
                }
            }
        }
        // transfer data to boundary_detector
        write_channel_intel(DATAR2BU_CELLVECTOR_CHANNEL, cellvector);

        if (DEBUG)
            printf("data_reader:transfer group(%d,%d) to bu\n", group_x, group_y);
        // compute index
        group_index += N_VECTOR;
        group_x++;
        if (group_x == groupsize_x)
        {
            group_index += size_x;
            group_x = 0;
            group_y++;
        }
    }
}