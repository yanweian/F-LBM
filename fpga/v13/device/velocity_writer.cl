TASK kernel void velocity_writer(
    char iswrite,
    global Vector2DReal *restrict u,
    LongIndex groupsize,
    Index size_x)
{
    LongIndex groupindex = 0;
#pragma ivdep
    while (groupindex < groupsize)
    {
        if (DEBUG)
            printf("velocity_writer:get group=%d\n", groupindex);
        CellVector cell_vector = read_channel_intel(CU2VW_CELLVECTOR_CHANNEL);
        if (iswrite == 0x0)
        {
            // write velocity to ddr
            int i = 0;
            int j = 0;
            while (j < N_VECTOR)
            {
                Cell cell = cell_vector.cells[j][i];
                LongIndex index = cell.position.y * size_x + cell.position.x;
                if (DEBUG)
                    printf("velocity_writer:group %d index %d (x,y)=(%d,%d) (j,i)=(%d,%d) u=(%f,%f)\n", groupindex, index, cell.position.x, cell.position.y, j, i, cell.u.x, cell.u.y);
                // write u to global_mem
                u[index] = cell.u;
                if (++i == N_VECTOR)
                {
                    i = 0;
                    j++;
                }
            }
        }
        groupindex++;
    }
}