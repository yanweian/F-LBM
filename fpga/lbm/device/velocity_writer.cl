kernel void
velocity_writer(char iswrite, global Vector2DReal *restrict u, Index size_x, Index size_y)
{
    if (DEBUG)
        printf("velocity_writer: iswrite=%d\n", iswrite);
    size_t end = size_x * size_y;
#pragma ivdep
    while ((end--) > 0)
    {
        Cell cell = read_channel_intel(CU2VW_CELL_CHANNEL);
        if (iswrite == 0x0)
        {
            if (DEBUG)
                printf("velocity_writer:(x,y)=(%d,%d) u=(%f,%f)\n", cell.position.x, cell.position.y, cell.u.x, cell.u.y);
            size_t index = cell.position.y * size_x + cell.position.x;
            // write u to global_mem
            u[index] = cell.u;
        }
    }
}