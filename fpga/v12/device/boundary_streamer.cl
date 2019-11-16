/**
writer cell data from global buffer */
TASK kernel void boundary_streamer(
    global Cell *restrict cells,
    LongIndex ncells, Index size_x, Index size_y)
{
    bool buf_flag = false;
    Vector2DIndex size = MakeVecIndex(size_x, size_y);
    // LongIndex count_flag = (size.x + size.y - 2); // avoide divide
    LongIndex count_flag = 0; // avoide divide
    LongIndex fullsize = (size.x + size.y - 2) * 2;
    LongIndex count = fullsize;
    LongIndex offset = ncells;

#pragma ivdep
    while (true)
    {
        Cell boundary = read_channel_intel(BU2BSTREAMER_CELL_CHANNEL);
        if (DEBUG)
            printf("boundary_streamer: getted data from bu count=%d\n", count);
        LongIndex cellIndex = getIndex(boundary.position, size) + offset;
        cells[cellIndex] = boundary;
        count--;
        if (count == count_flag)
            write_channel_intel(BSTREAMER2DATAR_CHAR_CHANNEL, '1');
        if (count == 0)
        {
            count = fullsize;
            // ping-pong buffer, using offset
            offset = 0;
            if (buf_flag)
                offset = ncells;
            buf_flag = !buf_flag;
        }
    }
}