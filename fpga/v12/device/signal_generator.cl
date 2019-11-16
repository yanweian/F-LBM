/**
    transfer signal from cpu to data_reader
 */
TASK kernel void signal_generator()
{
    write_channel_intel(SINGALG2DATAR_CHAR_CHANNEL, 'a');
    char s = read_channel_intel(VW2SINGALG_CHAR_CHANNEL);
    if (DEBUG)
        printf("signal_generator: s= %c\n", s);
}