using System;

namespace SSE_Scatter
{
    class Program
    {
        static void PrintSSE_Scatter(int size)
        {
            for (int gap = 2; gap < size; gap++)
            {
                for (int offset = 0; offset < gap; offset++)
                {
                    int[] permutation = new int[gap * gap * 3];
                    uint[] mask = new uint[gap * gap * 3];
                    for (int i = 0; i < 4; i++)
                    {
                        permutation[gap * i + offset] = i;
                        mask[gap * i + offset] = 1;
                    }

                    for (int line = 0; line < gap; line++)
                    {
                        int[] bits = new int[4];
                        uint m = 0;
                        for (int i = 0; i < 4; i++)
                        {
                            bits[i] = permutation[line * 4 + i];
                        }

                        m = (mask[line * 4] * 0xff) +
                            ((mask[line * 4 + 1] * 0xff) << 8) +
                            ((mask[line * 4 + 2] * 0xff) << 16) +
                            ((mask[line * 4 + 3] * 0xff) << 24);

                        Console.WriteLine("SET_SCATTER_SHUFFLE(" + gap + ", " + offset + ", " + line +
                            ", _MM_SHUFFLE(" + bits[3] + "," + bits[2] + "," + bits[1] + "," + bits[0] + "), 0x"
                            + m.ToString("X8") + ");");

                        int[] gbits = new int[4];
                        uint[] gmaks = new uint[4];

                        for (int i = 0; i < 4; i++)
                        {
                            if (mask[i] != 0)
                            {
                                gbits[bits[i]] = i;
                                gmaks[bits[i]] = 0xff;
                            }
                        }

                        m = (gmaks[0] * 0xff) +
                            ((gmaks[1] * 0xff) << 8) +
                            ((gmaks[2] * 0xff) << 16) +
                            ((gmaks[3] * 0xff) << 24);

                        Console.WriteLine("SET_GATHER_SHUFFLE(" + gap + ", " + offset + ", " + line +
                            ", _MM_SHUFFLE(" + gbits[3] + "," + gbits[2] + "," + gbits[1] + "," + gbits[0] + "), 0x"
                            + m.ToString("X8") + ");");
                    }
                }
            }
        }



        static void Main(string[] args)
        {
            PrintSSE_Scatter(6);
        }
    }
}
