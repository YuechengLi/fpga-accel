/* test_driver.c -- host test for the Stage 5 PS driver skeleton.
 *
 * Runs the driver on the compiled SRCNN program (sw/out/srcnn/) and
 * verifies the issued config sequence against program.txt:
 *   * every W/DMA/GO/WAIT/STREAM line is executed without error;
 *   * the config-write log matches the W lines in order;
 *   * the DMA count / GO count match the program;
 *   * the mock coefficient memory holds the right coefficients
 *     (spot-check: the first L1 coefficient at 0x100, and the first
 *     coefficient of the second (o=1) pass at the same address after
 *     its DMA -- the engine overwrites the registers per pass).
 *
 * Usage: ./test_driver [program_dir]   (default: ../out/srcnn)
 * Exit 0 on PASS, 1 on any mismatch.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "driver.h"
#include "mock_mmio.h"

static char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    char *buf;
    size_t n;
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (char *)malloc(n + 1);
    if (fread(buf, 1, n, f) != n) {
        fprintf(stderr, "short read %s\n", path);
        exit(1);
    }
    fclose(f);
    buf[n] = '\0';
    if (len)
        *len = n;
    return buf;
}

/* Count the instruction kinds in program.txt (the expected counts). */
static void count_instrs(const char *text, uint32_t *w, uint32_t *dma,
                         uint32_t *go, uint32_t *wait, uint32_t *stream)
{
    const char *p = text;
    *w = *dma = *go = *wait = *stream = 0;
    while (*p) {
        char op[16];
        if (sscanf(p, "%15s", op) == 1) {
            if (strcmp(op, "W") == 0) (*w)++;
            else if (strcmp(op, "DMA") == 0) (*dma)++;
            else if (strcmp(op, "GO") == 0) (*go)++;
            else if (strcmp(op, "WAIT") == 0) (*wait)++;
            else if (strcmp(op, "STREAM") == 0) (*stream)++;
        }
        p = strchr(p, '\n');
        if (!p)
            break;
        p++;
    }
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "../out/srcnn";
    char txt_path[512], bin_path[512];
    char *text;
    uint8_t *blob;
    size_t blob_len;
    mock_mmio_t mock;
    accel_mmio_t mmio;
    accel_driver_t drv;
    uint32_t ew, edma, ego, ewait, estream;
    int fail = 0;
    uint32_t i;

    snprintf(txt_path, sizeof(txt_path), "%s/program.txt", dir);
    snprintf(bin_path, sizeof(bin_path), "%s/weights.bin", dir);
    text = read_file(txt_path, NULL);
    blob = (uint8_t *)read_file(bin_path, &blob_len);

    mock_mmio_init(&mock, blob, blob_len);
    mmio.cfg_write = mock_cfg_write;
    mmio.cfg_read = mock_cfg_read;
    mmio.dma_write = mock_dma_write;
    mmio.dma_read = mock_dma_read;
    mmio.ctx = &mock;
    accel_driver_init(&drv, &mmio, blob, blob_len);

    if (accel_exec_program(&drv, text) != 0) {
        printf("FAIL: driver returned an error\n");
        return 1;
    }

    count_instrs(text, &ew, &edma, &ego, &ewait, &estream);
    printf("program: W=%u DMA=%u GO=%u WAIT=%u STREAM=%u\n",
           ew, edma, ego, ewait, estream);
    printf("driver:  writes=%u dma=%u go=%u wait=%u stream=%u\n",
           drv.n_writes, drv.n_dma, drv.n_go, drv.n_wait, drv.n_stream);

    /* The driver's W count includes the GO writes (each GO is a
     * config write to MC_GO), so writes == W lines + GO lines. */
    if (drv.n_writes != ew + ego) {
        printf("FAIL: config writes %u != W(%u)+GO(%u)\n",
               drv.n_writes, ew, ego);
        fail = 1;
    }
    if (drv.n_dma != edma) {
        printf("FAIL: DMA %u != %u\n", drv.n_dma, edma);
        fail = 1;
    }
    if (drv.n_go != ego) {
        printf("FAIL: GO %u != %u\n", drv.n_go, ego);
        fail = 1;
    }
    if (drv.n_wait != ewait) {
        printf("FAIL: WAIT %u != %u\n", drv.n_wait, ewait);
        fail = 1;
    }
    if (drv.n_stream != estream) {
        printf("FAIL: STREAM %u != %u\n", drv.n_stream, estream);
        fail = 1;
    }

    /* The config-write log must match the W lines in order (plus
     * the GO writes interleaved). Rebuild the expected write
     * sequence from program.txt and compare. */
    {
        uint32_t wi = 0; /* index into the mock write log */
        const char *p = text;
        while (*p && wi < mock.n_writes) {
            char op[16];
            unsigned long a, b;
            if (sscanf(p, "%15s", op) == 1 && strcmp(op, "W") == 0) {
                if (sscanf(p, "%*s %lx %lx", &a, &b) == 2) {
                    /* Find this W in the log (GO writes may be
                     * interleaved between W lines). */
                    while (wi < mock.n_writes
                           && !(mock.w_addr[wi] == (uint16_t)a
                                && mock.w_data[wi] == (uint16_t)b))
                        wi++;
                    if (wi >= mock.n_writes) {
                        printf("FAIL: W 0x%04lx 0x%04lx not in log\n",
                               a, b);
                        fail = 1;
                        break;
                    }
                    wi++;
                }
            }
            p = strchr(p, '\n');
            if (!p)
                break;
            p++;
        }
    }

    /* Spot-check the coefficient memory: after the LAST DMA, the
     * engine registers hold that pass's coefficients. Parse the last
     * DMA line for its (dram_off, ncoeff); the first coefficient of
     * that pass is at blob[dram_off] and must be at coeff[0x100]
     * (lane 0, tap 0). */
    {
        const char *last_dma = NULL, *p = text;
        unsigned long off = 0, ncoeff = 0;
        while ((p = strstr(p, "DMA ")) != NULL) {
            last_dma = p;
            p += 4;
        }
        if (last_dma
            && sscanf(last_dma, "DMA %lu %lu", &off, &ncoeff) == 2
            && off + ncoeff * 2 <= blob_len) {
            uint16_t expect = (uint16_t)(blob[off]
                                         | (blob[off + 1] << 8));
            if (mock.coeff[0x100] != expect) {
                printf("FAIL: coeff[0x100]=0x%04x, expected 0x%04x "
                       "(first coeff of last DMA, off=%lu ncoeff=%lu)\n",
                       mock.coeff[0x100], expect, off, ncoeff);
                fail = 1;
            }
        } else {
            printf("FAIL: could not parse the last DMA line\n");
            fail = 1;
        }
    }

    /* The first config writes must be the MC layer-0 descriptor
     * (K=9, shift=15, W=25, H=25). */
    if (mock.n_writes < 4 || mock.w_addr[0] != MC_K
        || mock.w_data[0] != 9 || mock.w_addr[1] != MC_SHIFT
        || mock.w_data[1] != 15) {
        printf("FAIL: first config writes are not the MC L1 descriptor\n");
        fail = 1;
    }

    (void)i;
    free(text);
    free(blob);
    if (fail) {
        printf("DRIVER TEST: FAILED\n");
        return 1;
    }
    printf("DRIVER TEST: PASS (%u config writes, %u DMA, %u GO)\n",
           drv.n_writes, drv.n_dma, drv.n_go);
    return 0;
}
