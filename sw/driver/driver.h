/* driver.h -- Stage 5 PS driver skeleton (C).
 *
 * The driver reads the compiler-emitted program (program.txt +
 * weights.bin) and issues the config sequence to the accelerator:
 * config-bus writes (W), weight-DMA programming (DMA), engine start
 * (GO), completion polling (WAIT), and the frame-stream marker
 * (STREAM, VDMA -- Stages 8-11).
 *
 * This is a skeleton: the MMIO layer is abstracted behind
 * accel_mmio_t so the same driver runs on the PS (real AXI-Lite
 * base addresses) and on the host (mock_mmio.c, for the unit test).
 * Full PS integration (VDMA frame streaming, interrupt handling,
 * multi-frame scheduling) is Stages 8-11 (board).
 *
 * Register map (config bus, 16-bit addr/data):
 *   conv2d_mc (rtl/conv2d_mc.sv): 0x00 K, 0x01 shift, 0x02 W, 0x03 H,
 *     0x04-0x07 pads, 0x08/0x09 strides, 0x0A C_in, 0x0B C_out,
 *     0x0C GO, 0x0D/0x0E out W/H, 0x0F sign, 0x10 fused act,
 *     0x11 slope, 0x100+l*128+k coefficients.
 *   layer_scheduler (rtl/layer_scheduler.sv): 0x0000 ctrl, 0x0001
 *     nlayer, 0x0002 layer select, 0x0003-0x0012 layer fields,
 *     0x0200+i bank A, 0x0400+i bank B.
 *   weight DMA (rtl/weight_dma.sv, AXI-Lite slave at DMA_BASE):
 *     0x00 src (DRAM byte offset), 0x01 count (coefficients),
 *     0x02 dst (config-bus base), 0x03 taps (121 per lane -- the
 *       full 11x11 grid, Stage 6 fix; was K*K),
 *     0x04 stride (address stride between lanes),
 *     0x05 ctrl (bit0 start), 0x06 status (bit0 done, bit1 busy).
 *   The DMA registers are 32-bit (AXI-Lite); the config bus is
 *   16-bit. The blob can exceed 64 KB (SRCNN grid blob is 518 KB),
 *   so the DMA src/count must NOT be truncated to 16 bits.
 */
#ifndef ACCEL_DRIVER_H
#define ACCEL_DRIVER_H

#include <stdint.h>
#include <stddef.h>

/* --- config-bus addresses (conv2d_mc) --- */
#define MC_K          0x00
#define MC_SHIFT      0x01
#define MC_W          0x02
#define MC_H          0x03
#define MC_PT         0x04
#define MC_PB         0x05
#define MC_PL         0x06
#define MC_PR         0x07
#define MC_SX         0x08
#define MC_SY         0x09
#define MC_CIN        0x0A
#define MC_COUT       0x0B
#define MC_GO         0x0C
#define MC_OUTW       0x0D
#define MC_OUTH       0x0E
#define MC_SIGN       0x0F
#define MC_FUSED_ACT  0x10
#define MC_SLOPE      0x11
#define MC_COEFF_BASE 0x100
#define MC_LANE_STRIDE 128

/* --- config-bus addresses (layer_scheduler) --- */
#define SCHED_CTRL    0x0000
#define SCHED_NLAYER  0x0001
#define SCHED_LSEL    0x0002
#define SCHED_BANK_A  0x0200
#define SCHED_BANK_B  0x0400

/* --- weight-DMA registers (AXI-Lite slave, offsets from DMA_BASE) --- */
#define DMA_SRC    0x00
#define DMA_COUNT  0x01
#define DMA_DST    0x02
#define DMA_TAPS   0x03
#define DMA_STRIDE 0x04
#define DMA_CTRL   0x05
#define DMA_STATUS 0x06
#define DMA_CTRL_START 0x0001
#define DMA_STATUS_DONE 0x0001
#define DMA_STATUS_BUSY 0x0002

/* --- MMIO abstraction ---------------------------------------------- */
/* write(addr, data): config-bus write (16-bit). read(addr): config-bus
 * read (16-bit). dma_write/dma_read: the DMA register block (32-bit
 * data -- the AXI-Lite slave is 32-bit; truncating to 16 bits would
 * break blobs > 64 KB). On the PS these are AXI-Lite accesses; on
 * the host they are the mock. */
typedef struct {
    void (*cfg_write)(void *ctx, uint16_t addr, uint16_t data);
    uint16_t (*cfg_read)(void *ctx, uint16_t addr);
    void (*dma_write)(void *ctx, uint16_t reg, uint32_t data);
    uint32_t (*dma_read)(void *ctx, uint16_t reg);
    void *ctx;
} accel_mmio_t;

/* --- driver state -------------------------------------------------- */
typedef struct {
    accel_mmio_t mmio;
    const uint8_t *weights;   /* weights.bin image (int16 LE) */
    size_t weights_len;       /* bytes */
    /* Statistics (for the host test + profiling). */
    uint32_t n_writes;        /* config-bus writes issued */
    uint32_t n_dma;           /* DMA transfers programmed */
    uint32_t n_go;            /* GO issued */
    uint32_t n_wait;          /* WAIT polls issued */
    uint32_t n_stream;        /* STREAM markers seen */
    int sched_programmed;     /* any scheduler descriptor written */
    int sched_mode;           /* current layer uses the scheduler path */
} accel_driver_t;

void accel_driver_init(accel_driver_t *d, const accel_mmio_t *mmio,
                       const uint8_t *weights, size_t weights_len);

/* Execute one instruction line from program.txt.
 * Returns 0 on success, <0 on a parse/execution error. */
int accel_exec_line(accel_driver_t *d, const char *line);

/* Execute a whole program.txt (already loaded into memory).
 * Returns 0 on success, <0 on the first failing line. */
int accel_exec_program(accel_driver_t *d, const char *text);

/* Start the scheduler (single-channel path) after all layer
 * descriptors are programmed. Called automatically at the end of
 * accel_exec_program if a scheduler descriptor was written. */
int accel_start_scheduler(accel_driver_t *d);

/* Poll the engine until idle (WAIT). Returns 0 when idle, <0 on
 * timeout (the mock never times out; the PS uses a bounded poll). */
int accel_wait_idle(accel_driver_t *d);

#endif /* ACCEL_DRIVER_H */
