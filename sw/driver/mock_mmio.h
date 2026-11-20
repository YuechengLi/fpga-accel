/* mock_mmio.h -- host-test mock of the accelerator MMIO.
 *
 * Records every config-bus write, and simulates the weight DMA:
 * on DMA_CTRL start it copies ncoeff int16 coefficients from the
 * weights blob (at the DMA source offset) into the mock coefficient
 * memory at the destination config-bus address (strided by taps /
 * stride, matching rtl/weight_dma.sv), then reports done/idle.
 * The engine (GO/WAIT) is always idle in the mock.
 */
#ifndef MOCK_MMIO_H
#define MOCK_MMIO_H

#include <stdint.h>
#include <stddef.h>
#include "driver.h"

#define MOCK_MAX_WRITES 65536
#define MOCK_COEFF_WORDS 65536  /* 16-bit words of coefficient memory */

typedef struct {
    /* config-write log */
    uint16_t w_addr[MOCK_MAX_WRITES];
    uint16_t w_data[MOCK_MAX_WRITES];
    uint32_t n_writes;
    /* coefficient memory (the engine's / banks' registers) */
    uint16_t coeff[MOCK_COEFF_WORDS];
    /* DMA registers (32-bit -- the AXI-Lite slave is 32-bit; the
     * blob can exceed 64 KB, so src/count must not be 16-bit). */
    uint32_t dma_src, dma_count, dma_dst, dma_taps, dma_stride;
    uint32_t dma_status;
    /* weights blob (for the DMA to copy from) */
    const uint8_t *weights;
    size_t weights_len;
} mock_mmio_t;

void mock_mmio_init(mock_mmio_t *m, const uint8_t *weights,
                    size_t weights_len);

/* accel_mmio_t vtable (pass &mock as ctx). */
void mock_cfg_write(void *ctx, uint16_t addr, uint16_t data);
uint16_t mock_cfg_read(void *ctx, uint16_t addr);
void mock_dma_write(void *ctx, uint16_t reg, uint32_t data);
uint32_t mock_dma_read(void *ctx, uint16_t reg);

#endif /* MOCK_MMIO_H */
