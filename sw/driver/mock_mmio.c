/* mock_mmio.c -- host-test mock of the accelerator MMIO. */
#include "mock_mmio.h"

#include <string.h>

void mock_mmio_init(mock_mmio_t *m, const uint8_t *weights,
                    size_t weights_len)
{
    memset(m, 0, sizeof(*m));
    m->weights = weights;
    m->weights_len = weights_len;
    m->dma_status = DMA_STATUS_DONE; /* idle */
}

void mock_cfg_write(void *ctx, uint16_t addr, uint16_t data)
{
    mock_mmio_t *m = (mock_mmio_t *)ctx;
    if (m->n_writes < MOCK_MAX_WRITES) {
        m->w_addr[m->n_writes] = addr;
        m->w_data[m->n_writes] = data;
        m->n_writes++;
    }
    /* Coefficient memory: the config bus writes coefficients to the
     * engine (0x100+l*128+k) or to a scheduler bank (0x0200+i /
     * 0x0400+i). Mirror them into the mock memory so the test can
     * verify the DMA + driver wrote the right values. */
    if ((uint32_t)addr < MOCK_COEFF_WORDS)
        m->coeff[addr] = data;
}

uint16_t mock_cfg_read(void *ctx, uint16_t addr)
{
    mock_mmio_t *m = (mock_mmio_t *)ctx;
    if ((uint32_t)addr < MOCK_COEFF_WORDS)
        return m->coeff[addr];
    return 0;
}

void mock_dma_write(void *ctx, uint16_t reg, uint16_t data)
{
    mock_mmio_t *m = (mock_mmio_t *)ctx;
    switch (reg) {
    case DMA_SRC:    m->dma_src = data; break;
    case DMA_COUNT:  m->dma_count = data; break;
    case DMA_DST:    m->dma_dst = data; break;
    case DMA_TAPS:   m->dma_taps = data; break;
    case DMA_STRIDE: m->dma_stride = data; break;
    case DMA_CTRL:
        if (data & DMA_CTRL_START) {
            /* Simulate the transfer: ncoeff int16 values from the
             * weights blob at dma_src, written to the coefficient
             * memory at dst, strided by taps/stride (the weight
             * DMA's addressing: lane l at dst + l*stride, tap k at
             * +k). The blob is in hardware order (o, t, l, k), so
             * the i-th coefficient goes to lane i/taps, tap
             * i%taps. */
            uint16_t i;
            m->dma_status = DMA_STATUS_BUSY;
            for (i = 0; i < m->dma_count; i++) {
                size_t off = (size_t)m->dma_src + (size_t)i * 2;
                uint16_t v = 0;
                uint16_t lane, tap, dst;
                if (off + 1 < m->weights_len)
                    v = (uint16_t)(m->weights[off]
                                   | (m->weights[off + 1] << 8));
                lane = (m->dma_taps == 0) ? 0 : (uint16_t)(i / m->dma_taps);
                tap = (m->dma_taps == 0) ? i : (uint16_t)(i % m->dma_taps);
                dst = (uint16_t)(m->dma_dst + lane * m->dma_stride + tap);
                if ((uint32_t)dst < MOCK_COEFF_WORDS)
                    m->coeff[dst] = v;
            }
            m->dma_status = DMA_STATUS_DONE;
        }
        break;
    default:
        break;
    }
}

uint16_t mock_dma_read(void *ctx, uint16_t reg)
{
    mock_mmio_t *m = (mock_mmio_t *)ctx;
    if (reg == DMA_STATUS)
        return m->dma_status;
    return 0;
}
