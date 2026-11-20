/* driver.c -- Stage 5 PS driver skeleton (C).
 *
 * Parses the compiler-emitted program.txt and issues the config
 * sequence. See driver.h for the register map.
 *
 * Instruction format (one per line, '#' = comment):
 *   W <addr> <data>              config-bus write
 *   DMA <dram_off> <ncoeff> <dst_base> <taps> <stride>
 *                                program the weight DMA + wait done
 *   GO                         start the current engine
 *   WAIT                       poll until idle
 *   STREAM <bytes>             frame-stream marker (VDMA, Stages 8-11)
 *
 * Engine tracking: the driver is in MC mode by default. A write to
 * SCHED_LSEL (0x0002) switches it to scheduler mode (single-channel
 * layers). GO writes MC_GO (0x0C) in MC mode, SCHED_CTRL bit0 in
 * scheduler mode. The scheduler is started once, after all layer
 * descriptors are programmed (accel_start_scheduler, called at the
 * end of accel_exec_program when a scheduler descriptor was seen).
 */
#include "driver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WAIT_MAX_POLL 1000000

void accel_driver_init(accel_driver_t *d, const accel_mmio_t *mmio,
                       const uint8_t *weights, size_t weights_len)
{
    memset(d, 0, sizeof(*d));
    d->mmio = *mmio;
    d->weights = weights;
    d->weights_len = weights_len;
}

static void cfg_write(accel_driver_t *d, uint16_t addr, uint16_t data)
{
    d->mmio.cfg_write(d->mmio.ctx, addr, data);
    d->n_writes++;
    /* NOTE: the MC and scheduler register maps OVERLAP (both use
     * 0x0000-0x0012 as offsets within their own module), so the mode
     * is NOT detected from the address -- it is tracked from the
     * "# layer ... (scheduler ...)" / "(conv2d_mc ...)" comment in
     * accel_exec_line. */
}

int accel_wait_idle(accel_driver_t *d)
{
    /* Poll the DMA status (busy) until idle. The engine itself is
     * started by GO and signals done via the scheduler/MC status;
     * here we poll the DMA busy bit as the completion proxy (the
     * DMA is the only long-latency step in this skeleton). */
    uint32_t i;
    for (i = 0; i < WAIT_MAX_POLL; i++) {
        uint16_t st = d->mmio.dma_read(d->mmio.ctx, DMA_STATUS);
        if (!(st & DMA_STATUS_BUSY))
            return 0;
    }
    return -1; /* timeout */
}

int accel_start_scheduler(accel_driver_t *d)
{
    cfg_write(d, SCHED_CTRL, 0x0001); /* bit0 = start */
    d->n_go++;
    return accel_wait_idle(d);
}

/* Parse an unsigned integer with base auto-detection (strtol base 0:
 * "0x..." is hex, otherwise decimal). The emit format mixes bases:
 * W uses 0x..., DMA uses decimal offsets/counts but a 0x... dst base. */
static int parse_u(const char **pp, unsigned long *out)
{
    char *end;
    unsigned long v = strtoul(*pp, &end, 0);
    if (end == *pp)
        return -1;
    *out = v;
    *pp = end;
    return 0;
}

static void skip_ws(const char **pp)
{
    while (**pp == ' ' || **pp == '\t')
        (*pp)++;
}

int accel_exec_line(accel_driver_t *d, const char *line)
{
    char op[16];
    unsigned long a, b, c, e, f;
    const char *p = line;

    /* Skip leading whitespace and comments/blank lines. The layer
     * comment carries the execution path (the MC and scheduler
     * register maps overlap, so the path is tracked here, not from
     * the config address). */
    while (*p == ' ' || *p == '\t')
        p++;
    if (*p == '\0' || *p == '\n')
        return 0;
    if (*p == '#') {
        if (strncmp(p, "# layer", 7) == 0) {
            if (strstr(p, "(scheduler") != NULL) {
                d->sched_mode = 1;
                d->sched_programmed = 1;
            } else if (strstr(p, "(conv2d_mc") != NULL) {
                d->sched_mode = 0;
            }
        }
        return 0;
    }
    if (sscanf(p, "%15s", op) != 1)
        return 0;

    if (strcmp(op, "W") == 0) {
        p += 1; /* skip "W" */
        if (parse_u(&p, &a) != 0)
            return -1;
        skip_ws(&p);
        if (parse_u(&p, &b) != 0)
            return -1;
        cfg_write(d, (uint16_t)a, (uint16_t)b);
        return 0;
    }
    if (strcmp(op, "DMA") == 0) {
        /* DMA <dram_off> <ncoeff> <dst_base> <taps> <stride> */
        p += 3; /* skip "DMA" */
        if (parse_u(&p, &a) != 0)
            return -1;
        skip_ws(&p);
        if (parse_u(&p, &b) != 0)
            return -1;
        skip_ws(&p);
        if (parse_u(&p, &c) != 0)
            return -1;
        skip_ws(&p);
        if (parse_u(&p, &e) != 0)
            return -1;
        skip_ws(&p);
        if (parse_u(&p, &f) != 0)
            return -1;
        /* The weights blob must cover the transfer. */
        if (a + b * 2 > d->weights_len)
            return -2;
        d->mmio.dma_write(d->mmio.ctx, DMA_SRC, (uint16_t)a);
        d->mmio.dma_write(d->mmio.ctx, DMA_COUNT, (uint16_t)b);
        d->mmio.dma_write(d->mmio.ctx, DMA_DST, (uint16_t)c);
        d->mmio.dma_write(d->mmio.ctx, DMA_TAPS, (uint16_t)e);
        d->mmio.dma_write(d->mmio.ctx, DMA_STRIDE, (uint16_t)f);
        d->mmio.dma_write(d->mmio.ctx, DMA_CTRL, DMA_CTRL_START);
        d->n_dma++;
        return accel_wait_idle(d);
    }
    if (strcmp(op, "GO") == 0) {
        if (d->sched_mode) {
            cfg_write(d, SCHED_CTRL, 0x0001);
        } else {
            cfg_write(d, MC_GO, 0x0001);
        }
        d->n_go++;
        return 0;
    }
    if (strcmp(op, "WAIT") == 0) {
        d->n_wait++;
        return accel_wait_idle(d);
    }
    if (strcmp(op, "STREAM") == 0) {
        /* Frame-stream marker: the VDMA streams the input frame
         * (Stages 8-11). In this skeleton it is a no-op that the
         * host test counts. */
        d->n_stream++;
        return 0;
    }
    return -3; /* unknown instruction */
}

int accel_exec_program(accel_driver_t *d, const char *text)
{
    const char *p = text;
    char line[512];
    int rc = 0;

    while (*p) {
        const char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        if (len >= sizeof(line))
            return -4;
        memcpy(line, p, len);
        line[len] = '\0';
        rc = accel_exec_line(d, line);
        if (rc != 0)
            return rc;
        p = nl ? nl + 1 : p + len;
    }
    /* Single-channel path: the scheduler is started once, after all
     * layer descriptors and the first bank are programmed. */
    if (d->sched_programmed)
        rc = accel_start_scheduler(d);
    return rc;
}
