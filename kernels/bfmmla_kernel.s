        .type bfmmla_kernel, %function
        .global bfmmla_kernel

bfmmla_kernel:
        ptrue p0.d // only set every eigth bit

        ld1h {z0.h}, p0/z, [x0]
        ld1h {z1.h}, p0/z, [x1]
        ld1w {z2.s}, p0/z, [x2]

        BFMMLA z2.S, z1.H, z0.H

        st1w {z2.s}, p0, [x2]

        ret
