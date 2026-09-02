#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-side Vallado SGP4 initialization formulas.

The equations follow Vallado et al. (2006) and are used only while preparing
a fixed propagation plan; transformed propagation is implemented below with
JAX primitives.
"""

from __future__ import annotations

from math import atan2, cos, fabs, pi, sin, sqrt

import jax
import jax.numpy as jnp


deg2rad = pi / 180.0
twopi = 2.0 * pi


def _dpper(satrec, inclo, init, ep, inclp, nodep, argpp, mp, opsmode):
    e3 = satrec.e3
    ee2 = satrec.ee2
    peo = satrec.peo
    pgho = satrec.pgho
    pho = satrec.pho
    pinco = satrec.pinco
    plo = satrec.plo
    se2 = satrec.se2
    se3 = satrec.se3
    sgh2 = satrec.sgh2
    sgh3 = satrec.sgh3
    sgh4 = satrec.sgh4
    sh2 = satrec.sh2
    sh3 = satrec.sh3
    si2 = satrec.si2
    si3 = satrec.si3
    sl2 = satrec.sl2
    sl3 = satrec.sl3
    sl4 = satrec.sl4
    t = satrec.t
    xgh2 = satrec.xgh2
    xgh3 = satrec.xgh3
    xgh4 = satrec.xgh4
    xh2 = satrec.xh2
    xh3 = satrec.xh3
    xi2 = satrec.xi2
    xi3 = satrec.xi3
    xl2 = satrec.xl2
    xl3 = satrec.xl3
    xl4 = satrec.xl4
    zmol = satrec.zmol
    zmos = satrec.zmos
    zns = 1.19459e-05
    zes = 0.01675
    znl = 0.00015835218
    zel = 0.0549
    zm = zmos + zns * t
    if init == "y":
        zm = zmos
    zf = zm + 2.0 * zes * sin(zm)
    sinzf = sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * cos(zf)
    ses = se2 * f2 + se3 * f3
    sis = si2 * f2 + si3 * f3
    sls = sl2 * f2 + sl3 * f3 + sl4 * sinzf
    sghs = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf
    shs = sh2 * f2 + sh3 * f3
    zm = zmol + znl * t
    if init == "y":
        zm = zmol
    zf = zm + 2.0 * zel * sin(zm)
    sinzf = sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * cos(zf)
    sel = ee2 * f2 + e3 * f3
    sil = xi2 * f2 + xi3 * f3
    sll = xl2 * f2 + xl3 * f3 + xl4 * sinzf
    sghl = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf
    shll = xh2 * f2 + xh3 * f3
    pe = ses + sel
    pinc = sis + sil
    pl = sls + sll
    pgh = sghs + sghl
    ph = shs + shll
    if init == "n":
        pe = pe - peo
        pinc = pinc - pinco
        pl = pl - plo
        pgh = pgh - pgho
        ph = ph - pho
        inclp = inclp + pinc
        ep = ep + pe
        sinip = sin(inclp)
        cosip = cos(inclp)
        if inclp >= 0.2:
            ph /= sinip
            pgh -= cosip * ph
            argpp += pgh
            nodep += ph
            mp += pl
        else:
            sinop = sin(nodep)
            cosop = cos(nodep)
            alfdp = sinip * sinop
            betdp = sinip * cosop
            dalf = ph * cosop + pinc * cosip * sinop
            dbet = -ph * sinop + pinc * cosip * cosop
            alfdp = alfdp + dalf
            betdp = betdp + dbet
            nodep = nodep % twopi if nodep >= 0.0 else -(-nodep % twopi)
            if nodep < 0.0 and opsmode == "a":
                nodep = nodep + twopi
            xls = mp + argpp + pl + pgh + (cosip - pinc * sinip) * nodep
            xnoh = nodep
            nodep = atan2(alfdp, betdp)
            if nodep < 0.0 and opsmode == "a":
                nodep = nodep + twopi
            if fabs(xnoh - nodep) > pi:
                if nodep < xnoh:
                    nodep = nodep + twopi
                else:
                    nodep = nodep - twopi
            mp += pl
            argpp = xls - mp - cosip * nodep
    return (ep, inclp, nodep, argpp, mp)


def _dscom(
    epoch,
    ep,
    argpp,
    tc,
    inclp,
    nodep,
    np,
    e3,
    ee2,
    peo,
    pgho,
    pho,
    pinco,
    plo,
    se2,
    se3,
    sgh2,
    sgh3,
    sgh4,
    sh2,
    sh3,
    si2,
    si3,
    sl2,
    sl3,
    sl4,
    xgh2,
    xgh3,
    xgh4,
    xh2,
    xh3,
    xi2,
    xi3,
    xl2,
    xl3,
    xl4,
    zmol,
    zmos,
):
    zes = 0.01675
    zel = 0.0549
    c1ss = 2.9864797e-06
    c1l = 4.7968065e-07
    zsinis = 0.39785416
    zcosis = 0.91744867
    zcosgs = 0.1945905
    zsings = -0.98088458
    nm = np
    em = ep
    snodm = sin(nodep)
    cnodm = cos(nodep)
    sinomm = sin(argpp)
    cosomm = cos(argpp)
    sinim = sin(inclp)
    cosim = cos(inclp)
    emsq = em * em
    betasq = 1.0 - emsq
    rtemsq = sqrt(betasq)
    peo = 0.0
    pinco = 0.0
    plo = 0.0
    pgho = 0.0
    pho = 0.0
    day = epoch + 18261.5 + tc / 1440.0
    xnodce = (4.523602 - 0.00092422029 * day) % twopi
    stem = sin(xnodce)
    ctem = cos(xnodce)
    zcosil = 0.91375164 - 0.03568096 * ctem
    zsinil = sqrt(1.0 - zcosil * zcosil)
    zsinhl = 0.089683511 * stem / zsinil
    zcoshl = sqrt(1.0 - zsinhl * zsinhl)
    gam = 5.8351514 + 0.001944368 * day
    zx = 0.39785416 * stem / zsinil
    zy = zcoshl * ctem + 0.91744867 * zsinhl * stem
    zx = atan2(zx, zy)
    zx = gam + zx - xnodce
    zcosgl = cos(zx)
    zsingl = sin(zx)
    zcosg = zcosgs
    zsing = zsings
    zcosi = zcosis
    zsini = zsinis
    zcosh = cnodm
    zsinh = snodm
    cc = c1ss
    xnoi = 1.0 / nm
    for lsflg in (1, 2):
        a1 = zcosg * zcosh + zsing * zcosi * zsinh
        a3 = -zsing * zcosh + zcosg * zcosi * zsinh
        a7 = -zcosg * zsinh + zsing * zcosi * zcosh
        a8 = zsing * zsini
        a9 = zsing * zsinh + zcosg * zcosi * zcosh
        a10 = zcosg * zsini
        a2 = cosim * a7 + sinim * a8
        a4 = cosim * a9 + sinim * a10
        a5 = -sinim * a7 + cosim * a8
        a6 = -sinim * a9 + cosim * a10
        x1 = a1 * cosomm + a2 * sinomm
        x2 = a3 * cosomm + a4 * sinomm
        x3 = -a1 * sinomm + a2 * cosomm
        x4 = -a3 * sinomm + a4 * cosomm
        x5 = a5 * sinomm
        x6 = a6 * sinomm
        x7 = a5 * cosomm
        x8 = a6 * cosomm
        z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3
        z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4
        z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4
        z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * emsq
        z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * emsq
        z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * emsq
        z11 = -6.0 * a1 * a5 + emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5)
        z12 = -6.0 * (a1 * a6 + a3 * a5) + emsq * (
            -24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5)
        )
        z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6)
        z21 = 6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7)
        z22 = 6.0 * (a4 * a5 + a2 * a6) + emsq * (
            24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8)
        )
        z23 = 6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8)
        z1 = z1 + z1 + betasq * z31
        z2 = z2 + z2 + betasq * z32
        z3 = z3 + z3 + betasq * z33
        s3 = cc * xnoi
        s2 = -0.5 * s3 / rtemsq
        s4 = s3 * rtemsq
        s1 = -15.0 * em * s4
        s5 = x1 * x3 + x2 * x4
        s6 = x2 * x3 + x1 * x4
        s7 = x2 * x4 - x1 * x3
        if lsflg == 1:
            ss1 = s1
            ss2 = s2
            ss3 = s3
            ss4 = s4
            ss5 = s5
            ss6 = s6
            ss7 = s7
            sz1 = z1
            sz2 = z2
            sz3 = z3
            sz11 = z11
            sz12 = z12
            sz13 = z13
            sz21 = z21
            sz22 = z22
            sz23 = z23
            sz31 = z31
            sz32 = z32
            sz33 = z33
            zcosg = zcosgl
            zsing = zsingl
            zcosi = zcosil
            zsini = zsinil
            zcosh = zcoshl * cnodm + zsinhl * snodm
            zsinh = snodm * zcoshl - cnodm * zsinhl
            cc = c1l
    zmol = (4.7199672 + 0.2299715 * day - gam) % twopi
    zmos = (6.2565837 + 0.017201977 * day) % twopi
    se2 = 2.0 * ss1 * ss6
    se3 = 2.0 * ss1 * ss7
    si2 = 2.0 * ss2 * sz12
    si3 = 2.0 * ss2 * (sz13 - sz11)
    sl2 = -2.0 * ss3 * sz2
    sl3 = -2.0 * ss3 * (sz3 - sz1)
    sl4 = -2.0 * ss3 * (-21.0 - 9.0 * emsq) * zes
    sgh2 = 2.0 * ss4 * sz32
    sgh3 = 2.0 * ss4 * (sz33 - sz31)
    sgh4 = -18.0 * ss4 * zes
    sh2 = -2.0 * ss2 * sz22
    sh3 = -2.0 * ss2 * (sz23 - sz21)
    ee2 = 2.0 * s1 * s6
    e3 = 2.0 * s1 * s7
    xi2 = 2.0 * s2 * z12
    xi3 = 2.0 * s2 * (z13 - z11)
    xl2 = -2.0 * s3 * z2
    xl3 = -2.0 * s3 * (z3 - z1)
    xl4 = -2.0 * s3 * (-21.0 - 9.0 * emsq) * zel
    xgh2 = 2.0 * s4 * z32
    xgh3 = 2.0 * s4 * (z33 - z31)
    xgh4 = -18.0 * s4 * zel
    xh2 = -2.0 * s2 * z22
    xh3 = -2.0 * s2 * (z23 - z21)
    return (
        snodm,
        cnodm,
        sinim,
        cosim,
        sinomm,
        cosomm,
        day,
        e3,
        ee2,
        em,
        emsq,
        gam,
        peo,
        pgho,
        pho,
        pinco,
        plo,
        rtemsq,
        se2,
        se3,
        sgh2,
        sgh3,
        sgh4,
        sh2,
        sh3,
        si2,
        si3,
        sl2,
        sl3,
        sl4,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        ss1,
        ss2,
        ss3,
        ss4,
        ss5,
        ss6,
        ss7,
        sz1,
        sz2,
        sz3,
        sz11,
        sz12,
        sz13,
        sz21,
        sz22,
        sz23,
        sz31,
        sz32,
        sz33,
        xgh2,
        xgh3,
        xgh4,
        xh2,
        xh3,
        xi2,
        xi3,
        xl2,
        xl3,
        xl4,
        nm,
        z1,
        z2,
        z3,
        z11,
        z12,
        z13,
        z21,
        z22,
        z23,
        z31,
        z32,
        z33,
        zmol,
        zmos,
    )


def _dsinit(
    xke,
    cosim,
    emsq,
    argpo,
    s1,
    s2,
    s3,
    s4,
    s5,
    sinim,
    ss1,
    ss2,
    ss3,
    ss4,
    ss5,
    sz1,
    sz3,
    sz11,
    sz13,
    sz21,
    sz23,
    sz31,
    sz33,
    t,
    tc,
    gsto,
    mo,
    mdot,
    no,
    nodeo,
    nodedot,
    xpidot,
    z1,
    z3,
    z11,
    z13,
    z21,
    z23,
    z31,
    z33,
    ecco,
    eccsq,
    em,
    argpm,
    inclm,
    mm,
    nm,
    nodem,
    irez,
    atime,
    d2201,
    d2211,
    d3210,
    d3222,
    d4410,
    d4422,
    d5220,
    d5232,
    d5421,
    d5433,
    dedt,
    didt,
    dmdt,
    dnodt,
    domdt,
    del1,
    del2,
    del3,
    xfact,
    xlamo,
    xli,
    xni,
):
    q22 = 1.7891679e-06
    q31 = 2.1460748e-06
    q33 = 2.2123015e-07
    root22 = 1.7891679e-06
    root44 = 7.3636953e-09
    root54 = 2.1765803e-09
    rptim = 0.0043752690880113
    root32 = 3.7393792e-07
    root52 = 1.1428639e-07
    x2o3 = 2.0 / 3.0
    znl = 0.00015835218
    zns = 1.19459e-05
    irez = 0
    if 0.0034906585 < nm < 0.0052359877:
        irez = 1
    if 0.00826 <= nm <= 0.00924 and em >= 0.5:
        irez = 2
    ses = ss1 * zns * ss5
    sis = ss2 * zns * (sz11 + sz13)
    sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq)
    sghs = ss4 * zns * (sz31 + sz33 - 6.0)
    shs = -zns * ss2 * (sz21 + sz23)
    if inclm < 0.052359877 or inclm > pi - 0.052359877:
        shs = 0.0
    if sinim != 0.0:
        shs = shs / sinim
    sgs = sghs - cosim * shs
    dedt = ses + s1 * znl * s5
    didt = sis + s2 * znl * (z11 + z13)
    dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq)
    sghl = s4 * znl * (z31 + z33 - 6.0)
    shll = -znl * s2 * (z21 + z23)
    if inclm < 0.052359877 or inclm > pi - 0.052359877:
        shll = 0.0
    domdt = sgs + sghl
    dnodt = shs
    if sinim != 0.0:
        domdt = domdt - cosim / sinim * shll
        dnodt = dnodt + shll / sinim
    dndt = 0.0
    theta = (gsto + tc * rptim) % twopi
    em = em + dedt * t
    inclm = inclm + didt * t
    argpm = argpm + domdt * t
    nodem = nodem + dnodt * t
    mm = mm + dmdt * t
    if irez != 0:
        aonv = pow(nm / xke, x2o3)
        if irez == 2:
            cosisq = cosim * cosim
            emo = em
            em = ecco
            emsqo = emsq
            emsq = eccsq
            eoc = em * emsq
            g201 = -0.306 - (em - 0.64) * 0.44
            if em <= 0.65:
                g211 = 3.616 - 13.247 * em + 16.29 * emsq
                g310 = -19.302 + 117.39 * em - 228.419 * emsq + 156.591 * eoc
                g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc
                g410 = -41.122 + 242.694 * em - 471.094 * emsq + 313.953 * eoc
                g422 = -146.407 + 841.88 * em - 1629.014 * emsq + 1083.435 * eoc
                g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.276 * eoc
            else:
                g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc
                g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc
                g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc
                g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc
                g422 = -3581.69 + 16178.11 * em - 24462.77 * emsq + 12422.52 * eoc
                if em > 0.715:
                    g520 = -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc
                else:
                    g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq
            if em < 0.7:
                g533 = -919.2277 + 4988.61 * em - 9064.77 * emsq + 5542.21 * eoc
                g521 = -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc
                g532 = -853.666 + 4690.25 * em - 8624.77 * emsq + 5341.4 * eoc
            else:
                g533 = -37995.78 + 161616.52 * em - 229838.2 * emsq + 109377.94 * eoc
                g521 = -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc
                g532 = -40023.88 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc
            sini2 = sinim * sinim
            f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq)
            f221 = 1.5 * sini2
            f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq)
            f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq)
            f441 = 35.0 * sini2 * f220
            f442 = 39.375 * sini2 * sini2
            f522 = (
                9.84375
                * sinim
                * (
                    sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq)
                    + 0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq)
                )
            )
            f523 = sinim * (
                4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq)
                + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq)
            )
            f542 = (
                29.53125
                * sinim
                * (2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq))
            )
            f543 = (
                29.53125
                * sinim
                * (-2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq))
            )
            xno2 = nm * nm
            ainv2 = aonv * aonv
            temp1 = 3.0 * xno2 * ainv2
            temp = temp1 * root22
            d2201 = temp * f220 * g201
            d2211 = temp * f221 * g211
            temp1 = temp1 * aonv
            temp = temp1 * root32
            d3210 = temp * f321 * g310
            d3222 = temp * f322 * g322
            temp1 = temp1 * aonv
            temp = 2.0 * temp1 * root44
            d4410 = temp * f441 * g410
            d4422 = temp * f442 * g422
            temp1 = temp1 * aonv
            temp = temp1 * root52
            d5220 = temp * f522 * g520
            d5232 = temp * f523 * g532
            temp = 2.0 * temp1 * root54
            d5421 = temp * f542 * g521
            d5433 = temp * f543 * g533
            xlamo = (mo + nodeo + nodeo - theta - theta) % twopi
            xfact = mdot + dmdt + 2.0 * (nodedot + dnodt - rptim) - no
            em = emo
            emsq = emsqo
        if irez == 1:
            g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq)
            g310 = 1.0 + 2.0 * emsq
            g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq)
            f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim)
            f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim)
            f330 = 1.0 + cosim
            f330 = 1.875 * f330 * f330 * f330
            del1 = 3.0 * nm * nm * aonv * aonv
            del2 = 2.0 * del1 * f220 * g200 * q22
            del3 = 3.0 * del1 * f330 * g300 * q33 * aonv
            del1 = del1 * f311 * g310 * q31 * aonv
            xlamo = (mo + nodeo + argpo - theta) % twopi
            xfact = mdot + xpidot - rptim + dmdt + domdt + dnodt - no
        xli = xlamo
        xni = no
        atime = 0.0
        nm = no + dndt
    return (
        em,
        argpm,
        inclm,
        mm,
        nm,
        nodem,
        irez,
        atime,
        d2201,
        d2211,
        d3210,
        d3222,
        d4410,
        d4422,
        d5220,
        d5232,
        d5421,
        d5433,
        dedt,
        didt,
        dmdt,
        dndt,
        dnodt,
        domdt,
        del1,
        del2,
        del3,
        xfact,
        xlamo,
        xli,
        xni,
    )


def _initl(xke, j2, ecco, epoch, inclo, no, method, opsmode):
    x2o3 = 2.0 / 3.0
    eccsq = ecco * ecco
    omeosq = 1.0 - eccsq
    rteosq = sqrt(omeosq)
    cosio = cos(inclo)
    cosio2 = cosio * cosio
    ak = pow(xke / no, x2o3)
    d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
    del_ = d1 / (ak * ak)
    adel = ak * (1.0 - del_ * del_ - del_ * (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
    del_ = d1 / (adel * adel)
    no = no / (1.0 + del_)
    ao = pow(xke / no, x2o3)
    sinio = sin(inclo)
    po = ao * omeosq
    con42 = 1.0 - 5.0 * cosio2
    con41 = -con42 - cosio2 - cosio2
    ainv = 1.0 / ao
    posq = po * po
    rp = ao * (1.0 - ecco)
    method = "n"
    if opsmode == "a":
        ts70 = epoch - 7305.0
        ds70 = (ts70 + 1e-08) // 1.0
        tfrac = ts70 - ds70
        c1 = 0.017202791694070362
        thgr70 = 1.7321343856509375
        fk5r = 5.075514194322695e-15
        c1p2p = c1 + twopi
        gsto = (thgr70 + c1 * ds70 + c1p2p * tfrac + ts70 * ts70 * fk5r) % twopi
        if gsto < 0.0:
            gsto = gsto + twopi
    else:
        gsto = _gstime(epoch + 2433281.5)
    return (
        no,
        method,
        ainv,
        ao,
        con41,
        con42,
        cosio,
        cosio2,
        eccsq,
        omeosq,
        posq,
        rp,
        rteosq,
        sinio,
        gsto,
    )


def sgp4init(
    whichconst,
    opsmode,
    satn,
    epoch,
    xbstar,
    xndot,
    xnddot,
    xecco,
    xargpo,
    xinclo,
    xmo,
    xno_kozai,
    xnodeo,
    satrec,
):
    temp4 = 1.5e-12
    satrec.isimp = 0
    satrec.method = "n"
    satrec.aycof = 0.0
    satrec.con41 = 0.0
    satrec.cc1 = 0.0
    satrec.cc4 = 0.0
    satrec.cc5 = 0.0
    satrec.d2 = 0.0
    satrec.d3 = 0.0
    satrec.d4 = 0.0
    satrec.delmo = 0.0
    satrec.eta = 0.0
    satrec.argpdot = 0.0
    satrec.omgcof = 0.0
    satrec.sinmao = 0.0
    satrec.t = 0.0
    satrec.t2cof = 0.0
    satrec.t3cof = 0.0
    satrec.t4cof = 0.0
    satrec.t5cof = 0.0
    satrec.x1mth2 = 0.0
    satrec.x7thm1 = 0.0
    satrec.mdot = 0.0
    satrec.nodedot = 0.0
    satrec.xlcof = 0.0
    satrec.xmcof = 0.0
    satrec.nodecf = 0.0
    satrec.irez = 0
    satrec.d2201 = 0.0
    satrec.d2211 = 0.0
    satrec.d3210 = 0.0
    satrec.d3222 = 0.0
    satrec.d4410 = 0.0
    satrec.d4422 = 0.0
    satrec.d5220 = 0.0
    satrec.d5232 = 0.0
    satrec.d5421 = 0.0
    satrec.d5433 = 0.0
    satrec.dedt = 0.0
    satrec.del1 = 0.0
    satrec.del2 = 0.0
    satrec.del3 = 0.0
    satrec.didt = 0.0
    satrec.dmdt = 0.0
    satrec.dnodt = 0.0
    satrec.domdt = 0.0
    satrec.e3 = 0.0
    satrec.ee2 = 0.0
    satrec.peo = 0.0
    satrec.pgho = 0.0
    satrec.pho = 0.0
    satrec.pinco = 0.0
    satrec.plo = 0.0
    satrec.se2 = 0.0
    satrec.se3 = 0.0
    satrec.sgh2 = 0.0
    satrec.sgh3 = 0.0
    satrec.sgh4 = 0.0
    satrec.sh2 = 0.0
    satrec.sh3 = 0.0
    satrec.si2 = 0.0
    satrec.si3 = 0.0
    satrec.sl2 = 0.0
    satrec.sl3 = 0.0
    satrec.sl4 = 0.0
    satrec.gsto = 0.0
    satrec.xfact = 0.0
    satrec.xgh2 = 0.0
    satrec.xgh3 = 0.0
    satrec.xgh4 = 0.0
    satrec.xh2 = 0.0
    satrec.xh3 = 0.0
    satrec.xi2 = 0.0
    satrec.xi3 = 0.0
    satrec.xl2 = 0.0
    satrec.xl3 = 0.0
    satrec.xl4 = 0.0
    satrec.xlamo = 0.0
    satrec.zmol = 0.0
    satrec.zmos = 0.0
    satrec.atime = 0.0
    satrec.xli = 0.0
    satrec.xni = 0.0
    (
        satrec.tumin,
        satrec.mu,
        satrec.radiusearthkm,
        satrec.xke,
        satrec.j2,
        satrec.j3,
        satrec.j4,
        satrec.j3oj2,
    ) = whichconst
    if isinstance(satn, int):
        satn = str(satn)
    satrec.error = 0
    satrec.operationmode = opsmode
    satrec.satnum_str = satn
    satrec.classification = "U"
    satrec.bstar = xbstar
    satrec.ndot = xndot
    satrec.nddot = xnddot
    satrec.ecco = xecco
    satrec.argpo = xargpo
    satrec.inclo = xinclo
    satrec.mo = xmo
    satrec.no_kozai = xno_kozai
    satrec.nodeo = xnodeo
    satrec.am = 0.0
    satrec.em = 0.0
    satrec.im = 0.0
    satrec.Om = 0.0
    satrec.mm = 0.0
    satrec.nm = 0.0
    ss = 78.0 / satrec.radiusearthkm + 1.0
    qzms2ttemp = (120.0 - 78.0) / satrec.radiusearthkm
    qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp
    x2o3 = 2.0 / 3.0
    satrec.init = "y"
    satrec.t = 0.0
    (
        satrec.no_unkozai,
        method,
        ainv,
        ao,
        satrec.con41,
        con42,
        cosio,
        cosio2,
        eccsq,
        omeosq,
        posq,
        rp,
        rteosq,
        sinio,
        satrec.gsto,
    ) = _initl(
        satrec.xke,
        satrec.j2,
        satrec.ecco,
        epoch,
        satrec.inclo,
        satrec.no_kozai,
        satrec.method,
        satrec.operationmode,
    )
    satrec.a = pow(satrec.no_unkozai * satrec.tumin, -2.0 / 3.0)
    satrec.alta = satrec.a * (1.0 + satrec.ecco) - 1.0
    satrec.altp = satrec.a * (1.0 - satrec.ecco) - 1.0
    if omeosq >= 0.0 or satrec.no_unkozai >= 0.0:
        satrec.isimp = 0
        if rp < 220.0 / satrec.radiusearthkm + 1.0:
            satrec.isimp = 1
        sfour = ss
        qzms24 = qzms2t
        perige = (rp - 1.0) * satrec.radiusearthkm
        if perige < 156.0:
            sfour = perige - 78.0
            if perige < 98.0:
                sfour = 20.0
            qzms24temp = (120.0 - sfour) / satrec.radiusearthkm
            qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp
            sfour = sfour / satrec.radiusearthkm + 1.0
        pinvsq = 1.0 / posq
        tsi = 1.0 / (ao - sfour)
        satrec.eta = ao * satrec.ecco * tsi
        etasq = satrec.eta * satrec.eta
        eeta = satrec.ecco * satrec.eta
        psisq = fabs(1.0 - etasq)
        coef = qzms24 * pow(tsi, 4.0)
        coef1 = coef / pow(psisq, 3.5)
        cc2 = (
            coef1
            * satrec.no_unkozai
            * (
                ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq))
                + 0.375
                * satrec.j2
                * tsi
                / psisq
                * satrec.con41
                * (8.0 + 3.0 * etasq * (8.0 + etasq))
            )
        )
        satrec.cc1 = satrec.bstar * cc2
        cc3 = 0.0
        if satrec.ecco > 0.0001:
            cc3 = (
                -2.0 * coef * tsi * satrec.j3oj2 * satrec.no_unkozai * sinio / satrec.ecco
            )
        satrec.x1mth2 = 1.0 - cosio2
        satrec.cc4 = (
            2.0
            * satrec.no_unkozai
            * coef1
            * ao
            * omeosq
            * (
                satrec.eta * (2.0 + 0.5 * etasq)
                + satrec.ecco * (0.5 + 2.0 * etasq)
                - satrec.j2
                * tsi
                / (ao * psisq)
                * (
                    -3.0 * satrec.con41 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta))
                    + 0.75
                    * satrec.x1mth2
                    * (2.0 * etasq - eeta * (1.0 + etasq))
                    * cos(2.0 * satrec.argpo)
                )
            )
        )
        satrec.cc5 = (
            2.0 * coef1 * ao * omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq)
        )
        cosio4 = cosio2 * cosio2
        temp1 = 1.5 * satrec.j2 * pinvsq * satrec.no_unkozai
        temp2 = 0.5 * temp1 * satrec.j2 * pinvsq
        temp3 = -0.46875 * satrec.j4 * pinvsq * pinvsq * satrec.no_unkozai
        satrec.mdot = (
            satrec.no_unkozai
            + 0.5 * temp1 * rteosq * satrec.con41
            + 0.0625 * temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4)
        )
        satrec.argpdot = (
            -0.5 * temp1 * con42
            + 0.0625 * temp2 * (7.0 - 114.0 * cosio2 + 395.0 * cosio4)
            + temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4)
        )
        xhdot1 = -temp1 * cosio
        satrec.nodedot = (
            xhdot1
            + (0.5 * temp2 * (4.0 - 19.0 * cosio2) + 2.0 * temp3 * (3.0 - 7.0 * cosio2))
            * cosio
        )
        xpidot = satrec.argpdot + satrec.nodedot
        satrec.omgcof = satrec.bstar * cc3 * cos(satrec.argpo)
        satrec.xmcof = 0.0
        if satrec.ecco > 0.0001:
            satrec.xmcof = -x2o3 * coef * satrec.bstar / eeta
        satrec.nodecf = 3.5 * omeosq * xhdot1 * satrec.cc1
        satrec.t2cof = 1.5 * satrec.cc1
        if fabs(cosio + 1.0) > 1.5e-12:
            satrec.xlcof = (
                -0.25 * satrec.j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio)
            )
        else:
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4
        satrec.aycof = -0.5 * satrec.j3oj2 * sinio
        delmotemp = 1.0 + satrec.eta * cos(satrec.mo)
        satrec.delmo = delmotemp * delmotemp * delmotemp
        satrec.sinmao = sin(satrec.mo)
        satrec.x7thm1 = 7.0 * cosio2 - 1.0
        if 2 * pi / satrec.no_unkozai >= 225.0:
            satrec.method = "d"
            satrec.isimp = 1
            tc = 0.0
            inclm = satrec.inclo
            (
                snodm,
                cnodm,
                sinim,
                cosim,
                sinomm,
                cosomm,
                day,
                satrec.e3,
                satrec.ee2,
                em,
                emsq,
                gam,
                satrec.peo,
                satrec.pgho,
                satrec.pho,
                satrec.pinco,
                satrec.plo,
                rtemsq,
                satrec.se2,
                satrec.se3,
                satrec.sgh2,
                satrec.sgh3,
                satrec.sgh4,
                satrec.sh2,
                satrec.sh3,
                satrec.si2,
                satrec.si3,
                satrec.sl2,
                satrec.sl3,
                satrec.sl4,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6,
                s7,
                ss1,
                ss2,
                ss3,
                ss4,
                ss5,
                ss6,
                ss7,
                sz1,
                sz2,
                sz3,
                sz11,
                sz12,
                sz13,
                sz21,
                sz22,
                sz23,
                sz31,
                sz32,
                sz33,
                satrec.xgh2,
                satrec.xgh3,
                satrec.xgh4,
                satrec.xh2,
                satrec.xh3,
                satrec.xi2,
                satrec.xi3,
                satrec.xl2,
                satrec.xl3,
                satrec.xl4,
                nm,
                z1,
                z2,
                z3,
                z11,
                z12,
                z13,
                z21,
                z22,
                z23,
                z31,
                z32,
                z33,
                satrec.zmol,
                satrec.zmos,
            ) = _dscom(
                epoch,
                satrec.ecco,
                satrec.argpo,
                tc,
                satrec.inclo,
                satrec.nodeo,
                satrec.no_unkozai,
                satrec.e3,
                satrec.ee2,
                satrec.peo,
                satrec.pgho,
                satrec.pho,
                satrec.pinco,
                satrec.plo,
                satrec.se2,
                satrec.se3,
                satrec.sgh2,
                satrec.sgh3,
                satrec.sgh4,
                satrec.sh2,
                satrec.sh3,
                satrec.si2,
                satrec.si3,
                satrec.sl2,
                satrec.sl3,
                satrec.sl4,
                satrec.xgh2,
                satrec.xgh3,
                satrec.xgh4,
                satrec.xh2,
                satrec.xh3,
                satrec.xi2,
                satrec.xi3,
                satrec.xl2,
                satrec.xl3,
                satrec.xl4,
                satrec.zmol,
                satrec.zmos,
            )
            satrec.ecco, satrec.inclo, satrec.nodeo, satrec.argpo, satrec.mo = _dpper(
                satrec,
                inclm,
                satrec.init,
                satrec.ecco,
                satrec.inclo,
                satrec.nodeo,
                satrec.argpo,
                satrec.mo,
                satrec.operationmode,
            )
            argpm = 0.0
            nodem = 0.0
            mm = 0.0
            (
                em,
                argpm,
                inclm,
                mm,
                nm,
                nodem,
                satrec.irez,
                satrec.atime,
                satrec.d2201,
                satrec.d2211,
                satrec.d3210,
                satrec.d3222,
                satrec.d4410,
                satrec.d4422,
                satrec.d5220,
                satrec.d5232,
                satrec.d5421,
                satrec.d5433,
                satrec.dedt,
                satrec.didt,
                satrec.dmdt,
                dndt,
                satrec.dnodt,
                satrec.domdt,
                satrec.del1,
                satrec.del2,
                satrec.del3,
                satrec.xfact,
                satrec.xlamo,
                satrec.xli,
                satrec.xni,
            ) = _dsinit(
                satrec.xke,
                cosim,
                emsq,
                satrec.argpo,
                s1,
                s2,
                s3,
                s4,
                s5,
                sinim,
                ss1,
                ss2,
                ss3,
                ss4,
                ss5,
                sz1,
                sz3,
                sz11,
                sz13,
                sz21,
                sz23,
                sz31,
                sz33,
                satrec.t,
                tc,
                satrec.gsto,
                satrec.mo,
                satrec.mdot,
                satrec.no_unkozai,
                satrec.nodeo,
                satrec.nodedot,
                xpidot,
                z1,
                z3,
                z11,
                z13,
                z21,
                z23,
                z31,
                z33,
                satrec.ecco,
                eccsq,
                em,
                argpm,
                inclm,
                mm,
                nm,
                nodem,
                satrec.irez,
                satrec.atime,
                satrec.d2201,
                satrec.d2211,
                satrec.d3210,
                satrec.d3222,
                satrec.d4410,
                satrec.d4422,
                satrec.d5220,
                satrec.d5232,
                satrec.d5421,
                satrec.d5433,
                satrec.dedt,
                satrec.didt,
                satrec.dmdt,
                satrec.dnodt,
                satrec.domdt,
                satrec.del1,
                satrec.del2,
                satrec.del3,
                satrec.xfact,
                satrec.xlamo,
                satrec.xli,
                satrec.xni,
            )
        if satrec.isimp != 1:
            cc1sq = satrec.cc1 * satrec.cc1
            satrec.d2 = 4.0 * ao * tsi * cc1sq
            temp = satrec.d2 * tsi * satrec.cc1 / 3.0
            satrec.d3 = (17.0 * ao + sfour) * temp
            satrec.d4 = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * satrec.cc1
            satrec.t3cof = satrec.d2 + 2.0 * cc1sq
            satrec.t4cof = 0.25 * (
                3.0 * satrec.d3 + satrec.cc1 * (12.0 * satrec.d2 + 10.0 * cc1sq)
            )
            satrec.t5cof = 0.2 * (
                3.0 * satrec.d4
                + 12.0 * satrec.cc1 * satrec.d3
                + 6.0 * satrec.d2 * satrec.d2
                + 15.0 * cc1sq * (2.0 * satrec.d2 + cc1sq)
            )
    satrec.init = "n"
    return True


def gstime(jdut1):
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = (
        -6.2e-06 * tut1 * tut1 * tut1
        + 0.093104 * tut1 * tut1
        + (876600.0 * 3600 + 8640184.812866) * tut1
        + 67310.54841
    )
    temp = temp * deg2rad / 240.0 % twopi
    if temp < 0.0:
        temp += twopi
    return temp


_gstime = gstime


class SGP4Coefficients:
    """Prepared scalar coefficients for one TLE and one gravity constant set."""

    def __init__(self):
        object.__setattr__(self, "_frozen", False)

    def __setattr__(self, name, value):
        if self.__dict__["_frozen"]:
            raise AttributeError("Prepared SGP4 coefficients are immutable.")
        object.__setattr__(self, name, value)

    def freeze(self):
        object.__setattr__(self, "_frozen", True)


def initialize_sgp4(
    *,
    satellite_number: int,
    epoch_julian_day: float,
    mean_motion_derivative: float,
    mean_motion_second_derivative: float,
    bstar: float,
    eccentricity: float,
    argument_of_perigee: float,
    inclination: float,
    mean_anomaly: float,
    mean_motion_revolutions_per_day: float,
    raan: float,
    mu: float,
    equatorial_radius: float,
    j2: float,
    j3: float,
    j4: float,
) -> SGP4Coefficients:
    """Prepare the Vallado coefficients without entering transformed execution."""
    xke = 60.0 / sqrt(equatorial_radius**3 / mu)
    constants = (
        1.0 / xke,
        mu,
        equatorial_radius,
        xke,
        j2,
        j3,
        j4,
        j3 / j2,
    )
    xpdotp = 1440.0 / (2.0 * pi)
    satellite = SGP4Coefficients()
    sgp4init(
        constants,
        "i",
        str(satellite_number),
        epoch_julian_day - 2433281.5,
        bstar,
        mean_motion_derivative / (xpdotp * 1440.0),
        mean_motion_second_derivative / (xpdotp * 1440.0 * 1440.0),
        eccentricity,
        argument_of_perigee,
        inclination,
        mean_anomaly,
        mean_motion_revolutions_per_day / xpdotp,
        raan,
        satellite,
    )
    satellite.freeze()
    return satellite


def _dpper_jax(
    satrec: SGP4Coefficients,
    t: jax.Array,
    ep: jax.Array,
    inclp: jax.Array,
    nodep: jax.Array,
    argpp: jax.Array,
    mp: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Apply the standard lunar-solar long-period terms."""
    zns = 1.19459e-5
    zes = 0.01675
    znl = 1.5835218e-4
    zel = 0.05490

    zm = satrec.zmos + zns * t
    zf = zm + 2.0 * zes * jnp.sin(zm)
    sinzf = jnp.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * jnp.cos(zf)
    ses = satrec.se2 * f2 + satrec.se3 * f3
    sis = satrec.si2 * f2 + satrec.si3 * f3
    sls = satrec.sl2 * f2 + satrec.sl3 * f3 + satrec.sl4 * sinzf
    sghs = satrec.sgh2 * f2 + satrec.sgh3 * f3 + satrec.sgh4 * sinzf
    shs = satrec.sh2 * f2 + satrec.sh3 * f3

    zm = satrec.zmol + znl * t
    zf = zm + 2.0 * zel * jnp.sin(zm)
    sinzf = jnp.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * jnp.cos(zf)
    sel = satrec.ee2 * f2 + satrec.e3 * f3
    sil = satrec.xi2 * f2 + satrec.xi3 * f3
    sll = satrec.xl2 * f2 + satrec.xl3 * f3 + satrec.xl4 * sinzf
    sghl = satrec.xgh2 * f2 + satrec.xgh3 * f3 + satrec.xgh4 * sinzf
    shll = satrec.xh2 * f2 + satrec.xh3 * f3

    pe = ses + sel - satrec.peo
    pinc = sis + sil - satrec.pinco
    pl = sls + sll - satrec.plo
    pgh = sghs + sghl - satrec.pgho
    ph = shs + shll - satrec.pho
    inclp = inclp + pinc
    ep = ep + pe
    sinip = jnp.sin(inclp)
    cosip = jnp.cos(inclp)

    safe_sinip = jnp.where(jnp.abs(sinip) > 1.5e-12, sinip, 1.0)
    direct_ph = ph / safe_sinip
    direct_argpp = argpp + pgh - cosip * direct_ph
    direct_nodep = nodep + direct_ph
    direct_mp = mp + pl

    sinop = jnp.sin(nodep)
    cosop = jnp.cos(nodep)
    alfdp = sinip * sinop + ph * cosop + pinc * cosip * sinop
    betdp = sinip * cosop - ph * sinop + pinc * cosip * cosop
    old_nodep = jnp.fmod(nodep, 2.0 * jnp.pi)
    lyddane_xls = mp + argpp + pl + pgh + (cosip - pinc * sinip) * old_nodep
    lyddane_nodep = jnp.arctan2(alfdp, betdp)
    wrap = jnp.abs(old_nodep - lyddane_nodep) > jnp.pi
    lyddane_nodep = jnp.where(
        wrap,
        jnp.where(
            lyddane_nodep < old_nodep,
            lyddane_nodep + 2.0 * jnp.pi,
            lyddane_nodep - 2.0 * jnp.pi,
        ),
        lyddane_nodep,
    )
    lyddane_mp = mp + pl
    lyddane_argpp = lyddane_xls - lyddane_mp - cosip * lyddane_nodep
    direct = inclp >= 0.2
    return (
        ep,
        inclp,
        jnp.where(direct, direct_nodep, lyddane_nodep),
        jnp.where(direct, direct_argpp, lyddane_argpp),
        jnp.where(direct, direct_mp, lyddane_mp),
    )


def _resonance_derivatives(
    satrec: SGP4Coefficients,
    atime: jax.Array,
    xli: jax.Array,
    xni: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate the synchronous or half-day resonance derivatives."""
    xldot = xni + satrec.xfact
    if satrec.irez == 1:
        xndt = (
            satrec.del1 * jnp.sin(xli - 0.13130908)
            + satrec.del2 * jnp.sin(2.0 * (xli - 2.8843198))
            + satrec.del3 * jnp.sin(3.0 * (xli - 0.37448087))
        )
        xnddt = (
            satrec.del1 * jnp.cos(xli - 0.13130908)
            + 2.0 * satrec.del2 * jnp.cos(2.0 * (xli - 2.8843198))
            + 3.0 * satrec.del3 * jnp.cos(3.0 * (xli - 0.37448087))
        ) * xldot
    else:
        xomi = satrec.argpo + satrec.argpdot * atime
        x2omi = 2.0 * xomi
        x2li = 2.0 * xli
        xndt = (
            satrec.d2201 * jnp.sin(x2omi + xli - 5.7686396)
            + satrec.d2211 * jnp.sin(xli - 5.7686396)
            + satrec.d3210 * jnp.sin(xomi + xli - 0.95240898)
            + satrec.d3222 * jnp.sin(-xomi + xli - 0.95240898)
            + satrec.d4410 * jnp.sin(x2omi + x2li - 1.8014998)
            + satrec.d4422 * jnp.sin(x2li - 1.8014998)
            + satrec.d5220 * jnp.sin(xomi + xli - 1.050833)
            + satrec.d5232 * jnp.sin(-xomi + xli - 1.050833)
            + satrec.d5421 * jnp.sin(xomi + x2li - 4.4108898)
            + satrec.d5433 * jnp.sin(-xomi + x2li - 4.4108898)
        )
        xnddt = (
            satrec.d2201 * jnp.cos(x2omi + xli - 5.7686396)
            + satrec.d2211 * jnp.cos(xli - 5.7686396)
            + satrec.d3210 * jnp.cos(xomi + xli - 0.95240898)
            + satrec.d3222 * jnp.cos(-xomi + xli - 0.95240898)
            + satrec.d5220 * jnp.cos(xomi + xli - 1.050833)
            + satrec.d5232 * jnp.cos(-xomi + xli - 1.050833)
            + 2.0
            * (
                satrec.d4410 * jnp.cos(x2omi + x2li - 1.8014998)
                + satrec.d4422 * jnp.cos(x2li - 1.8014998)
                + satrec.d5421 * jnp.cos(xomi + x2li - 4.4108898)
                + satrec.d5433 * jnp.cos(-xomi + x2li - 4.4108898)
            )
        ) * xldot
    return xndt, xldot, xnddt


def _dspace_jax(
    satrec: SGP4Coefficients,
    t: jax.Array,
    em: jax.Array,
    argpm: jax.Array,
    inclm: jax.Array,
    mm: jax.Array,
    nodem: jax.Array,
    nm: jax.Array,
    *,
    step_minutes: float,
    step_capacity: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Apply deep-space secular terms and bounded resonance integration."""
    theta = jnp.mod(satrec.gsto + t * 0.0043752690880113, 2.0 * jnp.pi)
    em = em + satrec.dedt * t
    inclm = inclm + satrec.didt * t
    argpm = argpm + satrec.domdt * t
    nodem = nodem + satrec.dnodt * t
    mm = mm + satrec.dmdt * t
    if satrec.irez == 0:
        return em, argpm, inclm, mm, nodem, nm, jnp.asarray(0, jnp.int32)

    direction = jnp.where(t >= 0.0, 1.0, -1.0)
    full_steps = jnp.floor(jnp.abs(t) / step_minutes + 1.0e-12).astype(jnp.int32)
    bounded_steps = jnp.minimum(full_steps, step_capacity)
    delta = direction * step_minutes
    step2 = 0.5 * step_minutes * step_minutes

    def integrate(carry, index):
        atime, xli, xni = carry
        active = index < bounded_steps
        xndt, xldot, xnddt = _resonance_derivatives(satrec, atime, xli, xni)
        candidate_xli = xli + xldot * delta + xndt * step2
        candidate_xni = xni + xndt * delta + xnddt * step2
        return (
            jnp.where(active, atime + delta, atime),
            jnp.where(active, candidate_xli, xli),
            jnp.where(active, candidate_xni, xni),
        ), active

    (atime, xli, xni), _ = jax.lax.scan(
        integrate,
        (
            jnp.asarray(0.0, dtype=t.dtype),
            jnp.asarray(satrec.xlamo, dtype=t.dtype),
            jnp.asarray(satrec.no_unkozai, dtype=t.dtype),
        ),
        jnp.arange(step_capacity),
    )
    xndt, xldot, xnddt = _resonance_derivatives(satrec, atime, xli, xni)
    ft = t - atime
    nm = xni + xndt * ft + 0.5 * xnddt * ft * ft
    xl = xli + xldot * ft + 0.5 * xndt * ft * ft
    if satrec.irez == 1:
        mm = xl - nodem - argpm + theta
    else:
        mm = xl - 2.0 * nodem + 2.0 * theta
    segments = jnp.ceil(jnp.abs(t) / step_minutes).astype(jnp.int32)
    return em, argpm, inclm, mm, nodem, nm, jnp.minimum(segments, step_capacity)


def propagate_sgp4(
    satrec: SGP4Coefficients,
    minutes_since_epoch: jax.Array,
    *,
    resonance_step_minutes: float,
    resonance_capacity: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Propagate one prepared TLE with the Vallado SGP4/SDP4 equations."""
    t = minutes_since_epoch
    xmdf = satrec.mo + satrec.mdot * t
    argpdf = satrec.argpo + satrec.argpdot * t
    nodedf = satrec.nodeo + satrec.nodedot * t
    argpm = argpdf
    mm = xmdf
    t2 = t * t
    nodem = nodedf + satrec.nodecf * t2
    tempa = 1.0 - satrec.cc1 * t
    tempe = satrec.bstar * satrec.cc4 * t
    templ = satrec.t2cof * t2

    if satrec.isimp != 1:
        delomg = satrec.omgcof * t
        delmtemp = 1.0 + satrec.eta * jnp.cos(xmdf)
        delm = satrec.xmcof * (delmtemp**3 - satrec.delmo)
        drag_angle = delomg + delm
        mm = xmdf + drag_angle
        argpm = argpdf - drag_angle
        t3 = t2 * t
        t4 = t3 * t
        tempa = tempa - satrec.d2 * t2 - satrec.d3 * t3 - satrec.d4 * t4
        tempe = tempe + satrec.bstar * satrec.cc5 * (jnp.sin(mm) - satrec.sinmao)
        templ = templ + satrec.t3cof * t3 + t4 * (satrec.t4cof + t * satrec.t5cof)

    nm = jnp.asarray(satrec.no_unkozai, dtype=t.dtype)
    em = jnp.asarray(satrec.ecco, dtype=t.dtype)
    inclm = jnp.asarray(satrec.inclo, dtype=t.dtype)
    resonance_steps = jnp.asarray(0, dtype=jnp.int32)
    if satrec.method == "d":
        em, argpm, inclm, mm, nodem, nm, resonance_steps = _dspace_jax(
            satrec,
            t,
            em,
            argpm,
            inclm,
            mm,
            nodem,
            nm,
            step_minutes=resonance_step_minutes,
            step_capacity=resonance_capacity,
        )

    mean_motion_valid = jnp.isfinite(nm) & (nm > 0.0)
    safe_nm = jnp.where(mean_motion_valid, nm, satrec.no_unkozai)
    am = (satrec.xke / safe_nm) ** (2.0 / 3.0) * tempa * tempa
    nm = satrec.xke / jnp.maximum(am, 1.0e-12) ** 1.5
    em = em - tempe
    mean_eccentricity_valid = jnp.isfinite(em) & (em >= -0.001) & (em < 1.0)
    em = jnp.clip(em, 1.0e-6, 1.0 - 1.0e-12)
    mm = mm + satrec.no_unkozai * templ
    xlm = mm + argpm + nodem
    nodem = jnp.fmod(nodem, 2.0 * jnp.pi)
    argpm = jnp.mod(argpm, 2.0 * jnp.pi)
    xlm = jnp.mod(xlm, 2.0 * jnp.pi)
    mm = jnp.mod(xlm - argpm - nodem, 2.0 * jnp.pi)

    ep = em
    xincp = inclm
    argpp = argpm
    nodep = nodem
    mp = mm
    if satrec.method == "d":
        ep, xincp, nodep, argpp, mp = _dpper_jax(satrec, t, ep, xincp, nodep, argpp, mp)
        negative_inclination = xincp < 0.0
        xincp = jnp.where(negative_inclination, -xincp, xincp)
        nodep = jnp.where(negative_inclination, nodep + jnp.pi, nodep)
        argpp = jnp.where(negative_inclination, argpp - jnp.pi, argpp)
    periodic_eccentricity_valid = jnp.isfinite(ep) & (ep >= 0.0) & (ep <= 1.0)
    ep = jnp.clip(ep, 1.0e-6, 1.0 - 1.0e-12)

    sinip = jnp.sin(xincp)
    cosip = jnp.cos(xincp)
    if satrec.method == "d":
        aycof = -0.5 * satrec.j3oj2 * sinip
        denominator = jnp.where(jnp.abs(cosip + 1.0) > 1.5e-12, 1.0 + cosip, 1.5e-12)
        xlcof = -0.25 * satrec.j3oj2 * sinip * (3.0 + 5.0 * cosip) / denominator
        con41 = 3.0 * cosip * cosip - 1.0
        x1mth2 = 1.0 - cosip * cosip
        x7thm1 = 7.0 * cosip * cosip - 1.0
    else:
        aycof = satrec.aycof
        xlcof = satrec.xlcof
        con41 = satrec.con41
        x1mth2 = satrec.x1mth2
        x7thm1 = satrec.x7thm1

    axnl = ep * jnp.cos(argpp)
    long_period_scale = 1.0 / (jnp.maximum(am, 1.0e-12) * (1.0 - ep * ep))
    aynl = ep * jnp.sin(argpp) + long_period_scale * aycof
    xl = mp + argpp + nodep + long_period_scale * xlcof * axnl
    u = jnp.mod(xl - nodep, 2.0 * jnp.pi)

    def solve_kepler(_, eccentric_longitude):
        sine = jnp.sin(eccentric_longitude)
        cosine = jnp.cos(eccentric_longitude)
        denominator = 1.0 - cosine * axnl - sine * aynl
        correction = (u - aynl * cosine + axnl * sine - eccentric_longitude) / denominator
        correction = jnp.clip(correction, -0.95, 0.95)
        return eccentric_longitude + correction

    eccentric_longitude = jax.lax.fori_loop(0, 10, solve_kepler, u)
    sine = jnp.sin(eccentric_longitude)
    cosine = jnp.cos(eccentric_longitude)
    kepler_residual = jnp.abs(u - aynl * cosine + axnl * sine - eccentric_longitude)
    ecose = axnl * cosine + aynl * sine
    esine = axnl * sine - aynl * cosine
    el2 = axnl * axnl + aynl * aynl
    pl = am * (1.0 - el2)
    semilatus_valid = jnp.isfinite(pl) & (pl > 0.0)
    safe_pl = jnp.maximum(pl, 1.0e-12)
    rl = am * (1.0 - ecose)
    safe_rl = jnp.where(jnp.abs(rl) > 1.0e-12, rl, 1.0e-12)
    rdotl = jnp.sqrt(jnp.maximum(am, 0.0)) * esine / safe_rl
    rvdotl = jnp.sqrt(safe_pl) / safe_rl
    betal = jnp.sqrt(jnp.maximum(1.0 - el2, 0.0))
    temp = esine / (1.0 + betal)
    sinu = am / safe_rl * (sine - aynl - axnl * temp)
    cosu = am / safe_rl * (cosine - axnl + aynl * temp)
    su = jnp.arctan2(sinu, cosu)
    sin2u = 2.0 * cosu * sinu
    cos2u = 1.0 - 2.0 * sinu * sinu
    inverse_pl = 1.0 / safe_pl
    temp1 = 0.5 * satrec.j2 * inverse_pl
    temp2 = temp1 * inverse_pl

    mrt = rl * (1.0 - 1.5 * temp2 * betal * con41) + 0.5 * temp1 * x1mth2 * cos2u
    su = su - 0.25 * temp2 * x7thm1 * sin2u
    xnode = nodep + 1.5 * temp2 * cosip * sin2u
    xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u
    mvt = rdotl - nm * temp1 * x1mth2 * sin2u / satrec.xke
    rvdot = rvdotl + nm * temp1 * (x1mth2 * cos2u + 1.5 * con41) / satrec.xke

    sinsu = jnp.sin(su)
    cossu = jnp.cos(su)
    snod = jnp.sin(xnode)
    cnod = jnp.cos(xnode)
    sini = jnp.sin(xinc)
    cosi = jnp.cos(xinc)
    xmx = -snod * cosi
    xmy = cnod * cosi
    ux = xmx * sinsu + cnod * cossu
    uy = xmy * sinsu + snod * cossu
    uz = sini * sinsu
    vx = xmx * cossu - cnod * sinsu
    vy = xmy * cossu - snod * sinsu
    vz = sini * cossu

    radius = mrt * satrec.radiusearthkm
    speed_scale = satrec.radiusearthkm * satrec.xke / 60.0
    position = radius * jnp.asarray((ux, uy, uz))
    velocity = speed_scale * jnp.asarray(
        (
            mvt * ux + rvdot * vx,
            mvt * uy + rvdot * vy,
            mvt * uz + rvdot * vz,
        )
    )
    finite = (
        jnp.all(jnp.isfinite(position))
        & jnp.all(jnp.isfinite(velocity))
        & jnp.isfinite(kepler_residual)
        & jnp.isfinite(mrt)
    )
    position = jnp.where(finite, position, jnp.zeros_like(position))
    velocity = jnp.where(finite, velocity, jnp.zeros_like(velocity))
    decayed = mrt < 1.0
    tolerance = jnp.maximum(
        jnp.asarray(1.0e-10, dtype=t.dtype),
        64.0 * jnp.finfo(t.dtype).eps,
    )
    valid = (
        mean_motion_valid
        & mean_eccentricity_valid
        & periodic_eccentricity_valid
        & semilatus_valid
        & ~decayed
        & finite
        & (kepler_residual <= tolerance)
    )
    error = jnp.where(
        ~mean_motion_valid,
        2,
        jnp.where(
            ~mean_eccentricity_valid,
            1,
            jnp.where(
                ~periodic_eccentricity_valid,
                3,
                jnp.where(~semilatus_valid, 4, jnp.where(decayed, 6, 0)),
            ),
        ),
    ).astype(jnp.int32)
    radius_check = (mrt - 1.0) * satrec.radiusearthkm
    return (
        position,
        velocity,
        valid,
        error,
        kepler_residual,
        radius_check,
        resonance_steps,
    )


__all__ = ["SGP4Coefficients", "initialize_sgp4", "propagate_sgp4"]
