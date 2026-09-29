/* Copyright © 2026 PHYDRA, Inc. All rights reserved.
 * Original FMI2 Co-Simulation specimen: one lumped thermal zone
 *   C dT/dt = G (T_b - T) + P
 * with boundary temperature T_b and heater power P held over each communication
 * step (FMI2 zero-order hold). The step is integrated exactly:
 *   G > 0: T_eq = T_b + P/G, T(h) = T_eq + (T - T_eq) exp(-G h / C);
 *   G = 0: T(h) = T + P h / C.
 * conducted_heat accumulates the heat conducted in from the boundary,
 * C (T(h) - T) - P h, so its increment is the exchanged energy of one step.
 * A zone that would exceed maximum_temperature stops exactly at the crossing
 * and returns fmi2Discard with that crossing as fmi2LastSuccessfulTime.
 * No foreign code. Compile as a host shared library and package with the
 * adjacent XML as an FMU.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct {
    double time, temperature, boundary, power, capacity, conductance, initial;
    double maximum, conducted, stop_time;
    int steps, mode, terminated, has_stop;
} Zone;

enum { OK=0, WARNING=1, DISCARD=2, ERROR=3 };

API const char *fmi2GetTypesPlatform(void) { return "default"; }
API const char *fmi2GetVersion(void) { return "2.0"; }

static void reset_zone(Zone *z) {
    memset(z, 0, sizeof(*z));
    z->boundary = 293.15;
    z->capacity = 1;
    z->initial = 293.15;
    z->temperature = 293.15;
    z->maximum = 1e9;
}

API void *fmi2Instantiate(const char *name, int kind, const char *guid,
        const char *resources, const void *callbacks, int visible, int logging) {
    (void)resources; (void)callbacks; (void)visible; (void)logging;
    if (!name || kind != 1 || !guid || strcmp(guid, "{phydrax-thermal-zone}")) return NULL;
    Zone *z = malloc(sizeof(*z));
    if (z) reset_zone(z);
    return z;
}
API void fmi2FreeInstance(void *c) { free(c); }
API int fmi2SetDebugLogging(void *c, int enabled, size_t n, const char **categories) {
    (void)enabled; (void)categories;
    /* This specimen has no log categories. */
    return c && n == 0 ? OK : ERROR;
}
API int fmi2SetupExperiment(void *c, int tolerance_defined, double tolerance,
        double start, int stop_defined, double stop) {
    Zone *z = c;
    if (!z || z->mode || !isfinite(start) ||
        (tolerance_defined && (!(tolerance > 0) || !isfinite(tolerance))) ||
        (stop_defined && (!(stop > start) || !isfinite(stop)))) return ERROR;
    z->time = start; z->has_stop = stop_defined; z->stop_time = stop;
    return OK;
}
API int fmi2EnterInitializationMode(void *c) {
    Zone *z = c; if (!z || z->mode != 0) return ERROR;
    z->mode = 1; return OK;
}
API int fmi2ExitInitializationMode(void *c) {
    Zone *z = c; if (!z || z->mode != 1) return ERROR;
    if (!(z->capacity > 0) || !(z->conductance >= 0) || !isfinite(z->initial) ||
        !isfinite(z->capacity) || !isfinite(z->conductance) ||
        !(z->initial <= z->maximum)) return ERROR;
    z->temperature = z->initial;
    z->mode = 2; return OK;
}
API int fmi2Terminate(void *c) {
    Zone *z = c; if (!z || z->mode != 2) return ERROR;
    z->mode = 3; return OK;
}
API int fmi2Reset(void *c) { if (!c) return ERROR; reset_zone(c); return OK; }

API int fmi2GetReal(void *c, const unsigned *vr, size_t n, double *values) {
    Zone *z = c; if (!z) return ERROR;
    for (size_t i=0; i<n; ++i) {
        switch (vr[i]) {
            case 0: values[i]=z->boundary; break;
            case 1: values[i]=z->power; break;
            case 2: values[i]=z->temperature; break;
            case 3: values[i]=z->conducted; break;
            case 4: values[i]=z->capacity; break;
            case 5: values[i]=z->conductance; break;
            case 6: values[i]=z->initial; break;
            case 7: values[i]=z->temperature - 273.15; break;
            case 9: values[i]=z->maximum; break;
            default: return ERROR;
        }
    }
    return OK;
}
API int fmi2SetReal(void *c, const unsigned *vr, size_t n, const double *values) {
    Zone *z = c; if (!z || z->mode > 2 || z->terminated) return ERROR;
    for (size_t i=0; i<n; ++i) {
        if (!isfinite(values[i])) return ERROR;
        if (vr[i]==0) z->boundary=values[i];
        else if (vr[i]==1) z->power=values[i];
        else if (vr[i]==4 && z->mode<2) z->capacity=values[i];
        else if (vr[i]==5 && z->mode<2) z->conductance=values[i];
        else if (vr[i]==6 && z->mode<2) z->initial=values[i];
        else if (vr[i]==9 && z->mode<2) z->maximum=values[i];
        else return ERROR;
    }
    return OK;
}
API int fmi2GetInteger(void *c, const unsigned *vr, size_t n, int *values) {
    Zone *z = c; if (!z) return ERROR;
    for (size_t i=0; i<n; ++i) { if (vr[i]!=10) return ERROR; values[i]=z->steps; }
    return OK;
}
API int fmi2SetInteger(void *c, const unsigned *vr, size_t n, const int *values) {
    /* The only Integer variable is an output. */
    (void)c; (void)vr; (void)values; return n == 0 ? OK : ERROR;
}
API int fmi2GetBoolean(void *c, const unsigned *vr, size_t n, int *values) {
    /* This specimen declares no Boolean variables. */
    (void)c; (void)vr; (void)values; return n == 0 ? OK : ERROR;
}
API int fmi2SetBoolean(void *c, const unsigned *vr, size_t n, const int *values) {
    /* This specimen declares no Boolean variables. */
    (void)c; (void)vr; (void)values; return n == 0 ? OK : ERROR;
}
API int fmi2GetString(void *c, const unsigned *vr, size_t n, const char **values) {
    /* This specimen declares no String variables. */
    (void)c; (void)vr; (void)values; return n == 0 ? OK : ERROR;
}
API int fmi2SetString(void *c, const unsigned *vr, size_t n, const char * const *values) {
    /* This specimen declares no String variables. */
    (void)c; (void)vr; (void)values; return n == 0 ? OK : ERROR;
}
API int fmi2DoStep(void *c, double current, double step, int no_prior_state) {
    (void)no_prior_state;
    Zone *z = c;
    if (!z || z->mode!=2 || z->terminated || !(step>0) || !isfinite(step) ||
        fabs(current-z->time)>1e-12*fmax(1.0, fabs(current)) ||
        (z->has_stop && current+step>z->stop_time)) return ERROR;
    double before = z->temperature, after, equilibrium = 0;
    if (z->conductance > 0) {
        equilibrium = z->boundary + z->power/z->conductance;
        after = equilibrium + (before - equilibrium)*exp(-z->conductance*step/z->capacity);
    } else {
        after = before + z->power*step/z->capacity;
    }
    int crossed = after > z->maximum;
    if (crossed) {
        /* The exact solution is monotone over one held step; stop at T = maximum. */
        step = z->conductance > 0
            ? -z->capacity/z->conductance*log((z->maximum - equilibrium)/(before - equilibrium))
            : (z->maximum - before)*z->capacity/z->power;
        after = z->maximum;
    }
    z->conducted += z->capacity*(after - before) - z->power*step;
    z->temperature = after;
    z->time = current+step; z->steps++;
    return crossed ? DISCARD : OK;
}
API int fmi2CancelStep(void *c) {
    /* Synchronous FMUs cannot have pending asynchronous work to cancel. */
    (void)c; return ERROR;
}
API int fmi2GetStatus(void *c, int kind, int *value) {
    if (!c || kind!=0) return ERROR; *value=OK; return OK;
}
API int fmi2GetRealStatus(void *c, int kind, double *value) {
    if (!c || kind!=2) return ERROR; *value=((Zone*)c)->time; return OK;
}
API int fmi2GetBooleanStatus(void *c, int kind, int *value) {
    if (!c || kind!=3) return ERROR; *value=((Zone*)c)->terminated; return OK;
}
API int fmi2GetIntegerStatus(void *c, int kind, int *value) {
    /* FMI2 defines no integer-valued Co-Simulation status kinds. */
    (void)c; (void)kind; (void)value; return ERROR;
}
API int fmi2GetStringStatus(void *c, int kind, const char **value) {
    /* Pending-status strings are undefined for this synchronous FMU. */
    (void)c; (void)kind; (void)value; return ERROR;
}
API int fmi2GetFMUstate(void *c, void **state) {
    if (!c || !state) return ERROR;
    if (!*state) *state=malloc(sizeof(Zone));
    if (!*state) return ERROR;
    memcpy(*state,c,sizeof(Zone)); return OK;
}
API int fmi2SetFMUstate(void *c, void *state) {
    if (!c || !state) return ERROR; memcpy(c,state,sizeof(Zone)); return OK;
}
API int fmi2FreeFMUstate(void *c, void **state) {
    if (!c || !state) return ERROR; free(*state); *state=NULL; return OK;
}
API int fmi2SerializedFMUstateSize(void *c, void *state, size_t *size) {
    if (!c || !state || !size) return ERROR; *size=sizeof(Zone); return OK;
}
API int fmi2SerializeFMUstate(void *c, void *state, char *bytes, size_t size) {
    if (!c || !state || !bytes || size!=sizeof(Zone)) return ERROR;
    memcpy(bytes,state,size); return OK;
}
API int fmi2DeSerializeFMUstate(void *c, const char *bytes, size_t size, void **state) {
    if (!c || !bytes || !state || size!=sizeof(Zone)) return ERROR;
    if (!*state) *state=malloc(size);
    if (!*state) return ERROR;
    memcpy(*state,bytes,size); return OK;
}
API int fmi2SetRealInputDerivatives(void *c, const unsigned *vr, size_t n,
        const int *orders, const double *values) {
    /* maxOutputDerivativeOrder is 0 and inputs are not interpolated. */
    (void)c; (void)vr; (void)orders; (void)values; return n == 0 ? OK : ERROR;
}
API int fmi2GetRealOutputDerivatives(void *c, const unsigned *vr, size_t n,
        const int *orders, double *values) {
    /* maxOutputDerivativeOrder is 0. */
    (void)c; (void)vr; (void)orders; (void)values; return n == 0 ? OK : ERROR;
}
API int fmi2GetDirectionalDerivative(void *c, const unsigned *unknown, size_t nu,
        const unsigned *known, size_t nk, const double *seed, double *result) {
    /* providesDirectionalDerivative is false. */
    (void)c; (void)unknown; (void)known; (void)seed; (void)result;
    return nu == 0 && nk == 0 ? OK : ERROR;
}
