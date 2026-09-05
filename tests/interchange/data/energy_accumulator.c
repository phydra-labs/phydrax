/* Copyright © 2026 PHYDRA, Inc. All rights reserved.
 * Original FMI2 Co-Simulation specimen: dx/dt = gain*u, with a time event
 * x -> -x at event_time. Exact zero-order-hold integration; no foreign code.
 * Compile as a host shared library and package with the adjacent XML as an FMU.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct {
    double time, x, u, gain, event_time, input_slope, stop_time;
    int steps, stop_at_event, event_done, terminated, mode, has_stop;
    char label[128];
} Model;

enum { OK=0, WARNING=1, DISCARD=2, ERROR=3 };

API const char *fmi2GetTypesPlatform(void) { return "default"; }
API const char *fmi2GetVersion(void) { return "2.0"; }

static void reset_model(Model *m) {
    memset(m, 0, sizeof(*m));
    m->gain = 1;
    m->event_time = 0.5;
    strcpy(m->label, "energy-accumulator");
}

API void *fmi2Instantiate(const char *name, int kind, const char *guid,
        const char *resources, const void *callbacks, int visible, int logging) {
    (void)resources; (void)callbacks; (void)visible; (void)logging;
    if (!name || kind != 1 || !guid || strcmp(guid, "{phydrax-energy-accumulator}")) return NULL;
    Model *m = malloc(sizeof(*m));
    if (m) reset_model(m);
    return m;
}
API void fmi2FreeInstance(void *c) { free(c); }
API int fmi2SetDebugLogging(void *c, int enabled, size_t n, const char **categories) {
    (void)enabled; (void)categories;
    /* This specimen has no log categories. */
    return c && n == 0 ? OK : ERROR;
}
API int fmi2SetupExperiment(void *c, int tolerance_defined, double tolerance,
        double start, int stop_defined, double stop) {
    Model *m = c;
    if (!m || m->mode || !isfinite(start) ||
        (tolerance_defined && (!(tolerance > 0) || !isfinite(tolerance))) ||
        (stop_defined && (!(stop > start) || !isfinite(stop)))) return ERROR;
    m->time = start; m->has_stop = stop_defined; m->stop_time = stop;
    return OK;
}
API int fmi2EnterInitializationMode(void *c) {
    Model *m = c; if (!m || m->mode != 0) return ERROR;
    m->mode = 1; return OK;
}
API int fmi2ExitInitializationMode(void *c) {
    Model *m = c; if (!m || m->mode != 1) return ERROR;
    m->event_done = m->event_time <= m->time;
    m->mode = 2; return OK;
}
API int fmi2Terminate(void *c) {
    Model *m = c; if (!m || m->mode != 2) return ERROR;
    m->mode = 3; return OK;
}
API int fmi2Reset(void *c) { if (!c) return ERROR; reset_model(c); return OK; }

API int fmi2GetReal(void *c, const unsigned *vr, size_t n, double *values) {
    Model *m = c; if (!m) return ERROR;
    for (size_t i=0; i<n; ++i) {
        switch (vr[i]) {
            case 0: values[i]=m->u; break;
            case 1: values[i]=m->x; break;
            case 2: values[i]=m->gain; break;
            case 3: values[i]=m->event_time; break;
            case 4: values[i]=m->gain*m->u; break;
            default: return ERROR;
        }
    }
    return OK;
}
API int fmi2SetReal(void *c, const unsigned *vr, size_t n, const double *values) {
    Model *m = c; if (!m || m->mode > 2 || m->terminated) return ERROR;
    for (size_t i=0; i<n; ++i) {
        if (!isfinite(values[i])) return ERROR;
        if (vr[i]==0) m->u=values[i];
        else if (vr[i]==1 && m->mode<2) m->x=values[i];
        else if (vr[i]==2 && m->mode<2) m->gain=values[i];
        else if (vr[i]==3 && m->mode<2) m->event_time=values[i];
        else return ERROR;
    }
    return OK;
}
API int fmi2GetInteger(void *c, const unsigned *vr, size_t n, int *values) {
    Model *m = c; if (!m) return ERROR;
    for (size_t i=0; i<n; ++i) { if (vr[i]!=10) return ERROR; values[i]=m->steps; }
    return OK;
}
API int fmi2SetInteger(void *c, const unsigned *vr, size_t n, const int *values) {
    Model *m = c; if (!m || m->mode>=2) return ERROR;
    for (size_t i=0; i<n; ++i) { if (vr[i]!=10 || values[i]<0) return ERROR; m->steps=values[i]; }
    return OK;
}
API int fmi2GetBoolean(void *c, const unsigned *vr, size_t n, int *values) {
    Model *m = c; if (!m) return ERROR;
    for (size_t i=0; i<n; ++i) {
        if (vr[i]==20) values[i]=m->stop_at_event;
        else if (vr[i]==21) values[i]=m->event_done;
        else return ERROR;
    }
    return OK;
}
API int fmi2SetBoolean(void *c, const unsigned *vr, size_t n, const int *values) {
    Model *m = c; if (!m || m->mode>=2) return ERROR;
    for (size_t i=0; i<n; ++i) {
        if (vr[i]!=20 || (values[i]!=0 && values[i]!=1)) return ERROR;
        m->stop_at_event=values[i];
    }
    return OK;
}
API int fmi2GetString(void *c, const unsigned *vr, size_t n, const char **values) {
    Model *m = c; if (!m) return ERROR;
    for (size_t i=0; i<n; ++i) { if (vr[i]!=30) return ERROR; values[i]=m->label; }
    return OK;
}
API int fmi2SetString(void *c, const unsigned *vr, size_t n, const char * const *values) {
    Model *m = c; if (!m || m->mode>=2) return ERROR;
    for (size_t i=0; i<n; ++i) {
        if (vr[i]!=30 || !values[i] || strlen(values[i])>=sizeof(m->label)) return ERROR;
        strcpy(m->label, values[i]);
    }
    return OK;
}
API int fmi2DoStep(void *c, double current, double step, int no_prior_state) {
    (void)no_prior_state;
    Model *m = c;
    if (!m || m->mode!=2 || m->terminated || !(step>0) || !isfinite(step) ||
        fabs(current-m->time)>1e-12 || (m->has_stop && current+step>m->stop_time)) return ERROR;
    double reached = current+step;
    int event = !m->event_done && m->event_time>current && m->event_time<=reached;
    if (event) reached=m->event_time;
    double h=reached-current;
    m->x += m->gain*(m->u*h + 0.5*m->input_slope*h*h);
    m->u += m->input_slope*h;
    m->time=reached; m->steps++;
    if (event) {
        m->x = -m->x; m->event_done=1; m->terminated=m->stop_at_event;
        return DISCARD;
    }
    return OK;
}
API int fmi2CancelStep(void *c) {
    /* Synchronous FMUs cannot have pending asynchronous work to cancel. */
    (void)c; return ERROR;
}
API int fmi2GetStatus(void *c, int kind, int *value) {
    Model *m=c; if (!m || kind!=0) return ERROR;
    *value=m->event_done && m->time==m->event_time ? DISCARD : OK; return OK;
}
API int fmi2GetRealStatus(void *c, int kind, double *value) {
    if (!c || kind!=2) return ERROR; *value=((Model*)c)->time; return OK;
}
API int fmi2GetBooleanStatus(void *c, int kind, int *value) {
    if (!c || kind!=3) return ERROR; *value=((Model*)c)->terminated; return OK;
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
    if (!*state) *state=malloc(sizeof(Model));
    if (!*state) return ERROR;
    memcpy(*state,c,sizeof(Model)); return OK;
}
API int fmi2SetFMUstate(void *c, void *state) {
    if (!c || !state) return ERROR; memcpy(c,state,sizeof(Model)); return OK;
}
API int fmi2FreeFMUstate(void *c, void **state) {
    if (!c || !state) return ERROR; free(*state); *state=NULL; return OK;
}
API int fmi2SerializedFMUstateSize(void *c, void *state, size_t *size) {
    if (!c || !state || !size) return ERROR; *size=sizeof(Model); return OK;
}
API int fmi2SerializeFMUstate(void *c, void *state, char *bytes, size_t size) {
    if (!c || !state || !bytes || size!=sizeof(Model)) return ERROR;
    memcpy(bytes,state,size); return OK;
}
API int fmi2DeSerializeFMUstate(void *c, const char *bytes, size_t size, void **state) {
    if (!c || !bytes || !state || size!=sizeof(Model)) return ERROR;
    if (!*state) *state=malloc(size);
    if (!*state) return ERROR;
    memcpy(*state,bytes,size); return OK;
}
API int fmi2SetRealInputDerivatives(void *c, const unsigned *vr, size_t n,
        const int *orders, const double *values) {
    Model *m=c; if (!m || m->mode!=2) return ERROR;
    for (size_t i=0;i<n;++i) {
        if (vr[i]!=0 || orders[i]!=1 || !isfinite(values[i])) return ERROR;
        m->input_slope=values[i];
    }
    return OK;
}
API int fmi2GetRealOutputDerivatives(void *c, const unsigned *vr, size_t n,
        const int *orders, double *values) {
    Model *m=c; if (!m || m->mode!=2) return ERROR;
    for (size_t i=0;i<n;++i) {
        if (vr[i]!=1 || orders[i]!=1) return ERROR;
        values[i]=m->gain*m->u;
    }
    return OK;
}
API int fmi2GetDirectionalDerivative(void *c, const unsigned *unknown, size_t nu,
        const unsigned *known, size_t nk, const double *seed, double *result) {
    Model *m=c; if (!m) return ERROR;
    for (size_t i=0;i<nu;++i) {
        if (unknown[i]!=4) return ERROR;
        result[i]=0;
        for (size_t j=0;j<nk;++j) {
            if (known[j]==0) result[i]+=m->gain*seed[j];
            else if (known[j]==2) result[i]+=m->u*seed[j];
            else return ERROR;
        }
    }
    return OK;
}
