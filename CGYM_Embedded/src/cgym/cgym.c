#include "cgym.h"


FILE *sopen(const char *program)
{
    int fds[2];
    pid_t pid;

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) < 0)
        return NULL;

    switch(pid=vfork()) {
    case -1:    /* Error */
        close(fds[0]);
        close(fds[1]);
        return NULL;
    case 0:     /* child */
        close(fds[0]);
        dup2(fds[1], 0);
        dup2(fds[1], 1);
        close(fds[1]);
        execl("/bin/sh", "sh", "-c", program, NULL);
        _exit(127);
    }
    /* parent */
    close(fds[1]);
    return fdopen(fds[0], "r+");
}

static void CGYM_EXEC(cgym_t *g, char exec[])
{
    int len = strlen(exec);
    fwrite(&len, sizeof(int), 1, g->python_gym);
    fwrite(exec, sizeof(char), len, g->python_gym);
    fflush(g->python_gym);
}

static char* CGYM_READ(cgym_t *g)
{
    int len = 0;
    fread(&len, sizeof(int), 1, g->python_gym);
    fread(g->buf, sizeof(char), len, g->python_gym);
    g->buf[len] = '\0';
    return g->buf;
}

static char* CGYM_READ_STATE(cgym_t *g)
{
    fread(g->state, sizeof(char), g->state_bytes, g->python_gym);
    return g->state;
}

static char* CGYM_READ_ACTION(cgym_t *g)
{
    fread(g->action, sizeof(char), g->action_bytes, g->python_gym);
    return g->action;
}

static void CGYM_SEND_ACTION(cgym_t *g)
{
    fwrite(g->action, sizeof(char), g->action_bytes, g->python_gym);
    fflush(g->python_gym);
}

static void CGYM_PRINT(cgym_t *g, int n)
{
    for(int i = 0; i < n; i++) fprintf(stderr, "%s\n",CGYM_READ(g));
}


cgym_t* gymCreate(char envname[])
{
    cgym_t* g = (cgym_t*)malloc(sizeof(cgym_t));
    g->python_gym = sopen("python3 cgym.py");
    CGYM_EXEC(g, envname);
    g->state_bytes = atoi(CGYM_READ(g));
    g->state = (char*)malloc(sizeof(char) * g->state_bytes);
    g->action_bytes = atoi(CGYM_READ(g));
    g->action = (char*)malloc(sizeof(char) * g->action_bytes);
    CGYM_READ_STATE(g);
    CGYM_READ_ACTION(g);
    g->reward = atof(CGYM_READ(g));
    g->done = atoi(CGYM_READ(g));
    return g;
}

void gymDestroy(cgym_t *g)
{
    CGYM_EXEC(g, "close");
    pclose(g->python_gym);
    free(g->state);
    free(g->action);
}

void gymEnvInfo(cgym_t *g)
{
    CGYM_EXEC(g, "envinfo");
    CGYM_PRINT(g, 3);
}

char* gymState(cgym_t *g)
{
    return g->state;
}

char* gymAction(cgym_t *g)
{
    return g->action;
}

void gymStep(cgym_t *g)
{
    CGYM_EXEC(g, "step");
    CGYM_SEND_ACTION(g);
    CGYM_READ_STATE(g);
    g->reward = atof(CGYM_READ(g));
    g->done   = atoi(CGYM_READ(g));
}

float gymReward(cgym_t *g)
{
    return g->reward;
}

int gymDone(cgym_t *g)
{
    return g->done;
}

void gymReset(cgym_t *g)
{
    CGYM_EXEC(g, "reset");
    CGYM_READ_STATE(g);
}

void gymRender(cgym_t *g)
{
    CGYM_EXEC(g, "render");
}
