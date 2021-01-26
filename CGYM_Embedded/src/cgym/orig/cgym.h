#include<stdio.h>
#include<stdlib.h>
#include<string.h>

typedef double float64;
typedef float  float32;
typedef long long int int64;

#define GYMBUFLEN 256
char    GYMBUF[GYMBUFLEN];
FILE*   GYM;
char*   GYM_STATE;
char*   GYM_ACTION;
int     GYM_STATE_BYTES;
int     GYM_ACTION_BYTES;
float   GYM_REWARD;
int     GYM_DONE;

void GYM_EXEC(char exec[])
{
    int len = strlen(exec);
    fwrite(&len, sizeof(int), 1, GYM);
    fwrite(exec, sizeof(char), len, GYM);
    fflush(GYM);
}

char* GYM_READ()
{
    int len = 0;
    fread(&len, sizeof(int), 1, GYM);
    fread(GYMBUF, sizeof(char), len, GYM);
    GYMBUF[len] = 0;
    return GYMBUF;
}

char* GYM_READ_STATE()
{
    fread(GYM_STATE, sizeof(char), GYM_STATE_BYTES, GYM);
    return GYM_STATE;
}

char* GYM_READ_ACTION()
{
    fread(GYM_ACTION, sizeof(char), GYM_ACTION_BYTES, GYM);
    return GYM_ACTION;
}

void GYM_SEND_ACTION()
{
    fwrite(GYM_ACTION, sizeof(char), GYM_ACTION_BYTES, GYM);
    fflush(GYM);
}

void GYM_PRINT(int n)
{
    for(int i = 0; i < n; i++) fprintf(stderr, "%s\n",GYM_READ());
}


void gymCreate(char envname[])
{
    //GYM = popen("python3 cgym.py","r+");
    GYM = popen("python3 cgym.py","r");
    GYM_EXEC(envname);
    GYM_STATE_BYTES  = atoi(GYM_READ());
    GYM_STATE        = (char*)malloc(sizeof(char) * GYM_STATE_BYTES);
    GYM_ACTION_BYTES = atoi(GYM_READ());
    GYM_ACTION       = (char*)malloc(sizeof(char) * GYM_ACTION_BYTES);
    GYM_READ_STATE();
    GYM_READ_ACTION();
    GYM_REWARD = atof(GYM_READ());
    GYM_DONE   = atoi(GYM_READ());
}

void gymDestroy()
{
    GYM_EXEC("close");
    pclose(GYM);
    free(GYM_STATE);
    free(GYM_ACTION);
}

void gymEnvInfo()
{
    GYM_EXEC("envinfo");
    GYM_PRINT(3);
}

char* gymState()
{
    return GYM_STATE;
}

char* gymAction()
{
    return GYM_ACTION;
}

void gymStep()
{
    GYM_EXEC("step");
    GYM_SEND_ACTION();
    GYM_READ_STATE();
    GYM_REWARD = atof(GYM_READ());
    GYM_DONE   = atoi(GYM_READ());
}

float gymReward()
{
    return GYM_REWARD;
}

int gymDone()
{
    return GYM_DONE;
}

void gymReset()
{
    GYM_EXEC("reset");
    GYM_READ_STATE();
}

void gymRender()
{
    GYM_EXEC("render");
}

