#ifndef __CGYM_H__
#define __CGYM_H__

#include<sys/types.h>
#include<sys/socket.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>

typedef double float64;
typedef float  float32;
typedef long long int int64;

#define CGYMBUFLEN 256

typedef struct {
  char    buf [ CGYMBUFLEN ];
  FILE*   python_gym;
  char*   state;
  char*   action;
  int     state_bytes;
  int     action_bytes;
  float   reward;
  int     done;
} cgym_t;

extern cgym_t* gymCreate(char []);
extern void gymDestroy(cgym_t *);
extern void gymEnvInfo(cgym_t *);
extern char* gymState(cgym_t *);
extern char* gymAction(cgym_t *);
extern void gymStep(cgym_t *);
extern float gymReward(cgym_t *);
extern int gymDone(cgym_t *);
extern void gymReset(cgym_t *);
extern void gymRender(cgym_t *);

#endif // __CGYM_H__
