#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>


#define T_MAX 10
#define N 4

struct thdata_A{
    int  tid;
    pthread_t th;
    sem_t *ready;
    sem_t *done;
};

struct thdata_B{
    pthread_t th;
    sem_t **A_done;
    sem_t B_ready;
    int A_num;
};

void *thread_A(void *thdata){
    struct thdata_A *priv = (struct thdata_A *)thdata;
    int t;
    for(t = 0; t <T_MAX;t++){
        sem_wait(&(priv->ready[t]));
        printf("\t%d:done %d\n",t,priv->tid);
        fflush(stdout);
        sem_post(&(priv->done[t]));
    }
    return (void*)NULL;
}

void *thread_B(void *thdata){
    struct thdata_B *priv = (struct thdata_B *)thdata;
    int i;
    int t;
    for(t=0; t<T_MAX;t++){
        sem_post(&priv->B_ready);
        for(i = 0; i < priv->A_num; i++){
            sem_wait(&(priv->A_done[i][t]));
        }
        printf("B_done:%d\n",t);
        fflush(stdout);

    }
}

int main(void){
    struct thdata_A dataA[N];
    struct thdata_B dataB;
    int i = 0;
    int t = 0;

    dataB.A_done = (sem_t **)malloc(sizeof(sem_t*)*N);

    for(i = 0; i < N; i++){
        dataA[i].tid = i;
        dataA[i].ready = (sem_t *)malloc(sizeof(sem_t)*T_MAX);
        dataA[i].done = (sem_t *)malloc(sizeof(sem_t)*T_MAX);
        for(t = 0; t < T_MAX; t++){
            sem_init(&(dataA[i].ready[t]), 0, 0);
            sem_init(&(dataA[i].done[t]), 0, 0);
        }
        dataB.A_done[i] = dataA[i].done;
    }

    dataB.A_num = N;
    sem_init(&dataB.B_ready, 0, 0);

    for(i = 0; i < N; i++) pthread_create(&dataA[i].th, NULL, thread_A, (void *)(&dataA[i]));
    pthread_create(&dataB.th, NULL, thread_B, (void *)(&dataB));


    for(t = 0; t < T_MAX; t++){
        sem_wait(&(dataB.B_ready));
        printf("%d\n",t);
        #pragma omp parallel for
        for(i = 0; i < N; i++){
            printf("\t%d:ready %d\n",t, i);
            fflush(stdout);
            sem_post(&(dataA[i].ready[t]));
        }
    }
    for(i = 0; i < N; i++){
        pthread_join(dataA[i].th, NULL);
        for(t=0;t<T_MAX;t++)
        sem_destroy(&(dataA[i].ready[t]));
    }
    pthread_join(dataB.th, NULL);
    for(i = 0; i < N; i++){
        for(t = 0; t < T_MAX; t++){
            sem_destroy(&(dataB.A_done[i][t]));
        }
    }
    free(dataB.A_done);
    return 0;
}
