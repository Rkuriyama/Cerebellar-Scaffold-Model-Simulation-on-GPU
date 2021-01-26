#include<stdio.h>
#include<stdlib.h>
#include"cgym.h"

int main(void)
{
    cgym_t *gym = gymCreate("MountainCar-v0");
    gymEnvInfo(gym);

    float64* state = (float64*)gymState(gym);
    int64* action = (int64*)gymAction(gym);
    float reward = 0.f;
    int done = 0;

    const int max_trial = 10;
    for(int trial = 0; trial < max_trial; trial++)
    {
        int t = 0;
        while(1)
        {
            //printf("%f %f %f %d\n",state[0], state[1], reward, done);
            //action[0] = state[1] > 0 ? 2 : 0;
            action[0] = 0;
            gymStep(gym);
            reward = gymReward(gym);
            done   = gymDone(gym);
            
            gymRender(gym);
            t++;

            if(done)
            {
                fprintf(stderr, "%d\t%d\n", trial, t);
                gymReset(gym);
                break;
            }
        }
    }

    gymDestroy(gym);
    return 0;
}
