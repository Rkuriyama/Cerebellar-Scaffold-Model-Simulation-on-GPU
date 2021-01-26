#include"cgym.h"

int main()
{
    gymCreate("MountainCar-v0");
    gymEnvInfo();
    
    float64* state = (float64*)gymState();
    int64* action = (int64*)gymAction();
    float reward = 0.f;
    int done = 0;

    fprintf(stderr, "start\n");
    const int max_trial = 10;
    for(int trial = 0; trial < max_trial; trial++)
    {
        while(1)
        {
            printf("%f %f %f %d\n",state[0], state[1], reward, done);
            action[0] = state[1] > 0 ? 2 : 0;
            gymStep();
            reward = gymReward();
            done   = gymDone();
            
            gymRender();

            if(done)
            {
                gymReset();
                break;
            }
        }
    }

    gymDestroy();
}
