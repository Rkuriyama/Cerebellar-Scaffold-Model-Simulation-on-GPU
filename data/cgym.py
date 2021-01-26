import gym
import sys
import time
import numpy as np

ascode = 'utf-8'
endian = 'little'

def send(message, encode = ascode):
    data = message.encode(encode)
    size = int(len(data)).to_bytes(4, endian)
    sys.stdout.buffer.write(size)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()

def read(decode = ascode):
    size = int.from_bytes(sys.stdin.buffer.read(4), endian)
    data = sys.stdin.buffer.read(size)
    return data.decode(ascode)

def sample_itemsize(sample):
    if type(sample) == int:
        return 8
    return sample.dtype.itemsize

def space_size(space):
    size = 1
    for dim in space.shape:
        size *= dim
    return size

def send_observation():
    sys.stdout.buffer.write(observation.tobytes())
    sys.stdout.buffer.flush()

def read_action():
    data = sys.stdin.buffer.read(act_bytes)
    if type(action) == int:
        return int.from_bytes(data, endian)
    else:
        return np.frombuffer(data, dtype = act_dtype)

def send_action():
    if type(action) == int:
        sys.stdout.buffer.write(action.to_bytes(8, endian))
    else:
        sys.stdout.buffer.write(action.tobytes())
    sys.stdout.buffer.flush()
        

# read environment name
envname     = read('utf-8')
# make environment
env         = gym.make(envname)
# initialize environment
observation = env.reset()
action      = env.action_space.sample()
done        = False
reward      = 0.0

# observation and action space
obs_space   = env.observation_space
act_space   = env.action_space
obs_dsize   = space_size(obs_space)
act_dsize   = space_size(act_space)
obs_bytes   = obs_dsize * sample_itemsize(observation)
act_bytes   = act_dsize * sample_itemsize(action)
obs_dtype   = observation.dtype
act_dtype   = act_space.dtype

# send observation and action bytesizes
send(str(obs_bytes))
send(str(act_bytes))

# send initial data
send_observation()
send_action()
send(str(reward))
send(str(1 if done else 0))

while True:
    command = read('utf-8')
    if command == 'step':
        action = read_action()
        observation, reward, done, _ = env.step(action)
        send_observation()
        send(str(reward))
        send(str(1 if done else 0))
    elif command == 'reset':
        observation = env.reset()
        send_observation()
    elif command == 'render':
        env.render()
        time.sleep(1/24)
    elif command == 'envinfo':
        send(str(env))
        send("observation : " + str(obs_dtype) + "[" + str(obs_dsize) + "]" + ", " + str(obs_space))
        send("action      : " + str(act_dtype) + "[" + str(act_dsize) + "]" + ", " + str(act_space))
    elif command == 'close':
        env.close()
        break
    else:
        print("invalid command :" + c, file=sys.stderr)
        env.close()
        break
