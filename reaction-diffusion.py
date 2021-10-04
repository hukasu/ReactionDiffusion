import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

animation_length = 5000

width = 256
height = 256

diffuseA = 1
diffuseB = 0.5
feed = 0.055
kill = 0.062

# Random Seed
curA = (np.random.rand(width, height) / 4) + .75
curB = (np.random.rand(width, height) / 4)

# Manual disturbance
for i in range(0, 1024, 8):
    curB[(i+2):(i+6), :] = 0.15

fig = plt.figure()
im = plt.imshow(curA, cmap='gist_gray_r', vmin=0, vmax=1, animated=True)

def laplace(nar: np.ndarray) -> np.array:
    # np.roll has wrap-around
    return (
        (-1 * nar) +
        (0.2  * np.roll(nar, (1 ,  0), axis=(0,1))) +
        (0.2  * np.roll(nar, (0 ,  1), axis=(0,1))) +
        (0.2  * np.roll(nar, (-1,  0), axis=(0,1))) +
        (0.2  * np.roll(nar, (0 , -1), axis=(0,1))) +
        (0.05 * np.roll(nar, (1 ,  1), axis=(0,1))) +
        (0.05 * np.roll(nar, (1 , -1), axis=(0,1))) +
        (0.05 * np.roll(nar, (-1,  1), axis=(0,1))) +
        (0.05 * np.roll(nar, (-1, -1), axis=(0,1)))
    )

def reaction_diffuse(i):
    global curA
    global curB

    if i == (animation_length - 1):
        print("Animation Finished.")

    abb = curA * curB * curB

    nextA = curA + (
        (diffuseA * laplace(curA)) -
        (abb) +
        (feed * (1 - curA))
    )
    nextB = curB + (
        (diffuseB * laplace(curB)) +
        (abb) -
        ((feed + kill) * curB)
    )
    
    im.set_data(nextA)

    curA = nextA
    curB = nextB
    
    return im

anim = ani.FuncAnimation(fig, reaction_diffuse, animation_length, interval=0, repeat=False)
plt.show()