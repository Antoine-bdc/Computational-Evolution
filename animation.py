import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation


fig = plt.figure()
plt.ion()
ims =[]
total = []
duration=1000
step=1

for i in range(0,duration,step): # a changer apres
    if i%1000==0:print(i)
    max_pred= 0.0
    max_agents =0.0
    file = open("simulation_"+str(1001)+"/"+str(i)+".dat","r")
    tab = [(0.0,0.0,0.0)]*50
    for j in range(50):
        tab[j] = [[0.0,0.0,0.0]]*50
    for line in file:
        text = line.split(" ")
        x = int(text[1])
        y = int(text[2])
        if text[0] == "a":
            tab[x][y] = [0.0,0.0,float(text[3])]
            if float(text[3])>=max_agents:
                max_agents = float(text[3])
        elif text[0] == "p":
            tab[x][y] = [float(text[3]),0.0,0.0]
            if float(text[3])>=max_pred:
                max_pred = float(text[3])
    for i in range(len(tab)):
        for j in range(len(tab[i])):
            tab[i][j][2] /= max_agents
            try: 
                tab[i][j][0] /= max_pred
            except: pass
    total.append(tab)
    #plt.clf()
    
    #time.sleep(0.2)

ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
im = ax.imshow(total[0], interpolation='nearest')

def animate(i):
    im.set_data(total[i])
    return (im,)
anim = animation.FuncAnimation(fig, animate,
                                   frames=500, interval=30)
writer = animation.FFMpegWriter(fps=10, bitrate=5000)
print("Saving ...")
anim.save("animbis.mp4", writer = writer, dpi=360)
print("Saved!")

