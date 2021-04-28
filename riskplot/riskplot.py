from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

probs = np.array([0,20,40,60,80,100])
impacts = np.array([0,1,2,3,4,5])
urgency_rate = 2.5
unurgency_rate = 10
linestrength = 0.2
perimeterthickness = 1.2

# risks_0 = np.array([["C01", 60, 3],
#                     ["C02", 40, 5],
#                     ["C03", 60, 3],
#                     ["C04", 20, 3],
#                     ["HR01", 40, 3],
#                     ["HR02", 40, 4],
#                     ["HR03", 40, 4],
#                     ["HR04", 20, 4],
#                     ["HR05", 40, 3],
#                     ["HR06", 0, 4],
#                     ["M01", 60, 5],
#                     ["M02", 40, 3],
#                     ["M03", 20, 4],
#                     ["M04", 40, 4],
#                     ["M05", 0, 4],
#                     ["M06", 60, 5],
#                     ["HS01", 20, 4],
#                     ["HS02", 40, 5],
#                     ["HS03", 20, 4],
#                     ["HS04", 40, 4],
#                     ["HS05", 80, 4],
#                     ["HS06", 0, 4]], dtype=object)

# technical risks
risks_0 = np.array([["LD01",  0, 5],
                      ["LD02", 40, 5],
                      ["LD03", 40, 4],
                      ["LD04", 40, 4],
                      ["LD05", 60, 5],
                      ["LD06", 40, 4],
                      ["TE01", 20, 4],
                      ["TE02", 20, 4],
                      ["TE03", 0, 5],
                      ["TE04", 0, 5],
                      ["TE05", 0, 4],
                      ["01", 60, 3]], dtype=object)


# risks_1 = np.zeros(np.shape(risks_0),dtype=object)
# risks_1[:, 0] = risks_0[:, 0]
# risks_1[:, 1:] = np.array([[20,3],   # c01
#                           [0,3],    # c02
#                           [40,1],   # c03
#                           [0,3],    # c04
#                           [20,1],   # hr01
#                           [0,4],    # hr02
#                           [20,3],   # hr03
#                           [0,1],    # hr04
#                           [20,3],   # hr05
#                           [0,2],    # hr06
#                           [40,2],   # m01
#                           [20,2],   # m02
#                           [0,3],    # m03
#                           [0,4],    # m04
#                           [0,3],    # m05
#                           [40,2],   # m06
#                           [20,2],   # hs01
#                           [20,3],   # hs02
#                           [0,4],    # hs03
#                           [20,2],   # hs04
#                           [20,3],   # hs05
#                           [0,2]])   #

#technical risks
risks_1 = np.zeros(np.shape(risks_0),dtype=object)
risks_1[:, 0] = risks_0[:, 0]
risks_1[:, 1:] = np.array([[0, 3],   # ld01
                          [ 0, 3],    # ld02
                          [40, 2],   # ld03
                          [20, 2],   # ld04
                          [0, 2],    # ld05
                          [20, 2],   # ld06
                          [ 0, 3],   # te01
                          [ 0, 1],   # te02
                          [ 0, 3],   # te03
                          [ 0, 2],   # te04
                          [ 0, 2],   # te05
                          [0, 2]])


vgrid = np.meshgrid(probs,impacts)
hgrid = np.meshgrid(impacts, probs)

# centering
risks_0[:,1] = risks_0[:,1] + 3
risks_1[:,1] = risks_1[:,1] + 3
risks_0[:,2] = risks_0[:,2] - 0.5
risks_1[:,2] = risks_1[:,2] - 0.5


sns.set_theme()
plt.title("Risk Map of Unmitigated Risks", fontsize="x-large")
plt.ylim(0,np.max(impacts))
plt.xlim(0, np.max(probs))
plt.plot(vgrid[0],vgrid[1], c = "k", ls = "-.", alpha = linestrength, zorder=4)
plt.plot(hgrid[1], hgrid[0], c = "k", ls = "-.", alpha = linestrength, zorder=4)
plt.ylabel("Measure of impact Scaled from 1 to 5", fontsize="x-large")
plt.xlabel("Probability of Occurrence [%]", fontsize="x-large")

for j,tile in enumerate(vgrid[0][0:-1]):
    for i,x in enumerate(tile[0:-1]):
        x1 = x
        x2 = tile[i+1]
        y1 = vgrid[1][j,i]
        y2 = vgrid[1][j+1,i]

        # red zone
        plt.fill([x1,x1,x2,x2],[y1,y2,y2,y1],c="r",alpha=((x2*y2)/
                                                          (np.max(probs)*np.max(impacts)))**(1/urgency_rate),
                 zorder=3)


        # green zone
        plt.fill([x1,x1,x2,x2],[y1,y2,y2,y1],c="seagreen",alpha=np.exp(-(x2*y2)/
                                                                       (np.max(probs)*np.max(impacts))*unurgency_rate),
                 zorder=4)

# unacceptable region
plt.hlines([3,2,1], [20,40,60], [40,60,100], linewidth=perimeterthickness,  zorder=4)
plt.vlines([20,40,60], [3,2,1], [5,3,2], linewidth=perimeterthickness, zorder=4)

# # risks_0
# for impact in impacts:
#     for prob in probs:
#         risksorderlist = np.where(risks_0[:,2]==impact-0.5)[0]
#         probsorderlist = np.where(risks_0[risksorderlist,1]==prob+3)[0]
#         order = 0.35 - np.linspace(0,np.size(probsorderlist),np.size(probsorderlist))/(np.size(probsorderlist)*1.5)
#         risks_0[risksorderlist[probsorderlist],2] = risks_0[risksorderlist[probsorderlist],2] + order
#
# # risks_1
# for impact in impacts:
#     for prob in probs:
#         risksorderlist = np.where(risks_1[:,2]==impact-0.5)[0]
#         probsorderlist = np.where(risks_1[risksorderlist,1]==prob+3)[0]
#         order = 0.35 - np.linspace(0,np.size(probsorderlist),np.size(probsorderlist))/(np.size(probsorderlist)*1.5)
#         risks_1[risksorderlist[probsorderlist],2] = risks_1[risksorderlist[probsorderlist],2] + order
#
# plt.scatter(risks_0[:,1], risks_0[:,2], marker="X", c = "k", zorder=6)

# risks_th_0
for impact in impacts:
    for prob in probs:
        risksorderlist = np.where(risks_th_0[:,2]==impact-0.5)[0]
        probsorderlist = np.where(risks_th_0[risksorderlist,1]==prob+3)[0]
        order = 0.35 - np.linspace(0,np.size(probsorderlist),np.size(probsorderlist))/(np.size(probsorderlist)*1.5)
        risks_th_0[risksorderlist[probsorderlist],2] = risks_th_0[risksorderlist[probsorderlist],2] + order

# risks_th_1
for impact in impacts:
    for prob in probs:
        risksorderlist = np.where(risks_th_1[:,2]==impact-0.5)[0]
        probsorderlist = np.where(risks_th_1[risksorderlist,1]==prob+3)[0]
        order = 0.35 - np.linspace(0,np.size(probsorderlist),np.size(probsorderlist))/(np.size(probsorderlist)*1.5)
        risks_th_1[risksorderlist[probsorderlist],2] = risks_th_1[risksorderlist[probsorderlist],2] + order

plt.scatter(risks_th_0[:,1], risks_th_0[:,2], marker="X", c = "k", zorder=6)


for i,risk in enumerate(risks_0[:,0]):
    plt.text(x=risks_0[i,1]+1.3, y=risks_0[i,2]-0.05, s=risk, fontsize="large", zorder=6)

# plt.scatter(risks_1[:,1], risks_1[:,2], marker="X", c = "k", zorder=6)
#
#
# for i,risk in enumerate(risks_1[:,0]):
#     plt.text(x=risks_1[i,1]+1.3, y=risks_1[i,2]-0.05, s=risk, fontsize="large", zorder=6)

plt.show()
