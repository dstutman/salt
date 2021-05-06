from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

probs = np.array([0,10,30,50,70,100])
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
                      ["LD02", 30, 5],
                      ["LD03", 30, 4],
                      ["LD04", 30, 4],
                      ["LD05", 0, 5],
                      ["LD06", 30, 4],
                      ["LD07", 30, 5],
                      ["OPS01", 0, 5],
                      ["OPS02", 0, 5],
                      ["OPS03", 10, 4],
                      ["OPS04", 10, 4],
                      ["OPS05", 0, 3],
                      ["OPS06", 10, 5],
                      ["OPS07", 10, 5],
                      ["OPS08", 0, 5],
                      ["OPT01", 10, 5],
                      ["OPT02", 10, 5],
                      ["OPT03", 0, 5],
                      ["PL01", 50, 5],
                      ["PL02", 70, 5],
                      ["PL03", 10, 5],
                      ["TE01", 10, 4],
                      ["TE02", 10, 4],
                      ["TE03", 0, 5],
                      ["TE04", 0, 5],
                      ["TE05", 0, 4]], dtype=object)


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
                          [ 0, 3],   # ld02
                          [30, 2],   # ld03
                          [10, 2],   # ld04
                          [0, 2],    # ld05
                          [10, 2],   # ld06
                          [0, 1],   # ld07
                          [ 0, 3],   # ops01
                          [ 0, 3],   # ops02
                          [ 10, 2],   # ops03
                          [ 10, 3],   # ops04
                          [ 0, 1],   # ops05
                          [ 10, 3],   # ops06
                          [ 10, 2],   # ops07
                          [ 0, 4],   # ops08
                          [ 10, 2],   # opt01
                          [ 0, 5],   # opt02
                          [ 0, 3],   # opt03
                          [ 10, 3],   # pl01
                          [ 30, 2],   # pl02
                          [0, 2],    # pl03
                          [ 0, 3],   # te01
                          [ 0, 1],   # te02
                          [ 0, 3],   # te03
                          [ 0, 2],   # te04
                          [ 0, 2]])  # te05


vgrid = np.meshgrid(probs,impacts)
hgrid = np.meshgrid(impacts, probs)

# centering
risks_0[:,1] = risks_0[:,1] + 3
risks_1[:,1] = risks_1[:,1] + 3
risks_0[:,2] = risks_0[:,2] - 0.5
risks_1[:,2] = risks_1[:,2] - 0.5


sns.set_theme()
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
plt.hlines([3,2,1], [10,30,50], [30,50,100], linewidth=perimeterthickness,  zorder=4)
plt.vlines([10,30,50], [3,2,1], [5,3,2], linewidth=perimeterthickness, zorder=4)

# risks_0
for impact in impacts:
    for prob in probs:
        risksorderlist = np.where(risks_0[:,2]==impact-0.5)[0]
        probsorderlist = np.where(risks_0[risksorderlist,1]==prob+3)[0]
        order = 0.35 - np.linspace(0,np.size(probsorderlist),np.size(probsorderlist))/(np.size(probsorderlist)*1.5)
        risks_0[risksorderlist[probsorderlist],2] = risks_0[risksorderlist[probsorderlist],2] + order

# risks_1
for impact in impacts:
    for prob in probs:
        risksorderlist = np.where(risks_1[:,2]==impact-0.5)[0]
        probsorderlist = np.where(risks_1[risksorderlist,1]==prob+3)[0]
        order = 0.35 - np.linspace(0,np.size(probsorderlist),np.size(probsorderlist))/(np.size(probsorderlist)*1.5)
        risks_1[risksorderlist[probsorderlist],2] = risks_1[risksorderlist[probsorderlist],2] + order


# plt.scatter(risks_0[:,1], risks_0[:,2], marker="X", c = "k", zorder=6)
# plt.title("Risk Map of Unmitigated Risks", fontsize="x-large")
#
# for i,risk in enumerate(risks_0[:,0]):
#     plt.text(x=risks_0[i,1]+1.3, y=risks_0[i,2]-0.05, s=risk, fontsize="large", zorder=6)

plt.scatter(risks_1[:,1], risks_1[:,2], marker="X", c = "k", zorder=6)
plt.title("Risk Map of Mitigated Risks", fontsize="x-large")

for i,risk in enumerate(risks_1[:,0]):
    plt.text(x=risks_1[i,1]+1.3, y=risks_1[i,2]-0.05, s=risk, fontsize="large", zorder=6)

plt.show()
