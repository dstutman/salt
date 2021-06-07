
def tipfinder(data, distanceMod = 0.01, timestep = 0.001):
    tips = []
    for index in range(len(data)):
        if index < 5:
            continue
        # -2 is the current peak we check
        firstCondition = (data[index-4] < data[index-2]
                        and data[index-3] < data[index-2])
        secondCondition = (data[index-2] > data[index-0]
                            and data[index-2] > data[index-1])
        if firstCondition and secondCondition and (data[index] > 0):
            if len(tips) > 10:
                    break
                # start case
            if len(tips) == 0:
                tips.append((index-2)*timestep)
            else:
                # this is to avoid duplicates
                if (index - tips[-1]) > distanceMod:
                    tips.append((index-2)*timestep)
    return tips

def getFreq(data):
    tips = tipfinder(data)
    period = (tips[-1] - tips[0])/len(tips)
    return 1/period
