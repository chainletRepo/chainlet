# Convert orbits of format x:y (x = orbit ID, y = orbit value) to columns of orbits ("o1", "o2", ..., "o47")
if __name__ == "__main__":
    orbitFile = "../rawData/articleOrbitsV3.csv"
    formattedOrbitFile = open("../data/articleOrbitsV3Formatted.csv", 'wb')
    rawOrbitFile = open(orbitFile, 'r')
    lines = rawOrbitFile.readlines()
    numOrbits = 48
    startingIndex = 4
    for line in lines:
        arrs = line.split("\t")
        arr = [0 for i in range(numOrbits+startingIndex)]
        arr[0] = arrs[0]
        arr[1]= arrs[1]
        arr[2]= arrs[2]
        arr[3] = arrs[3]
        for j in range(startingIndex, len(arrs)):
            split = arrs[j].strip().split(":")
            orbitId = int(split[0])
            orbitVal = split[1]
            arr[orbitId+startingIndex] = orbitVal

        formattedOrbitFile.write(bytes(('\t'.join(str(v) for v in arr)+"\r\n"),"UTF-8"))

    rawOrbitFile.close()
    formattedOrbitFile.close()