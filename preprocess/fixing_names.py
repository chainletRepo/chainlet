import os

# Edit the names of the days files to the 0000/00/00 format
files = r'days'
for f in os.listdir(files):
    print(f)
    date = f.split("_")
    if len(date[1]) == 1:
        date[1] = "0" + date[1]
        print(date[1])
    if len(date[2]) == 1:
        date[2] = "0" + date[2]
    date[2] = date[2][:2]
    name = date[0] + "_" + date [1] + "_" + date[2]
    print(name)
    os.rename("days/"+f,"days_edited/"+name)

