from openai import OpenAI

import time

client = OpenAI()

groupLimit = 20

# Read instructions from Instructions.txt into a single string. Ignore lines that start with #
with open("Instructions.txt", "r") as file:
    instructions = file.read()
instructions = instructions.split("\n")
instructions = [line.strip() for line in instructions if not line.startswith("#")]
# make it a single string that has '\n' between each line
instructions = "\n".join(instructions)

# Read workload from Workload.txt
with open("Workload.txt", "r") as file:
    workloads = file.read()
workloads = workloads.split("\n")
workloads = [line.strip() for line in workloads if not line.startswith("#")]
linecount = len(workloads)

# Separate workloads into chunks of 100 lines each, but repeatedly into two groups, each overlapping by 50 lines
group1 = []
group2 = []
for i in range(0, linecount, groupLimit):
    # check for index out of range
    group1.append(workloads[i:min(i + groupLimit, linecount)])
    if i + groupLimit // 2 < linecount:
        group2.append(workloads[i + groupLimit // 2 : min(i + groupLimit // 2 * 3, linecount)])

# print("Instructions:", instructions)
# print("Workload:", workload)


def requestCompletion(workload):
    return client.chat.completions.create(
        # model="gpt-4o-mini",
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": workload}
        ]
    ).choices[0].message.content

# Clear Result1.txt and Result2.txt
with open("Result1.txt", "w") as file:
    file.write("")
with open("Result2.txt", "w") as file:
    file.write("")

for i in range(len(group1)):
    print("Group 1, chunk", i, "of", len(group1))
    # get the first and last id from the group. 
    firstId = i * groupLimit
    lastId = firstId + len(group1[i]) - 1

    timeStart = time.time()
    result = requestCompletion("\n".join(group1[i]))
    timeEnd = time.time()
    timeElapsed = timeEnd - timeStart
    print("Elapsed", timeElapsed, "s")
    print("Time per line", round(timeElapsed / len(group1[i]) * 1000, 2), "ms")

    # if the result ends with a newline, remove it
    if result.endswith("\n"):
        result = result[:-1]
    
    # separate result into lines
    result = result.split("\n")
    # get the number of lines in the result
    linecount = len(result)
    # check if the number of lines in the result is the same as the number of lines in the workload
    # if not, append the missing lines with "id,ERROR"
    if linecount != len(group1[i]):
        for j in range(firstId + linecount, lastId + 1):
            result.append(f"{j},ERROR")
    # append to Result1.txt
    with open("Result1.txt", "a") as file:
        file.write("\n".join(result) + "\n")

exit()

for i in range(len(group2)):
    print("Group 2, chunk", i, "of", len(group2))
    # get the first and last id from the group.
    firstId = i * groupLimit + groupLimit // 2
    lastId = firstId + len(group2[i]) - 1
    
    timeStart = time.time()
    result = requestCompletion("\n".join(group2[i]))
    timeEnd = time.time()
    timeElapsed = timeEnd - timeStart
    print("Elapsed", timeElapsed, "s")
    print("Time per line", round(timeElapsed / len(group2[i]) * 1000, 2), "ms")
    
    if result.endswith("\n"):
        result = result[:-1]

    result = result.split("\n")
    linecount = len(result)
    if linecount != len(group2[i]):
        for j in range(firstId + linecount, lastId + 1):
            result.append(f"{j},ERROR")

    with open("Result2.txt", "a") as file:
        file.write("\n".join(result) + "\n")

