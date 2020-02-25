import csv
from statistics import mean

import matplotlib.pyplot as plt

name_train = "MP_NN_ALL_TRAIN_PC_Mon_Feb_24_15_14_23_2020"
name_run = "MP_NN_ALL_RUN_PC_Mon_Feb_24_15_18_52_2020"

temp, temp1, temp2, temp3 = [], [], [], []
cpu_percent = []
virtual_mem = []
time_diff = []
programs = ["compair", "friction", "narendra4", "pt2",
          "P0Y0_narendra4", "P0Y0_compair", "gradient"]
labels = ["compair", "friction", "narendra4", "pt2",
          "P0Y0_narendra4", "P0Y0_compair", "gradient", "Totaal"]
iterations = 2
label_i = 0
iteration_j = 0

#############################
# plotting train data

# Code om files in te lezen vanuit csv file
# Code werd opgeslagen als dictionary (wat is het verschil met gewone???)
with open('D:/School/Masterproef/Python/pyrenn/Logging/' + name_train + ".csv", mode='r') as results_file:
    results_reader = csv.DictReader(results_file)
    for row in results_reader:
        temp1.append(float(row['CPU Percentage']))
        temp2.append(round(float(row['timediff']), 5))
        temp3.append(row['virtual mem'])

for label in programs:
    for iteration in range(iterations):
        index = label_i*iterations + iteration_j
        temp.append(temp1[index])
        iteration_j += 1
    iteration_j = 0
    cpu_percent.append(round(mean(temp),5))
    temp = []

    for iteration in range(iterations):
        temp.append(temp2[label_i * iterations + iteration_j])
        iteration_j += 1
    iteration_j = 0
    time_diff.append(round(mean(temp),5))
    temp = []

    for iteration in range(iterations):
        temp += temp3[label_i * iterations + iteration_j]
        iteration_j += 1
    iteration_j = 0
    #virtual_mem.append(mean(temp))
    temp = []
    label_i += 1

cpu_percent.append(temp1[-1])
time_diff.append(temp2[-1])
virtual_mem.append(temp3[-1])

# Plotting the results
plt.subplot(141)
plt.bar(labels, time_diff)
plt.ylabel('Time Difference')
plt.xlabel('Programma')
plt.xticks(rotation=90)

plt.subplot(142)
plt.bar(labels, cpu_percent)
plt.ylabel('CPU Percentage')
plt.xlabel('Programma')
plt.xticks(rotation=90)
plt.suptitle('Resultaten')

#############################
# plotting run data
label_i = 0
iteration_j = 0
temp, temp1, temp2, temp3 = [], [], [], []
cpu_percent = []
virtual_mem = []
time_diff = []
###
# Code om files in te lezen vanuit csv file
# Code werd opgeslagen als dictionary (wat is het verschil met gewone???)
with open('D:/School/Masterproef/Python/pyrenn/Logging/' + name_run + ".csv", mode='r') as results_file:
    results_reader = csv.DictReader(results_file)
    for row in results_reader:
        temp1.append(float(row['CPU Percentage']))
        temp2.append(round(float(row['timediff']), 5))
        temp3.append(row['virtual mem'])


for label in programs:
    for iteration in range(iterations):
        index = label_i*iterations + iteration_j
        temp.append(temp1[index])
        iteration_j += 1
    iteration_j = 0
    cpu_percent.append(round(mean(temp),5))
    temp = []

    for iteration in range(iterations):
        temp.append(temp2[label_i * iterations + iteration_j])
        iteration_j += 1
    iteration_j = 0
    time_diff.append(round(mean(temp),5))
    temp = []

    for iteration in range(iterations):
        temp += temp3[label_i * iterations + iteration_j]
        iteration_j += 1
    iteration_j = 0
    #virtual_mem.append(mean(temp))
    temp = []
    label_i += 1

cpu_percent.append(temp1[-1])
time_diff.append(temp2[-1])
virtual_mem.append(temp3[-1])


###
# Plotting the results
plt.subplot(143)
plt.bar(labels, time_diff)
plt.ylabel('Time Difference')
plt.xlabel('Programma')
plt.xticks(rotation=90)

plt.subplot(144)
plt.bar(labels, cpu_percent)
plt.ylabel('CPU Percentage')
plt.xlabel('Programma')
plt.xticks(rotation=90)
plt.suptitle('Resultaten')
plt.show()