import csv
import matplotlib.pyplot as plt

name_train = "MP_NN_ALL_TRAIN_PC_Thu_Feb_20_14_39_54_2020"
name_run = "MP_NN_ALL_RUN_PC_Thu_Feb_20_14_40_05_2020"

cpu_percent = [0, 0, 0, 0, 0, 0, 0, 0]
virtual_mem = [0, 0, 0, 0, 0, 0, 0, 0]
time_diff = [0, 0, 0, 0, 0, 0, 0, 0]
labels = ["", "", "", "", "", "", "", ""]
line_index = 0


#############################
# plotting train data

###
# Code om files in te lezen vanuit csv file
# Code werd opgeslagen als dictionary (wat is het verschil met gewone???)
with open('D:/School/Masterproef/Python/pyrenn/Logging/' + name_train + ".csv", mode='r') as results_file:
    results_reader = csv.DictReader(results_file)

    for row in results_reader:
        labels[line_index] = row['Naam']
        cpu_percent[line_index] = float(row['CPU Percentage'])
        time_diff[line_index] = round(float(row['timediff']), 5)
        virtual_mem[line_index] = row['virtual mem']
        line_index += 1
    line_index = 0



###
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

###
# Code om files in te lezen vanuit csv file
# Code werd opgeslagen als dictionary (wat is het verschil met gewone???)
with open('D:/School/Masterproef/Python/pyrenn/Logging/' + name_run + ".csv", mode='r') as results_file:
    results_reader = csv.DictReader(results_file)

    for row in results_reader:
        labels[line_index] = row['Naam']
        cpu_percent[line_index] = float(row['CPU Percentage'])
        time_diff[line_index] = round(float(row['timediff']), 5)
        virtual_mem[line_index] = row['virtual mem']
        line_index += 1


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
