import csv
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

iterations = 2


name_train_PC = "MP_NN_ALL_TRAIN_PC_Thu_Mar_12_13_39_53_2020"
name_run_PC = "MP_NN_ALL_RUN_PC_Thu_Mar_12_15_11_03_2020"

name_train_BB = "MP_NN_ALL_TRAIN_PC_Tue_Mar_10_12_57_42_2020"
name_run_BB = "MP_NN_ALL_RUN_PC_Tue_Mar_10_12_58_04_2020"

name_train_PI = "MP_NN_ALL_TRAIN_PC_Tue_Mar_10_12_57_42_2020"
name_run_PI = "MP_NN_ALL_RUN_PC_Tue_Mar_10_12_58_04_2020"

name_train_TX2 = "MP_NN_ALL_TRAIN_PC_Tue_Mar_10_12_57_42_2020"
name_run_TX2 = "MP_NN_ALL_RUN_PC_Tue_Mar_10_12_58_04_2020"

name_train_NANO = "MP_NN_ALL_TRAIN_PC_Tue_Mar_10_12_57_42_2020"
name_run_NANO = "MP_NN_ALL_RUN_PC_Tue_Mar_10_12_58_04_2020"

name_train_CORAL = "MP_NN_ALL_TRAIN_CORAL_Mon_Mar_9_13_03_08_2020"
name_run_CORAL = "MP_NN_ALL_RUN_CORAL_Mon_Mar_9_13_20_52_2020"

devices_run = [name_run_PC, name_run_BB, name_run_PI, name_run_TX2, name_run_NANO, name_run_CORAL]
devices_train = [name_train_PC, name_train_BB, name_train_PI, name_train_TX2, name_train_NANO, name_train_CORAL]

temp, temp1, temp2, temp3 = [], [], [], []


data_PC_CPU, data_BB_CPU, data_PI_CPU, data_TX2_CPU, data_NANO_CPU, data_CORAL_CPU = [], [], [], [], [], []
data_PC_time, data_BB_time, data_PI_time, data_TX2_time, data_NANO_time, data_CORAL_time = [], [], [], [], [], []

data_train_CPU = [data_PC_CPU, data_BB_CPU, data_PI_CPU, data_TX2_CPU, data_NANO_CPU, data_CORAL_CPU]
data_train_time = [data_PC_time, data_BB_time, data_PI_time, data_TX2_time, data_NANO_time, data_CORAL_time]

cpu_percent = []
virtual_mem = []
time_diff = []

programs = ["compair", "friction", "narendra4", "pt2",
            "P0Y0_narendra4", "P0Y0_compair", "gradient"]
labels_cpu = ["compair", "friction", "narendra4", "pt2",
              "P0Y0_narendra4", "P0Y0_compair", "gradient", "no operations"]
labels_time = ["compair", "friction", "narendra4", "pt2",
               "P0Y0_narendra4", "P0Y0_compair", "gradient", "Totaal"]

program_i = 0
device_i = 0


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


for device in devices_train:
    with open('D:/School/Masterproef/Python/pyrenn/Logging/' + device + ".csv", mode='r') as results_file:
        results_reader = csv.DictReader(results_file)
        for row in results_reader:
            temp1.append(float(row['CPU Percentage']))
            temp2.append(round(float(row['timediff']), 5))

    for program in programs:
        cpu_avg = round(mean(temp1[program_i*iterations:(program_i*iterations+iterations)]), 5)
        time_diff_avg = round(mean(temp2[program_i*iterations:(program_i*iterations+iterations)]), 5)
        program_i += 1

        data_train_CPU[device_i].append(cpu_avg)
        data_train_time[device_i].append(time_diff_avg)
    data_train_CPU[device_i].append(temp1[-1])
    data_train_time[device_i].append(temp2[-1])

    program_i = 0
    device_i += 1
    temp1, temp2 = [], []

x = np.arange(len(labels_time))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 5*width / 2, data_train_time[0], width, label='PC')
rects2 = ax.bar(x - 3*width / 2, data_train_time[1], width, label='BB')
rects3 = ax.bar(x - width / 2, data_train_time[2], width, label='PI')
rects4 = ax.bar(x + width / 2, data_train_time[3], width, label='TX2')
rects5 = ax.bar(x + 3*width / 2, data_train_time[4], width, label='NANO')
rects6 = ax.bar(x + 5*width / 2, data_train_time[5], width, label='CORAL')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time program execution during training')
ax.set_title('Time program execution for each device during training')
ax.set_xticks(x)
ax.set_xticklabels(labels_time)
ax.legend()

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

fig.tight_layout()
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.yscale("log")
plt.show()

# plot cpu usage
x = np.arange(len(labels_cpu))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 5*width / 2, data_train_CPU[0], width, label='PC')
rects2 = ax.bar(x - 3*width / 2, data_train_CPU[1], width, label='BB')
rects3 = ax.bar(x - width / 2, data_train_CPU[2], width, label='PI')
rects4 = ax.bar(x + width / 2, data_train_CPU[3], width, label='TX2')
rects5 = ax.bar(x + 3*width / 2, data_train_CPU[4], width, label='NANO')
rects6 = ax.bar(x + 5*width / 2, data_train_CPU[5], width, label='CORAL')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('CPU usage during training')
ax.set_title('CPU usage for each device during training')
ax.set_xticks(x)
ax.set_xticklabels(labels_time)
ax.legend()

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

fig.tight_layout()
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()


# Plot running parameters
data_PC_CPU, data_BB_CPU, data_PI_CPU, data_TX2_CPU, data_NANO_CPU, data_CORAL_CPU = [], [], [], [], [], []
data_PC_time, data_BB_time, data_PI_time, data_TX2_time, data_NANO_time, data_CORAL_time = [], [], [], [], [], []

data_run_CPU = [data_PC_CPU, data_BB_CPU, data_PI_CPU, data_TX2_CPU, data_NANO_CPU, data_CORAL_CPU]
data_run_time = [data_PC_time, data_BB_time, data_PI_time, data_TX2_time, data_NANO_time, data_CORAL_time]

program_i = 0
device_i = 0

for device in devices_run:
    with open('D:/School/Masterproef/Python/pyrenn/Logging/' + device + ".csv", mode='r') as results_file:
        results_reader = csv.DictReader(results_file)
        for row in results_reader:
            temp1.append(float(row['CPU Percentage']))
            temp2.append(round(float(row['timediff']), 5))

    for program in programs:
        cpu_avg = round(mean(temp1[program_i*iterations:(program_i*iterations+iterations)]), 5)
        time_diff_avg = round(mean(temp2[program_i*iterations:(program_i*iterations+iterations)]), 5)
        program_i += 1

        data_run_CPU[device_i].append(cpu_avg)
        data_run_time[device_i].append(time_diff_avg)
    data_run_CPU[device_i].append(temp1[-1])
    data_run_time[device_i].append(temp2[-1])

    program_i = 0
    device_i += 1
    temp1, temp2 = [], []

x = np.arange(len(labels_time))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 5*width / 2, data_run_time[0], width, label='PC')
rects2 = ax.bar(x - 3*width / 2, data_run_time[1], width, label='BB')
rects3 = ax.bar(x - width / 2, data_run_time[2], width, label='PI')
rects4 = ax.bar(x + width / 2, data_run_time[3], width, label='TX2')
rects5 = ax.bar(x + 3*width / 2, data_run_time[4], width, label='NANO')
rects6 = ax.bar(x + 5*width / 2, data_run_time[5], width, label='CORAL')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time program execution during running')
ax.set_title('Time program execution for each device during running')
ax.set_xticks(x)
ax.set_xticklabels(labels_time)
ax.legend()


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

fig.tight_layout()
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.yscale("log")
plt.show()

# plot cpu usage
x = np.arange(len(labels_cpu))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 5*width / 2, data_run_CPU[0], width, label='PC')
rects2 = ax.bar(x - 3*width / 2, data_run_CPU[1], width, label='BB')
rects3 = ax.bar(x - width / 2, data_run_CPU[2], width, label='PI')
rects4 = ax.bar(x + width / 2, data_run_CPU[3], width, label='TX2')
rects5 = ax.bar(x + 3*width / 2, data_run_CPU[4], width, label='NANO')
rects6 = ax.bar(x + 5*width / 2, data_run_CPU[5], width, label='CORAL')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('CPU usage during running')
ax.set_title('CPU usage for each device during running')
ax.set_xticks(x)
ax.set_xticklabels(labels_time)
ax.legend()

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

fig.tight_layout()
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.show()
