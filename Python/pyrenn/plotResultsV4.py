import csv
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

iterations = 20
boxplot_bool = True

name_run_PC = "MP_NN_ALL_RUN_PC_Thu_Apr_2_22_05_01_2020"
name_run_PI = "MP_NN_ALL_RUN_PI_Fri_Apr_3_02_32_27_2020"
name_run_NANO = "MP_NN_ALL_RUN_NANO_Fri_Apr_3_02_11_17_2020"
name_run_CORAL = "MP_NN_ALL_RUN_CORAL_Wed_Apr_3_20_08_16_2020"

devices_run = [name_run_PC, name_run_PI, name_run_NANO, name_run_CORAL]

data_run_CPU = [[], [], [], []]
data_run_time = [[], [], [], []]
data_run_CPU_avg = [[], [], [], []]
data_run_time_avg = [[], [], [], []]

cpu_percent = []
virtual_mem = []
time_diff = []

programs = ["compair", "friction", "narendra4", "pt2", "P0Y0_narendra4",
            "P0Y0_compair", "gradient", "FashionMNIST",  "NumberMNIST", "catsVSdogs", "Im Rec"]
labels_cpu = programs + ["no operations"]
labels_time = programs + ["Total"]

ylabel_time = 'Time program execution during running'
ylabel_cpu = 'CPU usage during running'
title_time = 'Time program execution for each device during running'
title_cpu = 'CPU usage for each device during running'

devices = ["PC", "PI", "NANO", "CORAL"]
clockspeed = [2.5, 1.2, 1.43, 1.7]
device_price = [149.99, 99, 41.5, 900]

width_max = 0.22  # the width of the bars
width = 0.22


def show_plot(data, ylabel, titel, labels, log, show, boxplot):
    if not show:
        return

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    data_bar = [[], [], [], []]
    for device in range(len(devices)):
        for program in range(len(labels)):
            data_bar[device].append(float(round(mean(data[device][program]), 3)))
            for iteration in range(iterations):
                data[device][program][iteration] = round(data[device][program][iteration], 3)
    fig, ax = plt.subplots()

    x = np.arange(len(labels))  # the label locations
    rects1 = ax.bar(x - 3 * width_max / 2, data_bar[0], width, label='PC')
    rects2 = ax.bar(x - width_max / 2, data_bar[1], width, label='PI')
    rects3 = ax.bar(x + width_max / 2, data_bar[2], width, label='NANO')
    rects4 = ax.bar(x + 3 * width_max / 2, data_bar[3], width, label='CORAL')
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    if boxplot:
        ax.boxplot(data[0], positions=x - 3 * width_max / 2, widths=width, showfliers=False, patch_artist=True)
        ax.boxplot(data[1], positions=x - width_max / 2, widths=width, showfliers=False)
        ax.boxplot(data[2], positions=x + width_max / 2, widths=width, showfliers=False)
        ax.boxplot(data[3], positions=x + 3 * width_max / 2, widths=width, showfliers=False)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(titel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    if log:
        plt.yscale("log")
    plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data extraction
for device in range(len(devices_run)):
    temp1, temp2, = [], []

    with open('./logging/' + devices_run[device] + ".csv", mode='r') as results_file:
        results_reader = csv.DictReader(results_file)
        for row in results_reader:
            temp2.append(round(float(row['timediff']), 5))
            temp1.append(float(row['CPU Percentage']) / 100)

    for program in range(len(programs)):
        time_diff_avg = round(mean(temp2[program * iterations:(program * iterations + iterations)]), 5)
        cpu_avg = round(mean(temp1[program * iterations:(program * iterations + iterations)]), 5)
        data_run_CPU_avg[device].append(cpu_avg)
        data_run_time_avg[device].append(time_diff_avg)

        data_run_CPU[device].append(temp1[program * iterations:(program * iterations + iterations)])
        data_run_time[device].append(temp2[program * iterations:(program * iterations + iterations)])
    data_run_CPU[device].append([])
    data_run_time[device].append([])

    for iteration in range(iterations):
        data_run_CPU[device][-1].append(temp1[-1])
        data_run_time[device][-1].append(temp2[-1])
    data_run_CPU_avg[device].append(temp1[-1])
    data_run_time_avg[device].append(temp2[-1])

# Plotting figures
show_plot(data_run_time, ylabel_time, title_time, labels_time, log=True, show=True, boxplot=True)
show_plot(data_run_CPU, ylabel_cpu, title_cpu, labels_cpu, log=False, show=True, boxplot=True)

# making sure variables have right shape, content of data_run_time will be ignored
data_run_time_norm = []
data_run_time_MHzCPU = []
data_run_time_MHzCPU_norm = []
data_run_time_MHzCPUprice = []
data_run_time_MHzCPUprice_norm = []

for device in range(len(devices)):
    data_run_time_norm.append([])
    data_run_time_MHzCPU.append([])
    data_run_time_MHzCPU_norm.append([])
    data_run_time_MHzCPUprice.append([])
    data_run_time_MHzCPUprice_norm.append([])

    for program in range(len(labels_time)):
        data_run_time_norm[device].append([])
        data_run_time_MHzCPU[device].append([])
        data_run_time_MHzCPU_norm[device].append([])
        data_run_time_MHzCPUprice[device].append([])
        data_run_time_MHzCPUprice_norm[device].append([])

        for iteration in range(iterations):
            data_run_time_norm[device][program].append([])
            data_run_time_MHzCPU[device][program].append([])
            data_run_time_MHzCPU_norm[device][program].append([])
            data_run_time_MHzCPUprice[device][program].append([])
            data_run_time_MHzCPUprice_norm[device][program].append([])

'''
# converting zeros to nan in data_run_time
for device in range(len(devices)):
    for program in range(len(programs)):
        data_run_time_norm[device].append([])
        if data_run_time_avg[device][program] == 0:
            data_run_time_avg[device][program] = float('nan')
        if data_run_CPU_avg[device][program] == 0:
            data_run_CPU_avg[device][program] = float('nan')
        for iteration in range(iterations):
            if data_run_time[device][program][iteration] == 0:
                data_run_time[device][program][iteration] = float('nan')
            if data_run_CPU[device][program][iteration] == 0:
                data_run_CPU[device][program][iteration] = float('nan')
'''

# Rescaling to lowest nr of each program
program_values = []
for program in range(len(labels_time)):
    program_values = [data_run_time_avg[0][program], data_run_time_avg[1][program],
                      data_run_time_avg[2][program], data_run_time_avg[3][program]]
    minimum = np.nanmin(program_values)
    # print("minimum: ", minimum)
    for device in range(len(devices)):
        for iteration in range(iterations):
            data_run_time_norm[device][program][iteration] = data_run_time[device][program][iteration] / minimum

show_plot(data=data_run_time_norm,
          ylabel=ylabel_time,
          titel=title_time+", Normalised",
          labels=labels_time,
          log=True, show=True, boxplot=True)

# plotting time/MHz/cpu%
for device in range(len(devices)):
    for program in range(len(labels_time)):
        for iteration in range(iterations):
            data_run_time_MHzCPU[device][program][iteration] = \
                data_run_time[device][program][iteration] / clockspeed[device] / data_run_CPU_avg[device][program]

show_plot(data=data_run_time_MHzCPU,
          ylabel="Time compensated for each CPU% and MHz.",
          titel="Time compensated for each CPU% and MHz clockspeed for each device.",
          labels=labels_time, log=True, show=True, boxplot=True)

# plotting normalised time/MHz/cpu%
minimum = []
for label in range(len(labels_time)):
    program_values = []
    for device in range(len(devices)):
        program_values.append(mean(data_run_time_MHzCPU[device][label]))
    minimum.append(np.nanmin(program_values))

for device in range(len(devices)):
    for program in range(len(labels_time)):
        for iteration in range(iterations):
            data_run_time_MHzCPU_norm[device][program][iteration] = data_run_time_MHzCPU[device][program][iteration] \
                                                                    / minimum[program]
show_plot(data_run_time_MHzCPU_norm,
          ylabel="Time compensated for each CPU% and MHz.",
          titel="Time compensated for each CPU% and MHz, Normalised.",
          labels=labels_time,
          log=True, show=True, boxplot=True)

# plotting time/MHz/cpu%/$
for device in range(len(devices)):
    for program in range(len(labels_time)):
        for iteration in range(iterations):
            data_run_time_MHzCPUprice[device][program][iteration] = \
                data_run_time_MHzCPU[device][program][iteration] / device_price[device] * 1000

show_plot(data=data_run_time_MHzCPUprice,
          ylabel="Time compensated for each CPU%, MHz and dollar. (*1000)",
          titel="Time compensated for each CPU%, MHz and dollar for each device.(*1000)",
          labels=labels_time,
          log=True, show=True, boxplot=True)

# plotting normalised time/MHz/cpu%/$
minimum = []
for label in range(len(labels_time)):
    program_values = []
    for device in range(len(devices)):
        program_values.append(mean(data_run_time_MHzCPUprice[device][label]))
    minimum.append(np.nanmin(program_values))

for device in range(len(devices)):
    for program in range(len(labels_time)):
        for iteration in range(iterations):
            data_run_time_MHzCPUprice_norm[device][program][iteration] = data_run_time_MHzCPUprice[device][program][iteration]\
                                                                    / minimum[program]
show_plot(data=data_run_time_MHzCPUprice_norm,
          ylabel="Time compensated for each CPU%, MHz and dollar.",
          titel="Time compensated for each CPU%, MHz and dollar for each device, Normalised.",
          labels=labels_time,
          log=True, show=True, boxplot=True)


def tabel(data):
    table = [["PC"], ["PI"], ["NANO"], ["CORAL"]]

    for program in range(len(programs)):
        for device in range(len(devices)):
            table[device].append(round(mean(data[device][program]), 3))
    print()
    print(tabulate(table,
          headers=["Device", "compair", "friction", "narendra4", "pt2", "P0Y0_narendra4",
                   "P0Y0_compair", "gradient", "FashionMNIST", "NumberMNIST", "catsVSdogs", "Im Rec"]))
    print()


tabel(data_run_time)
tabel(data_run_time_norm)
tabel(data_run_time_MHzCPU)
tabel(data_run_time_MHzCPU_norm)
tabel(data_run_time_MHzCPUprice)
tabel(data_run_time_MHzCPUprice_norm)
