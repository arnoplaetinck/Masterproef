import csv
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

iterations = 20
boxplot_bool = False

name_PC = "Benchmark_PC_Thu_Apr_2_22_05_01_2020"
name_PI = "Benchmark_PI_Fri_Apr_3_02_32_27_2020"
name_NANO = "Benchmark_NANO_Fri_Apr_3_02_11_17_2020"
name_CORAL = "Benchmark_CORAL_Wed_Apr_3_20_08_16_2020"

file_names = [name_PC, name_PI, name_NANO, name_CORAL]

data_CPU = [[], [], [], []]
data_time = [[], [], [], []]
data_CPU_avg = [[], [], [], []]
data_time_avg = [[], [], [], []]

cpu_percent = []
virtual_mem = []
time_diff = []

programs = ["compair", "friction", "narendra4", "pt2", "P0Y0_narendra4",
            "gradient", "FashionMNIST", "NumberMNIST", "catsVSdogs", "Im Rec"]
labels_cpu = programs + ["no operations"]
labels_time = programs + ["Total*"]

ylabel_time = 'Time program execution during running'
ylabel_cpu = 'CPU usage during running'
title_time = 'Time program execution for each device during running'
title_cpu = 'CPU usage for each device during running'

devices = ["PC", "PI", "NANO", "CORAL"]
clockspeed = [3.25, 1.2, 1.479, 1.5]
device_price = [981, 41.5, 99, 149.99]
power = [79.9, 3.7, 5, 2.65]

width = 0.22
image_path = "./images/figures/"


def show_plot(data, ylabel, titel, labels, log, show, index, boxplot, normalise):
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
                data[device][program][iteration] = round(data[device][program][iteration], 5)
    fig, ax = plt.subplots()

    # the label locations
    x = np.arange(len(labels))

    # variables to be used for broken PC normalised line
    xmin = x - 3 * width / 2
    xmax = x + 3 * width / 2
    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    if not normalise:
        rects1 = ax.bar(x - 3 * width / 2, data_bar[1], width, label='PI', color='lightgreen')
        rects2 = ax.bar(x - width / 2, data_bar[2], width, label='NANO', color='limegreen')
        rects3 = ax.bar(x + width / 2, data_bar[3], width, label='CORAL', color='green')
        rects4 = ax.bar(x + 3 * width / 2, data_bar[0], width, label='PC', color='lightblue')
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
        if boxplot:
            ax.boxplot(data[1], positions=x - 3 * width / 2, widths=width, showfliers=True)
            ax.boxplot(data[2], positions=x - width / 2, widths=width, showfliers=True)
            ax.boxplot(data[3], positions=x + width / 2, widths=width, showfliers=True)
            ax.boxplot(data[0], positions=x + 3 * width / 2, widths=width, showfliers=True)
    elif normalise:
        rects2 = ax.bar(x - width, data_bar[1], width, label='PI', color='lightgreen')
        rects3 = ax.bar(x, data_bar[2], width, label='NANO', color='limegreen')
        rects4 = ax.bar(x + width, data_bar[3], width, label='CORAL', color='green')

        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)

        ax.hlines(y=1,
                  xmin=xmin[0],
                  xmax=xmax[-1],
                  colors='r', linestyles='solid', label='PC')
        if boxplot:
            ax.boxplot(data[1], positions=x - width, widths=width)
            ax.boxplot(data[2], positions=x, widths=width)
            ax.boxplot(data[3], positions=x + width, widths=width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(titel, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xticks(rotation=45)
    ax.legend(prop={'size': 20})

    fig.tight_layout()
    plt.grid()
    ax.set_axisbelow(True)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    if log:
        plt.yscale("log")
    plt.savefig(image_path + "Figure_{}".format(index))
    plt.show()


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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data extraction
for device in range(len(file_names)):
    temp1, temp2, = [], []

    with open('./logging/' + file_names[device] + ".csv", mode='r') as results_file:
        results_reader = csv.DictReader(results_file)
        for row in results_reader:
            temp2.append(float(row['timediff']))
            temp1.append(float(row['CPU Percentage']) / 100)

    for program in range(len(programs)):
        time_diff_avg = round(mean(temp2[program * iterations:(program * iterations + iterations)]), 5)
        cpu_avg = round(mean(temp1[program * iterations:(program * iterations + iterations)]), 5)
        data_CPU_avg[device].append(cpu_avg)
        data_time_avg[device].append(time_diff_avg)

        data_CPU[device].append(temp1[program * iterations:(program * iterations + iterations)])
        data_time[device].append(temp2[program * iterations:(program * iterations + iterations)])
    data_CPU[device].append([])
    data_time[device].append([])

    for iteration in range(iterations):
        data_CPU[device][-1].append(temp1[-1])
        data_time[device][-1].append(temp2[-1])
    data_CPU_avg[device].append(temp1[-1])
    data_time_avg[device].append(temp2[-1])

# Adding new total value to PI
total = 0
for program in range(len(programs)-1):
    total += mean(data_time[1][program])
data_time[1][-1] = []
for iteration in range(iterations):
    data_time[1][-1].append(total)

# Plotting figures
show_plot(data_time, ylabel_time, title_time, labels_time,
          log=True, show=True, index=0, boxplot=boxplot_bool, normalise=False)
show_plot(data_CPU, ylabel_cpu, title_cpu, labels_cpu,
          log=False, show=True, index=1, boxplot=boxplot_bool, normalise=False)

# making sure variables have right shape, content of data_time will be ignored
data_time_norm = []
data_energy = []
data_energy_norm = []
data_time_MHzCPU = []
data_time_MHzCPU_norm = []
data_time_MHzCPUprice = []
data_time_MHzCPUprice_norm = []

for device in range(len(devices)):
    data_time_norm.append([])
    data_energy.append([])
    data_energy_norm.append([])
    data_time_MHzCPU.append([])
    data_time_MHzCPU_norm.append([])
    data_time_MHzCPUprice.append([])
    data_time_MHzCPUprice_norm.append([])

    for program in range(len(labels_time)):
        data_time_norm[device].append([])
        data_energy[device].append([])
        data_energy_norm[device].append([])
        data_time_MHzCPU[device].append([])
        data_time_MHzCPU_norm[device].append([])
        data_time_MHzCPUprice[device].append([])
        data_time_MHzCPUprice_norm[device].append([])

        for iteration in range(iterations):
            data_time_norm[device][program].append([])
            data_energy[device][program].append([])
            data_energy_norm[device][program].append([])
            data_time_MHzCPU[device][program].append([])
            data_time_MHzCPU_norm[device][program].append([])
            data_time_MHzCPUprice[device][program].append([])
            data_time_MHzCPUprice_norm[device][program].append([])

# Rescaling to lowest nr of each program
pc_values = []
for program in range(len(labels_time)):
    pc_values.append(mean(data_time[0][program]))
    for device in range(len(devices)):
        for iteration in range(iterations):
            data_time_norm[device][program][iteration] = \
                data_time[device][program][iteration] / pc_values[program]

show_plot(data=data_time_norm,
          ylabel=ylabel_time,
          titel=title_time + ", Normalised",
          labels=labels_time,
          log=True, show=True, index=2,
          boxplot=boxplot_bool, normalise=True)

# plotting time/MHz/cpu%
for device in range(len(devices)):
    for program in range(len(labels_time)):
        for iteration in range(iterations):
            data_time_MHzCPU[device][program][iteration] = \
                data_time[device][program][iteration] / clockspeed[device] / data_CPU_avg[device][program]

show_plot(data=data_time_MHzCPU,
          ylabel="Time / CPU% / MHz.",
          titel="Time / CPU% / MHz for each device.",
          labels=labels_time,
          log=True, show=True, index=3,
          boxplot=boxplot_bool, normalise=False)

# plotting normalised time/MHz/cpu%
pc_values = []
for label in range(len(labels_time)):
    pc_values.append(mean(data_time_MHzCPU[0][label]))

for device in range(len(devices)):
    for program in range(len(labels_time)):
        for iteration in range(iterations):
            data_time_MHzCPU_norm[device][program][iteration] = data_time_MHzCPU[device][program][iteration] \
                                                                    / pc_values[program]
show_plot(data_time_MHzCPU_norm,
          ylabel="Time / CPU% / MHz.",
          titel="Time / CPU% / MHz for each device, Normalised.",
          labels=labels_time,
          log=True, show=True, index=4,
          boxplot=boxplot_bool, normalise=True)

# plotting time*watt
for device in range(len(devices)):
    for program in range(len(labels_time)):
        for iteration in range(iterations):
            data_energy[device][program][iteration] = \
                power[device] * data_time[device][program][iteration]


show_plot(data=data_energy,
          ylabel="Time * Watt.",
          titel="Time * Watt for each device.",
          labels=labels_time,
          log=True, show=True, index=5,
          boxplot=boxplot_bool, normalise=False)

# plotting normalised time/watt
pc_values = []
for label in range(len(labels_time)):
    pc_values.append(mean(data_energy[0][label]))

for device in range(len(devices)):
    for program in range(len(labels_time)):
        for iteration in range(iterations):
            data_energy_norm[device][program][iteration] = data_energy[device][program][iteration] \
                                                                    / pc_values[program]
show_plot(data_energy_norm,
          ylabel="Time * Watt.",
          titel="Time * Watt, Normalised.",
          labels=labels_time,
          log=True, show=True, index=6,
          boxplot=boxplot_bool, normalise=True)

# plotting time/$
for device in range(len(devices)):
    for program in range(len(labels_time)):
        for iteration in range(iterations):
            data_time_MHzCPUprice[device][program][iteration] = \
                device_price[device] / data_time[device][program][iteration]

show_plot(data=data_time_MHzCPUprice,
          ylabel="Time * dollar.",
          titel="Time * dollar for each device.",
          labels=labels_time,
          log=True, show=True, index=7,
          boxplot=boxplot_bool, normalise=False)

# plotting normalised time/MHz/cpu%/$
pc_values = []
for label in range(len(labels_time)):
    pc_values.append(mean(data_time_MHzCPUprice[0][label]))

for device in range(len(devices)):
    for program in range(len(labels_time)):
        for iteration in range(iterations):
            data_time_MHzCPUprice_norm[device][program][iteration] = \
                data_time_MHzCPUprice[device][program][iteration] / pc_values[program]
show_plot(data=data_time_MHzCPUprice_norm,
          ylabel="Time * dollar.",
          titel="Time * dollar for each device, Normalised.",
          labels=labels_time,
          log=True, show=True, index=8,
          boxplot=boxplot_bool, normalise=True)

'''
tabel(data_time)
tabel(data_time_norm)
tabel(data_time_MHzCPU)
tabel(data_time_MHzCPU_norm)
tabel(data_time_MHzCPUprice)
tabel(data_time_MHzCPUprice_norm)
'''

