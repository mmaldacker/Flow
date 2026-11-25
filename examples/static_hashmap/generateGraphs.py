import csv
import pygal
from collections import defaultdict
from pygal.style import Style
from pygal.style import LightSolarizedStyle

custom_style = Style(background='#f5f4f0')

hashmapChart = pygal.Line(logarithmic=False, title='Hash Map insert timing', y_title='Time (ms)', x_title='Load', style=custom_style, show_y_guides=False)

hashmapDict = defaultdict(list)

sizes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

with open('timing.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        name = row['name']
        hashmapDict[name].append(row['time'])

for name, values in hashmapDict.items():
    hashmapChart.add(name, [float(value) for value in values])

hashmapChart.x_labels = sizes
hashmapChart.render_to_file('hashmap.svg')