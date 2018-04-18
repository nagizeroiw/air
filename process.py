from __future__ import print_function
import json
import time
import datetime

import numpy as np

aq = json.load(open('beijing_aq.json'))

print('> get %d entries' % len(aq))

aq = [x for x in aq if x['station_id'] == 'aotizhongxin']

print('> get %d entries for %s' % (len(aq), 'aotizhongxin'))

def adjust_time(x):
    dt_obj = datetime.datetime.strptime(x['time'], '%Y-%m-%d %H:%M:%S')
    ts = time.mktime(dt_obj.timetuple())
    x['time'] = ts
    return x

aq = [adjust_time(x) for x in aq]
aq = map(dict, set(tuple(sorted(x.items())) for x in aq))
aq = sorted(aq, key=lambda a: a['time'])

old = None
count = 0
best_missing_time = 0
new_aq = []
for x in aq:
    if old is not None:
        try:
            assert x['time'] - old['time'] == 3600.
        except AssertionError:
            missing_time = x['time'] - old['time']
            if missing_time > best_missing_time:
                best_missing_time = missing_time
            # print(old)
            # print(x)
            count += 1
            # print('--------------------------------')
            c_time = old['time'] + 3600
            while c_time < x['time']:
                new_x = x.copy()
                new_x['time'] = c_time
                new_aq.append(new_x)
                c_time += 3600

    old = x
print('> missing data count', count)
print('> best missing time', best_missing_time)

new_aq.extend(aq)
new_aq = sorted(new_aq, key=lambda a: a['time'])
aq = new_aq
print('> finally %d entries for %s' % (len(aq), 'aotizhongxin'))

features = ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2']

count = 0
for i, x in enumerate(aq):
    for feat in features:
        if x[feat] != x[feat]:
            count += 1
            j = i + 1
            while aq[j][feat] != aq[j][feat]:
                j += 1
            x[feat] = aq[j][feat]

print('> NaN feature count', count)

bj_aq = np.zeros((len(aq), 6))
for i, x in enumerate(aq):
    for j, feat in enumerate(features):
        bj_aq[i, j] = aq[i][feat]
np.save(open('bj_aq.npy', 'wb'), bj_aq)
