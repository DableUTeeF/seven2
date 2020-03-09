"""

"""

source = open('/home/palm/PycharmProjects/Seven/stuff/data1-30-9.txt').read().split('\n')[:-1]
clsed = []
open('/home/palm/PycharmProjects/keras-retinanet/datasetstuff/7classes.csv', 'w')
with open('/home/palm/PycharmProjects/keras-retinanet/datasetstuff/data1-30-9.csv', 'w') as wr:
    for s in source:
        x = s.split()
        x1 = min(480, max(0, min(int(x[1]), int(x[2]))))
        x2 = min(480, max(0, max(int(x[1]), int(x[2]))))
        y1 = min(640, max(0, min(int(x[3]), int(x[4]))))
        y2 = min(640, max(0, max(int(x[3]), int(x[4]))))
        cls = x[0].split('/')[-2]
        if abs(x1-x2) < 10 or abs(y1-y2) < 10:
            continue
        wr.write(f'{x[0]},{x1},{y1},{x2},{y2},{cls}\n')
        if cls not in clsed:
            with open('/home/palm/PycharmProjects/keras-retinanet/datasetstuff/7classes.csv', 'a') as wr2:
                wr2.write(f'{cls},{len(clsed)}\n')
            clsed.append(cls)


