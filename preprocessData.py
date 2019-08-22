import cv2
import numpy
import json
import pickle
import random

with open('fn_data.json') as f:
    names = json.load(f)

paintings = {'Vincent_van_Gogh_': 877, 'Edgar_Degas_': 702, 'Pablo_Picasso_': 439, 'Pierre-Auguste_Renoir_': 336, 'Paul_Gauguin_':311, 'Francisco_Goya_':291, 'Rembrandt_':262, 'Alfred_Sisley_':259, 'Titian_':255, 'Marc_Chagall_':239}
labels = {0:'Vincent_van_Gogh_', 1:'Edgar_Degas_', 2:'Pablo_Picasso_', 3:'Pierre-Auguste_Renoir_', 4:'Paul_Gauguin_', 5:'Francisco_Goya_', 6:'Rembrandt_', 7:'Alfred_Sisley_', 8:'Titian_', 9:'Marc_Chagall_'}

def get_train(paintings,labels):
    Xs = []
    Ys = []
    for label in range(4):
        painter = labels[label]
        print(painter)
        for num_paint in names[painter]:
            if int(num_paint) <= paintings[painter]*0.8:
                for n in range(names[painter][num_paint]):
                    path = './resized/resized/cropped/' + painter + num_paint + '_' + str(n) + '.jpg'
                    img = cv2.imread(path)
                    b, g, r = cv2.split(img)
                    print(r.shape)
                    r_p = r.reshape(1, 128*128)
                    g_p = g.reshape(1, 128*128)
                    b_p = b.reshape(1, 128*128)
                    l = [r_p, g_p, b_p]
                    x = numpy.concatenate(l, axis=None).reshape(1, 3*128*128)
                    print(x)
                    print(x.shape)
                    Xs.append(x)
                    Ys.append(label)

    X = numpy.concatenate(Xs, axis=0)
    Y = numpy.array(Ys)
    print(X.shape)
    print(Y.shape)
    numpy.save('X_train.npy', X)
    numpy.save('Y_train.npy', Y)


def get_test(paintings, labels):
    Xs = []
    Ys = []
    paint_nums = []
    for label in range(4):
        painter = labels[label]
        print(painter)
        for num_paint in names[painter]:
            if int(num_paint) > paintings[painter] * 0.8:
                crops = []
                for n in range(names[painter][num_paint]):
                    path = './resized/resized/cropped/' + painter + num_paint + '_' + str(n) + '.jpg'
                    img = cv2.imread(path)
                    b, g, r = cv2.split(img)
                    print(r.shape)
                    r_p = r.reshape(1, 128 * 128)
                    g_p = g.reshape(1, 128 * 128)
                    b_p = b.reshape(1, 128 * 128)
                    l = [r_p, g_p, b_p]
                    x = numpy.concatenate(l, axis=None).reshape(1, 3 * 128 * 128)
                    print(x)
                    print(x.shape)
                    crops.append(x)
                paint = numpy.concatenate(crops, axis=0)
                Xs.append(paint)
                Ys.append(label)
                paint_nums.append(num_paint)
    print(len(Ys),len(Xs),len(num_paint))
    dic = {'labels':Ys, 'data':Xs, 'paint_number':paint_nums}
    with open('test_batch.pkl', 'wb') as f:
        pickle.dump(dic, f)


def get_train_new(paintings, labels):
    Xs = []
    Ys = []
    paint_nums = []
    for label in range(4):
        painter = labels[label]
        print(painter)
        for num_paint in names[painter]:
            if int(num_paint) <= paintings[painter] * 0.8:
                for n in range(names[painter][num_paint]):
                    path = './resized/resized/cropped/' + painter + num_paint + '_' + str(n) + '.jpg'
                    img = cv2.imread(path)
                    b, g, r = cv2.split(img)
                    print(r.shape)
                    r_p = r.reshape(1, 128 * 128)
                    g_p = g.reshape(1, 128 * 128)
                    b_p = b.reshape(1, 128 * 128)
                    x = numpy.concatenate((r_p, g_p, b_p), axis=None).reshape(1, 3 * 128 * 128)
                    print(x)
                    print(x.shape)
                    Xs.append(x)
                    Ys.append(label)
                    paint_nums.append(num_paint)
    print(len(Ys),len(Xs),len(num_paint))
    newX=[]
    newY=[]
    new_paint_nums = []
    idxs = [i for i in range(len(Xs))]
    random.shuffle(idxs)
    for idx in idxs:
        newX.append(Xs[idx])
        newY.append(Ys[idx])
        new_paint_nums.append(paint_nums[idx])
    npX = numpy.concatenate(newX, axis=0)
    npY = numpy.array(newY)
    print(npX.shape,npY.shape,len(new_paint_nums))

    dic = {'labels':npY, 'data':npX, 'paint_number':new_paint_nums}
    with open('train_batch.pkl', 'wb') as f:
        pickle.dump(dic, f)


def get_test_new(paintings, labels):
    Xs = []
    Ys = []
    paint_nums = []
    for label in range(4):
        painter = labels[label]
        print(painter)
        for num_paint in names[painter]:
            if int(num_paint) > paintings[painter] * 0.8:
                for n in range(names[painter][num_paint]):
                    path = './resized/resized/cropped/' + painter + num_paint + '_' + str(n) + '.jpg'
                    img = cv2.imread(path)
                    b, g, r = cv2.split(img)
                    print(r.shape)
                    r_p = r.reshape(1, 128 * 128)
                    g_p = g.reshape(1, 128 * 128)
                    b_p = b.reshape(1, 128 * 128)
                    x = numpy.concatenate((r_p, g_p, b_p), axis=None).reshape(1, 3 * 128 * 128)
                    print(x)
                    print(x.shape)
                    Xs.append(x)
                    Ys.append(label)
                    paint_nums.append(num_paint)
    print(len(Ys),len(Xs),len(num_paint))
    newX=[]
    newY=[]
    new_paint_nums = []
    idxs = [i for i in range(len(Xs))]
    random.shuffle(idxs)
    for idx in idxs:
        newX.append(Xs[idx])
        newY.append(Ys[idx])
        new_paint_nums.append(paint_nums[idx])
    npX = numpy.concatenate(newX, axis=0)
    npY = numpy.array(newY)
    print(npX.shape,npY.shape,len(new_paint_nums))

    dic = {'labels':npY, 'data':npX, 'paint_number':new_paint_nums}
    with open('test_batch_crop128_new.pkl', 'wb') as f:
        pickle.dump(dic, f)


get_test_new(paintings, labels)
