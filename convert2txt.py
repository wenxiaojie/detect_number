import h5py
import cv2 as cv
import glob

img_list = glob.glob("train/*.png")
# img_list.sort()
# print(img_list)
# for img in img_list:
# data = cv.imread(img)
# height, width, channels = data.shape

hdf5_data = h5py.File("digitStruct.mat", 'r')

for index in range(len(img_list)):

    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values

    print(attrs)
    print('train/' + str(index + 1) + '.png')
    data = cv.imread('train/' + str(index + 1) + '.png')

    for i in range(len(attr)):
        
        x_center = attrs['left'][i] + attrs['width'][i] / 2
        print("index:", index + 1, "attrs['left'][i]", attrs['left'][i])
        print("index:", index + 1, "attrs['width'][i]", attrs['width'][i])
        print("index:", index + 1, 'x_center', x_center)
        y_center = attrs['top'][i] + attrs['height'][i] / 2
        
        x_center = x_center / data.shape[1]
        print("index:", index, 'data.shape[1]', data.shape[1])
        y_center = y_center / data.shape[0]
        
        width = attrs['width'][i] / data.shape[1]
        height = attrs['height'][i] / data.shape[0]

        file = open("labels/{}.txt".format(str(index + 1)), 'a')
        file.writelines(str(int(attrs['label'][i]) - 1) + " " + str(x_center) + " " + str(y_center)
                        + " " + str(width) + " " + str(height) + '\n')
        # write (label, x_center, y_center, width, height) to txt
        file.close()
