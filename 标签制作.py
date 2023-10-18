import os
import natsort as ns
abc = "D:/shenjingwangluo/output/output3/train"
files_list = os.listdir(abc)
i = 0
j = 0
txt_path = "D:/shenjingwangluo/output/train.txt"
fw = open(txt_path, "w")
while j <= 9:
    picture_path1 = "D:/shenjingwangluo/output/output3/train/"
    picture_path2 = files_list[i]
    picture_path = picture_path1 + picture_path2
    #fw = open(txt_path, "w")                       此处会导致名字覆盖写入
    picture_name = os.listdir(picture_path)
    picture_name = ns.natsorted(picture_name)
    #print(picture_name)
    # 遍历并写入所有文件名
    for pname in picture_name:
        fw.write(pname + ' ' + pname[0:pname.rfind('-', 1)] + '\n')
    i += 1
    j += 1
print('生成txt文件成功')
fw.close()