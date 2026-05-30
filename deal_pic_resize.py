import os
import cv2
from PIL import Image

# 自制数据集：
# 数据文件夹
data_dir = "./myPic"
object_data_dir = "./pic_object"

def resize_img(img, size):
    iw,ih = img.size
    w,h = size
    scale = min(w/iw,h/ih)                          #获取变形比例
    nw = int(iw*scale)                              #计算变形后的长宽
    nh = int(ih*scale)
    #  旋转
    if(w>h & iw<ih):
        img = img.rotate(0)
    #  变形
    img = img.resize((nw,nh),Image.BICUBIC)
    new_img = Image.new('RGB',size,(255,255,255))   #创建一张白色背景
    new_img.paste(img,((w-nw)//2,(h-nh)//2))        #将变形后的图片贴进背景中央
    return new_img

#----用以控制目标分辨率
sizeX = (224,224)
#--------------------
inFile = "./myPic"
outFile = "./pic_object"
if (os.path.exists(outFile) == False):
    os.mkdir(outFile)        #当文件夹不存在时创建该路径
if(os.path.exists(inFile)):
    fileList = os.listdir(inFile)
    for file in fileList:  # 遍历文件夹中所有文件
        fullFile = inFile +'\\'+ str(file)     #组合完整的输入路径
        f = open(fullFile,'rb')
        img = Image.open(f)                    #打开图片
        newimg = resize_img(img, sizeX)        #图片变形
        print(file)                            #打印变形成功的文件
        newimg.save(os.path.join(outFile,file))#
