'''
data_root:str
record:{"image_id":int,
 "file_name":str,
 "height":int,
 "width":int,
 "annotations":[annotation]
 }
annotation:{
    "image_id": str,
    "category": str,
    "bbox": [x,y,width,height],
}
'''
import os,shutil
import sys
import xml.etree.ElementTree as ET
import json
import pandas as pd

VOC_categories = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                  "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


class Basedataset(object):
    def __init__(self, data_root, categories,trainListFile,testListFile):
        self.data_root = data_root
        self.categories = categories
        self.trainListFile=trainListFile
        self.testListFile=testListFile
        self.__loadTrainIds__()
        self.__loadTestIds__()

    def __loadTrainIds__(self):  # 必须实现，将所有train集id读入到self.train_ids集合里面
        pass

    def __loadTestIds__(self):  # 必须实现，将所有test集id读入到self.test_ids集合里面
        pass

    def getfilebasename(self, image_id):  # 必须实现，对应id的文件名（不含后缀名）
        pass

    def getFilename(self, image_id):  # 必须实现，对应id的文件名（含后缀名）
        pass

    def getImagePath(self, image_id):  # 必须实现，对应id的图片路径
        pass

    def getSingleRecord(self, image_id):  # 必须实现，对应id的单条记录（格式参考注释）
        pass

    def toDarknet(self, target_root):
        categories = self.categories
        ncat = len(categories)
        data_name = os.path.basename(target_root)
        os.makedirs(target_root)
        image_dir = os.path.join(target_root, 'JPEGImages')
        label_dir = os.path.join(target_root, 'labels')
        backup_dir=os.path.join(target_root,'backup')
        os.makedirs(image_dir)
        os.makedirs(label_dir)
        data_path = os.path.join(target_root, "{0}.data".format(data_name))
        name_path = os.path.join(target_root, "{0}.names".format(data_name))
        with open(name_path, 'w') as f1:
            for category in categories:
                f1.write(category+'\n')

        with open(data_path, 'w') as f2:
            f2.write("classes = {0}\n".format(ncat))
            f2.write("train = {0}\n".format(os.path.join(target_root,'train.txt')))
            f2.write("valid = {0}\n".format(os.path.join(target_root,'test.txt')))
            f2.write("names = {0}\n".format(name_path))
            f2.write("backup = {0}\n".format(backup_dir))
        
        def convert(size,bbox):
            dw = 1./(size[0])
            dh = 1./(size[1])
            x = bbox[0]*dw
            w = bbox[2]*dw
            y = bbox[1]*dh
            h = bbox[3]*dh
            return (x,y,w,h)
        
        with open(os.path.join(target_root,'train.txt'),'w') as f4:
            for id in self.train_ids:
                dstpath=os.path.join(image_dir,self.getFilename(id))
                f4.write(dstpath+'\n')
        
        with open(os.path.join(target_root,'test.txt'),'w') as f5:
            for id in self.test_ids:
                dstpath=os.path.join(image_dir,self.getFilename(id))
                f5.write(dstpath+'\n')

        def cpFiles(ids,dirs):
            for id in ids:
                print(id)
                srcFile=self.getImagePath(id)
                dstFile=os.path.join(dirs,self.getFilename(id))
                shutil.copyfile(srcFile,dstFile)
                record=self.getSingleRecord(id)
                w,h=record['width'],record['height']
                with open(os.path.join(label_dir,self.getfilebasename(id)+'.txt'),'w') as f3:
                    for annotation in record['annotations']:
                        cat=annotation['category']
                        bbox=annotation['bbox']
                        cat_id=categories.index(cat)
                        bb=convert((w,h),bbox)
                        f3.write(str(cat_id)+" "+" ".join([str(a) for a in bb])+'\n')
        cpFiles((self.train_ids | self.test_ids),image_dir)

class VOCdataset(Basedataset):
    '''
    输入：数据根目录，类别列表，训练集图片列表，测试集图片列表
    data_root example: data/VOCdevkit/VOC2012
    '''

    def __getIds__(self, file):
        image_ids = file.readlines()
        image_ids = ''.join(image_ids).strip('\n').splitlines()
        image_ids = set([int(id[:4]+id[5:]) for id in image_ids])
        return image_ids

    def __loadTrainIds__(self):
        with open(self.trainListFile, 'r') as f1:
            self.train_ids = self.__getIds__(f1)

    def __loadTestIds__(self):
        with open(self.testListFile, 'r') as f2:
            self.test_ids = self.__getIds__(f2)

    def getfilebasename(self, image_id):  # 必须实现
        return str(image_id)[:4]+'_'+str(image_id)[4:]

    def getFilename(self, image_id):  # 必须实现
        return self.getfilebasename(image_id)+'.jpg'

    def getImagePath(self, image_id):  # 必须实现
        return os.path.join(self.data_root, 'JPEGImages', self.getFilename(image_id))

    def getSingleRecord(self, image_id):  # 必须实现
        record = {'image_id': image_id,
                  'file_name': self.getFilename(image_id)
                  }
        with open(os.path.join(self.data_root, 'Annotations', "{0}.xml".format(self.getfilebasename(image_id))), 'r') as f1:
            tree = ET.parse(f1)
            root = tree.getroot()
            size = root.find('size')
            record['width'] = int(size.find('width').text)
            record['height'] = int(size.find('height').text)
            record['annotations'] = []
            for obj in root.iter('object'):
                xmlbox = obj.find('bndbox')
                x = int(xmlbox.find('xmin').text)
                y = int(xmlbox.find('ymin').text)
                w = int(xmlbox.find('xmax').text)-x
                h = int(xmlbox.find('ymax').text)-y
                record['annotations'].append(
                    {'image_id': image_id, 'category': obj.find('name').text, 'bbox': [x, y, w, h]})
        return record

class COCOdataset(Basedataset):
    '''
    输入：数据集根目录，类别列表，训练集标注文件路径，测试集标注文件路径，训练集图片位置，测试集图片位置
    '''
    def __init__(self,data_root, categories,trainAnnotations,testAnnotations,train_dir,test_dir):
        super(COCOdataset,self).__init__(data_root,categories,trainAnnotations,testAnnotations)
        self.train_dir=train_dir
        self.test_dir=test_dir
        self.fileDict=dict(self.train_dict.items()+self.test_dict.items())
        with open(trainAnnotations,'r') as f1:
            trainjs=json.load(f1)
        with open(testAnnotations,'r') as f2:
            testjs=json.load(f2)
        self.trainandf=pd.DataFrame(trainjs['annotations'])
        self.testandf=pd.DataFrame(testjs['annotations'])


    def __getidsAndfileDict__(self,file):
        ids=set()
        fileDict={}
        with open(file,'r') as f1:
            js1=json.load(f1)
            for image in js1['images']:
                ids.add(image['id'])
                fileDict[id]=image['file_name']
        return ids,fileDict
    
    def __loadTrainIds__(self):
        self.train_ids,self.train_dict=self.__getidsAndfileDict__(self.trainListFile)
    
    def __loadTestIds__(self):
        self.test_ids,self.test_dict=self.__getidsAndfileDict__(self.testListFile)
    
    def getFilename(self,image_id):
        return self.fileDict[image_id]
    
    def getfilebasename(self,image_id):
        return self.getFilename(image_id).split('.')[0]
    
    def getImagePath(self,image_id):
        if image_id in self.train_ids:
            return os.path.join(self.train_dir,self.getFilename(image_id))
        else:
            return os.path.join(self.test_dir,self.getFilename(image_id))

    def getSingleRecord(self, image_id):
        if image_id in self.train_ids:
            pass
        else:
            pass
    #def getImagePath(self,image_id)

    
        
    



if __name__ == "__main__":
    voc = VOCdataset('data/VOCdevkit/VOC2012',VOC_categories,'data/VOCdevkit/VOC2012/ImageSets/Main/train.txt','data/VOCdevkit/VOC2012/ImageSets/Main/val.txt')
    voc.toDarknet('./data/voc_darknet')

# data/VOCdevkit/VOC2012/ImageSets/Main/train.txt
