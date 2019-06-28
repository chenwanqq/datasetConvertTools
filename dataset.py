'''
data_root:str
record:{"image_id":int,
 "file_name":str,
 "height":int,
 "width":int,
 "annotations":[annotation]
 }
annotation:{
    "image_id": int,
    "category": str,
    "bbox": [x,y,width,height],
}
'''
import os
import shutil
import sys
import xml.etree.ElementTree as ET
import json
import pandas as pd
from xml.dom import minidom

VOC_categories = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                  "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


class Basedataset(object):

    def __init__(self, data_root, trainListFile, testListFile, categories=None):
        self.data_root = data_root
        self.trainListFile = trainListFile
        self.testListFile = testListFile
        self.__loadTrainIds__()
        self.__loadTestIds__()
        self.__loadCategories__(categories)

    # 必须实现，把类别加载为列表形式到self.categories里面，categories可以为None，代表从数据集标签中找到类别信息
    def __loadCategories__(self, categories):
        pass

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

    def __singleSetCOCO__(self, target_root, set_name, setIds, annoidStart=1):
        resultjs = {'info': {
            'description': '{0} Dataset'.format(os.path.basename(target_root))
        }, 'licenses': [], 'images': [], 'annotations': [], 'categories': []}
        os.makedirs(os.path.join(target_root, set_name))
        annoId = annoidStart
        idCount = 0
        amount = len(setIds)
        for id in setIds:
            idCount += 1
            if idCount % 500 == 0:
                print("{0} {1}/{2}".format(set_name, idCount, amount))
            record = self.getSingleRecord(id)
            resultjs['images'].append({
                'file_name': record['file_name'],
                'height': record['height'],
                'width': record['width'],
                'id': id,
            })
            for annotation in record['annotations']:
                resultjs['annotations'].append({
                    'image_id': id,
                    'bbox': annotation['bbox'],
                    'category_id': self.categories.index(annotation['category'])+1,
                    'id': annoId,
                    'segmentation': [[]],
                    'iscrowd': 0,
                    'area': 0,
                })
                annoId += 1
            shutil.copyfile(self.getImagePath(id), os.path.join(
                target_root, set_name, record['file_name']))
        for i in range(len(self.categories)):
            resultjs['categories'].append({
                'supercategory': 'object',
                'id': i+1,
                'name': self.categories[i]
            })
        with open(os.path.join(target_root, 'annotations', '{0}.json'.format(set_name)), 'w') as f1:
            json.dump(resultjs, f1)
        return annoId

    def toCOCO(self, target_root):
        os.makedirs(target_root)
        os.makedirs(os.path.join(target_root, 'annotations'))
        nextId = self.__singleSetCOCO__(
            target_root, 'train', self.train_ids, 1)
        self.__singleSetCOCO__(target_root, 'test', self.test_ids, nextId)

    def toVOC(self, target_root):
        image_folder = os.path.join(target_root, 'JPEGImages')
        label_folder = os.path.join(target_root, 'Annotations')
        set_folder = os.path.join(target_root, 'ImageSets', 'Main')
        os.makedirs(image_folder)
        os.makedirs(label_folder)
        os.makedirs(set_folder)
        with open(os.path.join(set_folder, 'train.txt'), 'w') as f1:
            for id in self.train_ids:
                f1.write(str(self.getfilebasename(id))+'\n')
        with open(os.path.join(set_folder, 'test.txt'), 'w') as f2:
            for id in self.test_ids:
                f2.write(str(self.getfilebasename(id))+'\n')
        if type(self.train_ids) == set:
            ids = (self.train_ids | self.test_ids)
        elif type(self.train_ids) == list:
            ids = (self.train_ids+self.test_ids)
        idCount=0
        amount=len(ids)
        for id in ids:
            idCount+=1
            if idCount%500==0:
                print("{0}/{1}".format(idCount,amount))
            record=self.getSingleRecord(id)
            doc=minidom.getDOMImplementation().createDocument(None,'annotation',None)
            annotation=doc.documentElement

            def textElement(name,s,dom):
                res=dom.createElement(name)
                res.appendChild(dom.createTextNode(s))
                return res

            annotation.appendChild(textElement('folder',os.path.basename(target_root),doc))
            annotation.appendChild(textElement('filename',self.getFilename(id),doc))
            
            size=doc.createElement('size')
            annotation.appendChild(size)
            
            size.appendChild(textElement('width',str(int(record['width'])),doc))
            size.appendChild(textElement('height',str(int(record['height'])),doc))
            size.appendChild(textElement('depth',str(int(3)),doc))

            annotation.appendChild(textElement('segmented',str(0),doc))

            
            for anno in record['annotations']:
                obj=doc.createElement('object')
                annotation.appendChild(obj)
                obj.appendChild(textElement('name',anno['category'],doc))
                obj.appendChild(textElement('pose','Unspecified',doc))
                obj.appendChild(textElement('truncated',str(0),doc))
                obj.appendChild(textElement('difficult',str(0),doc))
                bndbox=doc.createElement('bndbox')
                obj.appendChild(bndbox)
                bndbox.appendChild(textElement('xmin',str(int(anno['bbox'][0])),doc))
                bndbox.appendChild(textElement('ymin',str(int(anno['bbox'][1])),doc))
                bndbox.appendChild(textElement('xmax',str(int(anno['bbox'][0]+anno['bbox'][2])),doc))
                bndbox.appendChild(textElement('ymax',str(int(anno['bbox'][1]+anno['bbox'][3])),doc))
            with open(os.path.join(label_folder,"{0}.xml".format(self.getfilebasename(id))),'w') as f:
                doc.writexml(f,addindent='\t',newl='\n',encoding='utf-8',)
            shutil.copyfile(self.getImagePath(id),os.path.join(image_folder,self.getFilename(id)))

    def toDarknet(self, target_root):
        categories = self.categories
        ncat = len(categories)
        data_name = os.path.basename(target_root)
        os.makedirs(target_root)
        image_dir = os.path.join(target_root, 'JPEGImages')
        label_dir = os.path.join(target_root, 'labels')
        backup_dir = os.path.join(target_root, 'backup')
        os.makedirs(image_dir)
        os.makedirs(label_dir)
        data_path = os.path.join(target_root, "{0}.data".format(data_name))
        name_path = os.path.join(target_root, "{0}.names".format(data_name))
        with open(name_path, 'w') as f1:
            for category in categories:
                f1.write(category+'\n')

        with open(data_path, 'w') as f2:
            f2.write("classes = {0}\n".format(ncat))
            f2.write("train = {0}\n".format(
                os.path.join(target_root, 'train.txt')))
            f2.write("valid = {0}\n".format(
                os.path.join(target_root, 'test.txt')))
            f2.write("names = {0}\n".format(name_path))
            f2.write("backup = {0}\n".format(backup_dir))

        def convert(size, bbox):
            dw = 1./(size[0])
            dh = 1./(size[1])
            x = bbox[0]*dw
            w = bbox[2]*dw
            y = bbox[1]*dh
            h = bbox[3]*dh
            return (x, y, w, h)

        with open(os.path.join(target_root, 'train.txt'), 'w') as f4:
            for id in self.train_ids:
                dstpath = os.path.join(image_dir, self.getFilename(id))
                f4.write(dstpath+'\n')

        with open(os.path.join(target_root, 'test.txt'), 'w') as f5:
            for id in self.test_ids:
                dstpath = os.path.join(image_dir, self.getFilename(id))
                f5.write(dstpath+'\n')

        def cpFiles(ids, dirs):
            amount = len(ids)
            count = 0
            for id in ids:
                count += 1
                if count % 500 == 0:
                    print("{0}/{1}".format(count, amount))
                srcFile = self.getImagePath(id)
                dstFile = os.path.join(dirs, self.getFilename(id))
                shutil.copyfile(srcFile, dstFile)
                record = self.getSingleRecord(id)
                w, h = record['width'], record['height']
                with open(os.path.join(label_dir, self.getfilebasename(id)+'.txt'), 'w') as f3:
                    for annotation in record['annotations']:
                        cat = annotation['category']
                        bbox = annotation['bbox']
                        cat_id = categories.index(cat)
                        bb = convert((w, h), bbox)
                        f3.write(str(cat_id)+" " +
                                 " ".join([str(a) for a in bb])+'\n')
        if type(self.train_ids) == set:
            cpFiles((self.train_ids | self.test_ids), image_dir)
        elif type(self.train_ids) == list:
            cpFiles((self.train_ids+self.test_ids), image_dir)


class VOCdataset(Basedataset):
    '''
    输入：数据根目录，类别列表，训练集图片列表，测试集图片列表
    data_root example: data/VOCdevkit/VOC2012
    VOC读取文件名功能还不完善，如有需要请自行复写getfilebasename与getFilename两个方法
    '''

    def __init__(self, data_root, trainListFile, testListFile, categories):
        super(VOCdataset, self).__init__(
            data_root, trainListFile, testListFile, categories)

    def __loadCategories__(self, categories):
        self.categories = categories

    def __getIds__(self, file):
        image_ids = file.readlines()
        image_ids = ''.join(image_ids).strip('\n').splitlines()
        image_ids = [int(id[:4]+id[5:]) for id in image_ids]
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
    输入：数据集根目录,训练集标注文件路径，测试集标注文件路径，训练集图片位置，测试集图片位置
    '''

    def __init__(self, data_root, trainAnnotations, testAnnotations, train_dir, test_dir):
        super(COCOdataset, self).__init__(
            data_root, trainAnnotations, testAnnotations, testAnnotations)
        print("finish 1")
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.fileDict = {**self.train_dict, **self.test_dict}
        self.isTrain = {**self.trainFlag, **self.testFlag}
        with open(trainAnnotations, 'r') as f1:
            trainjs = json.load(f1)
        with open(testAnnotations, 'r') as f2:
            testjs = json.load(f2)
        traindf = pd.DataFrame(trainjs['annotations'])
        trainimdf = pd.DataFrame(trainjs['images'])
        testdf = pd.DataFrame(testjs['annotations'])
        testimdf = pd.DataFrame(testjs['images'])
        self.df = pd.concat([traindf, testdf])
        self.imdf = pd.concat([trainimdf, testimdf])

    def __loadCategories__(self, categories):  # categories此处为标注文件路径信息，内含标签信息
        self.categories = []
        self.cocoCatMap = {}  # coco数据集category_id的映射关系
        with open(categories, 'r') as f1:
            js1 = json.load(f1)
            for cat in js1['categories']:
                self.categories.append(cat['name'])
                cocoCatId = cat['id']
                cocoCatName = cat['name']
                self.cocoCatMap[cocoCatId] = cocoCatName

    def __getidsAndfileDict__(self, file, trainFlag):
        ids = set()
        fileDict = {}
        isTrain = {}
        with open(file, 'r') as f1:
            js1 = json.load(f1)
            for image in js1['images']:
                ids.add(image['id'])
                fileDict[image['id']] = image['file_name']
                isTrain[image['id']] = trainFlag
        return ids, fileDict, isTrain

    def __loadTrainIds__(self):
        self.train_ids, self.train_dict, self.trainFlag = self.__getidsAndfileDict__(
            self.trainListFile, 1)

    def __loadTestIds__(self):
        self.test_ids, self.test_dict, self.testFlag = self.__getidsAndfileDict__(
            self.testListFile, 0)

    def getFilename(self, image_id):
        return self.fileDict[image_id]

    def getfilebasename(self, image_id):
        return self.getFilename(image_id).split('.')[0]

    def getImagePath(self, image_id):
        if self.isTrain[image_id] == 1:
            return os.path.join(self.train_dir, self.getFilename(image_id))
        else:
            return os.path.join(self.test_dir, self.getFilename(image_id))

    def getSingleRecord(self, image_id):
        record = {'image_id': image_id,
                  'file_name': self.getFilename(image_id),
                  'annotations': []
                  }
        for index, row in self.imdf[self.imdf.id == image_id].iterrows():
            record['height'] = row['height']
            record['width'] = row['width']
        for index, row in self.df[self.df.image_id == image_id].iterrows():
            record['annotations'].append({
                'image_id': image_id,
                'category': self.cocoCatMap[row['category_id']],
                'bbox': row['bbox']
            })
        # print(record)
        return record

    # def getImagePath(self,image_id)


if __name__ == "__main__":
    voc = VOCdataset('data/VOCdevkit/VOC2012', 'data/VOCdevkit/VOC2012/ImageSets/Main/train.txt',
                     'data/VOCdevkit/VOC2012/ImageSets/Main/val.txt', VOC_categories)
    voc.toVOC('data/voc_voc')
    #voc.toCOCO('data/voc_coco')
    # voc.toDarknet('./data/voc_darknet')
    # coco=COCOdataset('data/coco','data/coco/annotations/instances_train2017.json','data/coco/annotations/instances_val2017.json','data/coco/train2017','data/coco/val2017')
    # coco.toDarknet('./data/coco_darknet')

# data/VOCdevkit/VOC2012/ImageSets/Main/t rain.txt
