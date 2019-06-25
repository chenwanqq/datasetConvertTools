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
import os,sys
import xml.etree.ElementTree as ET

VOC_categories=["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
class Basedataset(object):
    def __init__(self,data_root):
        self.data_root=data_root
        self.__loadTrainIds__()
        self.__loadTestIds__()

    def __loadTrainIds__(self): #必须实现，将所有train集id读入到self.train_ids里面
        pass

    def __loadTestIds__(self): #必须实现，将所有test集id读入到self.test_ids里面
        pass

    def getfilebasename(self,image_id): #必须实现，对应id的文件名（不含后缀名）
        pass
    
    def getFilename(self,image_id): #必须实现，对应id的文件名（含后缀名）
        pass

    def getImagePath(self,image_id): #必须实现，对应id的图片路径
        pass
    
    def getSingleRecord(self,image_id): #必须实现，对应id的单条记录（格式参考注释）
        pass
    
    def toDarknet(self,target_root,categories):
        data_name=os.path.basename(target_root)
        os.makedirs(target_root)
        train_dir=os.path.join(target_root,'trainImages')
        test_dir=os.path.join(target_root,'testImages')
        data_path=os.path.join(target_root,"{0}.data".format(data_name))
        name_path=os.path.join(target_root,"{0}.names".format(data_name))
        
        
        
        





class VOCdataset(Basedataset):
    '''
    data_root example: data/VOCdevkit/VOC2012
    '''
    def __getIds__(self,file):
        image_ids=file.readlines()
        image_ids=''.join(image_ids).strip('\n').splitlines()
        image_ids=[int(id[:4]+id[5:]) for id in image_ids]
        return image_ids
    
    def __loadTrainIds__(self):
        with open(os.path.join(self.data_root,'ImageSets','Main','train.txt'),'r') as f1:
            self.train_ids=self.__getIds__(f1)
    
    def __loadTestIds__(self):
        with open(os.path.join(self.data_root,'ImageSets','Main','val.txt'),'r') as f2:
            self.test_ids=self.__getIds__(f2)    


    def getfilebasename(self,image_id): #必须实现
        return str(image_id)[:4]+'_'+str(image_id)[4:]
    def getFilename(self,image_id):  #必须实现
        return self.getfilebasename(image_id)+'.jpg'

    def getImagePath(self,image_id): #必须实现
        return os.path.join(self.data_root,'JPEGImages',self.getFilename(image_id)) 
    
    def getSingleRecord(self,image_id): #必须实现
        record={'image_id':image_id,
                'file_name':self.getFilename(image_id)
                }
        with open(os.path.join(self.data_root,'Annotations',"{0}.xml".format(self.getfilebasename(image_id))),'r') as f1:
            tree=ET.parse(f1)
            root=tree.getroot()
            size=root.find('size')
            record['width']=int(size.find('width').text)
            record['height']=int(size.find('height').text)
            record['annotations']=[]
            for obj in root.iter('object'):
                xmlbox=obj.find('bndbox')
                x=int(xmlbox.find('xmin').text)
                y=int(xmlbox.find('ymin').text)
                w=int(xmlbox.find('xmax').text)-x
                h=int(xmlbox.find('ymax').text)-y
                record['annotations'].append({'image_id':image_id,'category':obj.find('name').text,'bbox':[x,y,w,h]})
        return record
        
if __name__ == "__main__":
    voc=VOCdataset('data/VOCdevkit/VOC2012')
    print(voc.getSingleRecord(2007000033))

#data/VOCdevkit/VOC2012/ImageSets/Main/train.txt