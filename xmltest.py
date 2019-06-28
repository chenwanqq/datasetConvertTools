from xml.dom import minidom
def writeToXml(filename):
    doc=minidom.getDOMImplementation().createDocument(None,'annotation',None)
    annotation=doc.documentElement
    folder=doc.createElement('folder')
    annotation.appendChild(folder)
    folder_text=doc.createTextNode("VOC2012")
    folder.appendChild(folder_text)
    with open(filename,'w') as f:
        doc.writexml(f,addindent='\t',newl='\n',encoding='utf-8',)

if __name__ == "__main__":
    writeToXml('test.xml')