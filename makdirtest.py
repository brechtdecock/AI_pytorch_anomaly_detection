import os

save_dir = "../../../../../THESIS_local/testfolder"

#"C:\\Users\\brech\\Documents\\Thesis_local\\ToyTrain" 
#C:\Users\brech\OneDrive\Documenten\THESIS\VScode\AI_pytorch_anomaly_detection  


def make_dir( dir_name ):  
    if(os.path.isdir(dir_name)==False):
        print("Make directory: "+dir_name)
        os.mkdir(dir_name)  
        print("directory has been made")
        


make_dir(save_dir)    
