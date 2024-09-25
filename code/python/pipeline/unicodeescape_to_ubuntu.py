# from uni (windows) to ubuntu
def uni_to_ubuntu(uni): 
    return(uni.replace("\%", "/").replace("C:/%", "/mnt/c/"))
    
# from ubunto to uni (windows)
def ubuntu_to_uni(uni): 
    return(uni.replace("/mnt/c/", "C:/"))

print(uni_to_ubuntu("C:\Users\Johan\Documents\ITligence\data\uncleaned_data\JerryWeiAIData\train_orig.csv"))