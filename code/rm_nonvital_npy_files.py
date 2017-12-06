import sys, os

def remove_nonvital_npy_files(directory) :
    for t3 in os.walk(directory):
        if t3[0][1:].find("./") == -1 and t3[0]!=directory:
            subdir = t3[0]
            if subdir.find("savedata_") != -1:
                imgfiles = [i for i in os.listdir(subdir) if i.endswith('.npy')]
                imgfiles.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f)))))
                pref_file_epoch = -1
                for file in imgfiles:
                    file_epoch = int(file.split("_")[0])
                    if file_epoch == pref_file_epoch:
                        os.remove(os.path.join(subdir, prevfile))
                    prevfile = file
                    pref_file_epoch = file_epoch
            else:
                remove_nonvital_npy_files(subdir)


remove_nonvital_npy_files(".")
