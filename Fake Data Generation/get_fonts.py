import os
import shutil
for root, dirs, files in os.walk("./fonts_zip", topdown=False):
   for name in files:
        if(name.endswith(".ttf")):
            shutil.copy(os.path.join(root,name),os.path.join("fonts",name))