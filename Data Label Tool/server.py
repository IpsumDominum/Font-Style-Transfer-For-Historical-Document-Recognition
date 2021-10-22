from flask import render_template
from flask import Flask
import pickle
import os
from flask import request,jsonify,make_response,redirect,url_for
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

if(os.path.isfile("checked.pkl")):
    with open('checked.pkl', 'rb') as f:
        checked = pickle.load(f)
else:
    checked = {}


"""
For all the files in Data, if the Data name is not in checked keys
then present the file, each time someone finishes labeling, the key
is appended to checked, each of the checked files would have the following:
filename = "xxx.jpg"
dataset = "manuscriptonline"
list_of_rectangles = []
"""
directory = os.path.join("static")
"""
@app.route('/')
def whichone():
    return render_template("whichOneAreYou.html")
"""
@app.route('/',methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        #Check for the next piece of data which haven't been labeled...
        filename = request.args.get('filename')
        redirection = request.args.get('redirect')  
        flag = ""
        if(filename!=None):
            #filename specified...            
            try:
                item = checked[filename]            
                if(redirection==None):
                    pass
                else:                                                
                    if(redirection=="next"):                    
                        if(item['index']+1 < len(checked.keys())):
                            new_index = item['index']+1
                        else:
                            new_index = item['index']
                            flag = "exceeded"
                        filename = list(checked.keys())[new_index]
                    elif(redirection=="back"):
                        if(item['index']-1 >= 0):
                            new_index = item['index']-1
                        else:
                            new_index = item['index']                    
                            flag = "first_item"
                        filename = list(checked.keys())[new_index]
            except KeyError:
                filename = None            
            
        if(filename==None):
            #Find the first item that hasen't been labeled...
            for c in checked.keys():
                if(len(checked[c]["rects"])==0 and checked[c]["invalid"]==False):
                    filename = c
                    break
        if(filename==None):
            filename = list(checked.keys())[0]         
        return render_template('index.html',
                filename=filename,
                current_index=checked[filename]["index"],
                total_images=len(checked.keys())-1,
                rects=checked[filename]["rects"],
                invalid=checked[filename]["invalid"]
                )
    else:
        data = request.get_json()        
        filename = data["filename"]
        checked[filename]["rects"] = data["rects"]        
        checked[filename]["invalid"] = data["invalid"]
        if data["invalid"]==True:
            return make_response(jsonify(
                    {"redirect": "/?filename={}&redirect=next".format(filename)}
                ),200)
        with open('checked.pkl', 'wb') as f:
            pickle.dump(checked, f)
        return make_response(jsonify(
                {"redirect": "."}
            ),200)

if __name__=="__main__":
    app.run()