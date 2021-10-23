### First generate the samples
```
python generate_fake_data.py --number_of_samples_per_font

#ie:

python generate_fake_data.py 100
```
You should get a file as output, when all is finished, called "labels.pkl" This contains text labels and 
bounding boxes.

### Running the program

If you wish to run the program, you would need to supply three things : 
1. Fonts ending in ".ttf" inside the fonts folder.
2. Blank files ending in .png inside the Blanks folder.
3. Text file inside data folder. Default file is kingjamesbible. Please edit text in this file. Otherwise edit scripts/gen_samples.py.



### load_labels.py is an example of loading the saved labels
Please read load_labels.py for more details.
```
python load_labels.py
```
### Example of fonts and blanks
![Image 1](docs.PNG)

### Results from generation (With bounding box)
![Image 1](Results.PNG)

### Comparison with real historical document. (Rightmost is real)
![Image 1](comp.PNG)

