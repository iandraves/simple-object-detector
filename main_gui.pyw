import os
try:
    from tkinter import filedialog
    from tkinter import messagebox
    from tkinter.font import Font
    from tkinter import *
except ImportError:
    from Tkinter import filedialog
    from Tkinter import messagebox
    from Tkinter.font import Font
    from Tkinter import *
import tensorflow as tf

# Setting up variables
window = Tk()
imgPath = ''
predarray = []
predarray1 = []
file_name = StringVar()
pred1 = StringVar()


def fileslct():
    global file_name
    global pred1
    global imgPath
    global imgLbl
    window.fileName = filedialog.askopenfilename(filetypes=(('JPG', '*.jpg'), ('All files', '*.*')))
    imgPath = window.fileName
    name = os.path.basename(imgPath)
    if name != '':
        file_name.set('Image: ' + name)
    pred1.set('')


def getobject(path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Getting image path
    image_path = path

    try:
        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    except:
        messagebox.showerror("No Image Was Selected!", "No Image Was Selected!")
        return

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            predarray1.append(('%s (score = %.5f)' % (human_string, score)))
            predarray.append(('%s (score = %.5f)' % (human_string, score)).split()[0])
            print('%s (score = %.5f)' % (human_string, score))

    # Showing final prediction
    global pred1
    prediction = predarray[0]
    start = '='
    end = ')'
    result = predarray1[0][predarray1[0].find(start) + len(start):predarray1[0].rfind(end)]
    result = result.replace(" ", "")
    accuracy = float(float(result) * 100)
    pred1.set('I believe it is a picture of (a/an) ' + prediction + ' with a ' + str(round(accuracy, 1)) + '% match.')


def step():
    # Re-setting variables
    global predarray
    global predarray1
    predarray = []
    predarray1 = []

    # Determining the object
    getobject(imgPath)


# Setting up GUI
window.title('Object Detector')
window.geometry('1000x600+1000+600')
window.resizable(width=False, height=False)
try:
    window.iconbitmap('res/logo.ico')
except TclError:
    pass
customFont0 = Font(family="Gadugi", size=13)
customFont = Font(family="Arial", size=18)
pathLbl = Label(window, text="Object Detector", font=customFont).pack()
file = Button(window, text='+ Select Image', command=fileslct, font=customFont0, bg='#00b9fe', activebackground='#22ade1').pack()
space = Label(window).pack()
detect = Button(window, text='Detect Object', command=step, font=customFont0, bg='#2eed6d', activebackground='#23c95a').pack()
filename = Label(window, textvariable=file_name, font=customFont0, fg='blue').pack()
space1 = Label(window).pack()
pred = Label(window, textvariable=pred1, font=customFont0, fg='Green').pack()


# Sustain window
window.mainloop()
