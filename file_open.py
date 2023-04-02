import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog
import os
from model_prediction import *
global output_dir
global filepath

def upload_video():
    global filepath
    # Ask the user to select the input video file
    filepath = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4;*.avi;*.mkv"), ("All files", "*.*")]
    )

    # Get the Text widget by its ID
    filepath_text = fileUploadWindow.nametowidget("filepath_text")
    # Insert the filepath into the Text widget
    filepath_text.insert("1.0", filepath)

    # Process the video file and save the output to the selected output directory
    # ...
    return filepath


def select_output_directory():
    global output_dir
    # Ask the user to select the output directory
    output_dir = filedialog.askdirectory(
        title="Select the output directory"
    )
    outputpath_text = fileUploadWindow.nametowidget("outputpath_text")
    # Insert the filepath into the Text widget
    outputpath_text.insert("1.0", output_dir)

    # Do something with the selected output directory, such as displaying it to the user
    # ...
    return output_dir


def view_output():
    global output_dir
    # Check if an output directory has been selected
    if output_dir:
        # Get the path of the output video file
        output_path = os.path.join(output_dir,"output.mp4")
        # Check if the output video file exists
        if os.path.exists(output_path):
            # Display the output video in a new window
            # TODO: Add code to display the video
            pass
        else:
            # Display a message indicating that no output video was found
            messagebox.showinfo("Error", "No output video was found in the selected directory.")
    else:
        # Display a message indicating that no output directory has been selected
        messagebox.showinfo("Error", "Please select an output directory before viewing the output video.")


import os

def processing():
    global output_dir
    # Check if a video file has been uploaded
    filepath_text = fileUploadWindow.nametowidget("filepath_text")
    filepath = filepath_text.get("1.0", "end-1c")  # Get the text of the filepath_text widget
    if not filepath:
        # Display an error message if no video file has been uploaded
        messagebox.showinfo("Error", "Please upload a video file before processing.")
        return
    else:
        output_dir_with_filename = os.path.join(output_dir,"output_detect.mp4")  # Combine the output directory and the filename
        output_dir_with_dectected = os.path.join(output_dir,"output.mp4")
        predict_on_video(filepath, output_dir_with_filename, output_dir_with_dectected, 30)

fileUploadWindow = tk.Tk()

fileUploadWindow.geometry("1280x700+10+10")
fileUploadWindow.resizable(0, 0)
fileUploadWindow.title("Supreme Visualization")

imageBg = Image.open("login.png")
photo = ImageTk.PhotoImage(imageBg)

labelBg = tk.Label(fileUploadWindow, image=photo)
labelBg.place(x=0, y=0)

filepath_label = tk.Label(fileUploadWindow, text="Selected file:", font=("Open Sans", 12, 'bold'), fg='white',
                          bg='#222')
filepath_label.place(x=200, y=310)

filepath_text = tk.Text(fileUploadWindow, name="filepath_text", font=("Open Sans", 12), width=50, height=1, bd=0,
                        bg='#ddd')
filepath_text.place(x=320, y=310)

fileOpenButton = tk.Button(fileUploadWindow, text="Upload Video", font=("Open Sans", 16, 'bold'),
                           fg='black', bg='white', cursor='hand2', bd=0, width=19, command=upload_video)
fileOpenButton.place(x=850, y=300)

outputpath_label = tk.Label(fileUploadWindow, text="Selected file:", font=("Open Sans", 12, 'bold'), fg='white',
                          bg='#222')
outputpath_label.place(x=200, y=410)

outputpath_text = tk.Text(fileUploadWindow, name="outputpath_text", font=("Open Sans", 12), width=50, height=1, bd=0,
                        bg='#ddd')
outputpath_text.place(x=320, y=410)


outputDirButton = tk.Button(fileUploadWindow, text="Select Output Directory", font=("Open Sans", 16, 'bold'),
                            fg='black', bg='white', cursor='hand2', bd=0, width=19, command=select_output_directory)
outputDirButton.place(x=850, y=400)


outputButton = tk.Button(fileUploadWindow, text="View Output", font=("Open Sans", 16, 'bold'),
                         fg='black', bg='white', cursor='hand2', bd=0, width=19, command=view_output)
outputButton.place(x=520, y=570)


start_button = tk.Button(fileUploadWindow, text="Start Processing", font=("Open Sans", 16, 'bold'),
                         fg='black', bg='white', cursor='hand2', bd=0, width=19, command=processing)
start_button.place(x=520, y=500)



fileUploadWindow.mainloop()