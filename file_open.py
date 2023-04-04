import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import subprocess
from model_prediction import *
from tkinter.ttk import Progressbar
import threading
global output_dir
global filepath

bol = True
bol2 = True

def open_file():

    def load_page():
        fileUploadWindow.destroy()
        from loading_page import loading
        loading()


    def question_page():
        fileUploadWindow.destroy()
        from Question import question_p
        question_p()

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
            output_path = os.path.join(output_dir, 'output.mp4')
            # Check if the output video file exists
            if os.path.exists(output_path):
                # Display the output video in a new window
                subprocess.run(["start", "", output_path], check=True, shell=True)
            else:
                # Display a message indicating that no output video was found
                messagebox.showinfo("Error", "No output video was found in the selected directory.")
        else:
            # Display a message indicating that no output directory has been selected
            messagebox.showinfo("Error", "Please select an output directory before viewing the output video.")


    def processing():
        global output_dir

        # Check if a video file has been uploaded
        filepath_text = fileUploadWindow.nametowidget("filepath_text")
        filepath = filepath_text.get("1.0", "end-1c")

        if not filepath:
            # Display an error message if no video file has been uploaded
            messagebox.showerror("Error", "Please upload a video file before processing.")
            return

        # Create a new window to display the progress bar
        progress_window = tk.Toplevel()
        progress_window.title("Processing!!!!!")
        progress_label = tk.Label(progress_window, text="Processing video...")
        progress_label.pack(pady=10)
        progress_bar = Progressbar(progress_window, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        progress_bar.pack(pady=10)

        # Define a function to run the predict_on_video function in a separate thread
        def run_predict_on_video():
            output_dir_with_filename = os.path.join(output_dir,"output_detect.mp4")  # Combine the output directory and the filename
            output_dir_with_detected = os.path.join(output_dir,"output.mp4")
            predict_on_video(filepath, output_dir_with_filename, output_dir_with_detected, 30)
            progress_window.update() # Update the progress window to ensure it is not stuck
            progress_bar.stop() # Stop the progress bar animation once the processing is complete
            progress_bar.destroy()    
            # Enable closing the progress window
            progress_window.protocol("WM_DELETE_WINDOW", progress_window.destroy)
            fileUploadWindow.protocol("WM_DELETE_WINDOW", progress_window.destroy)

        # Start the progress bar animation
        progress_bar.start()

        # Start a new thread to run the predict_on_video function
        threading.Thread(target=run_predict_on_video).start()
        
        # Define a function to handle closing the progress window
        def on_closing():
            if messagebox.askokcancel("Processing...", "Processing is not complete. Are you sure you want to close the window?"):
                progress_window.destroy()
                fileUploadWindow.destroy()

        # Disable closing the progress window until processing is complete
        progress_window.protocol("WM_DELETE_WINDOW", on_closing)
        fileUploadWindow.protocol("WM_DELETE_WINDOW", on_closing)

        # Block the mainloop until the processing is complete
        progress_window.wait_window(progress_window)
        fileUploadWindow.wait_window(progress_window)


    fileUploadWindow = tk.Tk()

    fileUploadWindow.geometry("1280x663+10+10")
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

    back_button = tk.Button(fileUploadWindow, text="Back", command=load_page,width=10, cursor='hand2', height=2, borderwidth=1, relief="groove", bg="#007FFF", fg="white")
    back_button.place(x=200,y=620)

    next_button = tk.Button(fileUploadWindow, text="Next", command=question_page,width=10, cursor='hand2', height=2, borderwidth=1, relief="groove", bg="#007FFF", fg="white")
    next_button.place(x=990,y=620)

    fileUploadWindow.mainloop()
    fileUploadWindow.destroy(all)