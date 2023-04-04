import tkinter as tk
from PIL import ImageTk, Image

def loading():
    def start_process():
        loading_window.destroy()
        
        from file_open import open_file
        open_file()


    # create loading window
    loading_window = tk.Tk()
    loading_window.geometry("1280x663+10+10")
    loading_window.resizable(0, 0)
    loading_window.title("Supreme Visualization")

    # add the background image to loading window
    image_bg = Image.open("loading.png")
    photo = ImageTk.PhotoImage(image_bg)
    label_bg = tk.Label(loading_window, image=photo)
    label_bg.place(x=0, y=0, relwidth=1, relheight=1)

    # add the welcome text to main window
    label_welcome = tk.Label(loading_window, text="Welcome to Supreme Visualization",font=("Arial", 24), fg="white", bg="black")
    label_welcome.place(x=370, y=55)

    # add the start button to loading window
    button_start = tk.Button(loading_window, text="START", command=start_process,width=10, height=2, borderwidth=4, relief="groove", bg="#007FFF", fg="white")
    button_start.place(x=600, y=500)

    # show loading window
    loading_window.mainloop()
    loading_window.destroy(all)