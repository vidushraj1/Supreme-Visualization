import tkinter as tk
from PIL import ImageTk, Image

def SR_Process():

    def Q_p():
        superRWindow.destroy()
            
        from Question import question_p
        question_p()

    def exit():
    
        superRWindow.destroy()


    superRWindow = tk.Tk()

    superRWindow.geometry("1280x663+10+10")
    superRWindow.resizable(0, 0)
    superRWindow.title("Supreme Visualization")

    imageBg = Image.open("login.png")
    photo = ImageTk.PhotoImage(imageBg)

    labelBg = tk.Label(superRWindow, image=photo)
    labelBg.place(x=0, y=0)

    yesButton = tk.Button(superRWindow, text="Process the Image", font=("Open Sans", 13, 'bold'),
                            fg='black', bg='white', cursor='hand2', bd=0, width=20)
    yesButton.place(x=530, y=320)

    NoButton = tk.Button(superRWindow, text="View the Image", font=("Open Sans", 13, 'bold'),
                            fg='black', bg='white', cursor='hand2', bd=0, width=20)
    NoButton.place(x=530, y=420)

    back_button = tk.Button(superRWindow, text="Back", command=Q_p,width=10, cursor='hand2', height=2, borderwidth=1, relief="groove", bg="#007FFF", fg="white")
    back_button.place(x=200,y=620)

    next_button = tk.Button(superRWindow, text="Exit", command=exit,width=10, cursor='hand2', height=2, borderwidth=1, relief="groove", bg="#007FFF", fg="white")
    next_button.place(x=990,y=620)

    superRWindow.mainloop()
    superRWindow.destroy(all)
#Deep learning SRGAN required