import tkinter as tk
from PIL import ImageTk, Image

superRWindow = tk.Tk()

superRWindow.geometry("1280x700+10+10")
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

superRWindow.mainloop()