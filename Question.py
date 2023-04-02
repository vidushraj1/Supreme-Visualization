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

sRLabel = tk.Label(superRWindow, text="Do you want to create the SR (Super Resolution) \n image of person in intrest",
                   font=("Microsoft Yahei UI Light", 14, 'bold'), bg='SlateBlue4', fg='white')
sRLabel.place(x=400, y=300)

yesButton = tk.Button(superRWindow, text="Yes", font=("Open Sans", 13, 'bold'),
                        fg='black', bg='white', cursor='hand2', bd=0, width=6)
yesButton.place(x=530, y=390)

NoButton = tk.Button(superRWindow, text="No", font=("Open Sans", 13, 'bold'),
                        fg='black', bg='white', cursor='hand2', bd=0, width=6)
NoButton.place(x=690, y=390)

superRWindow.mainloop()