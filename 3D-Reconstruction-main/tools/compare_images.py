#!/usr/bin/env python3
# Compare images from two folders side by side with navigation
import os
import sys
from tkinter import Tk, Label, Button, Frame, filedialog, Canvas
from PIL import Image, ImageTk
from tqdm import tqdm

def get_image_files(folder):
	valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
	files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_exts]
	files.sort()
	return files

class ImageComparer:
	def __init__(self, folder1, folder2):
		self.folder1 = folder1
		self.folder2 = folder2
		self.files1 = get_image_files(folder1)
		self.files2 = get_image_files(folder2)
		self.max_len = max(len(self.files1), len(self.files2))
		self.idx = 0

		# Preload all images into memory
		self.images1 = self.preload_images(self.folder1, self.files1)
		self.images2 = self.preload_images(self.folder2, self.files2)

		self.root = Tk()
		self.root.title("Compare Images Side by Side")
		self.canvas_width = 400
		self.canvas_height = 400
		self.frame = Frame(self.root)
		self.frame.pack(fill='both', expand=True)
		self.canvas1 = Canvas(self.frame, width=self.canvas_width, height=self.canvas_height, bg='black')
		self.canvas2 = Canvas(self.frame, width=self.canvas_width, height=self.canvas_height, bg='black')
		self.canvas1.grid(row=0, column=0, sticky='nsew')
		self.canvas2.grid(row=0, column=1, sticky='nsew')
		self.frame.columnconfigure(0, weight=1)
		self.frame.columnconfigure(1, weight=1)
		self.frame.rowconfigure(0, weight=1)

		# Bind resize event
		self.root.bind('<Configure>', self.on_resize)
		self.label1 = Label(self.frame, text="")
		self.label2 = Label(self.frame, text="")
		self.label1.grid(row=1, column=0)
		self.label2.grid(row=1, column=1)
		# Only one button for left and one for right
		self.left_idx = 0
		self.right_idx = 0
		self.btn_left = Button(self.root, text="Left", command=self.prev_img)
		self.btn_right = Button(self.root, text="Right", command=self.next_img)
		self.btn_left.pack(side='left', padx=20, pady=10)
		self.btn_right.pack(side='right', padx=20, pady=10)

		# Keyboard support: left/right arrows move both images
		self.root.bind('<Left>', lambda event: self.prev_img())
		self.root.bind('<Right>', lambda event: self.next_img())

		self.show_images()
		self.root.mainloop()

	def preload_images(self, folder, files):
		images = []
		for f in files:
			path = os.path.join(folder, f)
			try:
				img = Image.open(path).convert('RGB')
			except Exception:
				img = None
			images.append(img)
		return images

	def load_image(self, images, idx, target_width, target_height):
		# Ensure width and height are at least 1
		target_width = max(1, target_width)
		target_height = max(1, target_height)
		if idx < len(images) and images[idx] is not None:
			img = images[idx]
		else:
			img = Image.new('RGB', (target_width, target_height), 'black')
		# Resize with aspect ratio preserved, fit to width
		iw, ih = img.size
		scale = target_width / iw if iw > 0 else 1
		new_w = max(1, target_width)
		new_h = max(1, int(ih * scale)) if ih > 0 else 1
		if new_h > target_height:
			# If too tall, fit to height instead
			scale = target_height / ih if ih > 0 else 1
			new_h = max(1, target_height)
			new_w = max(1, int(iw * scale)) if iw > 0 else 1
		img = img.resize((new_w, new_h), Image.LANCZOS)
		# Create black background and paste centered
		bg = Image.new('RGB', (target_width, target_height), 'black')
		x = (target_width - new_w) // 2
		y = (target_height - new_h) // 2
		bg.paste(img, (x, y))
		return bg

	def show_images(self):
		# Get current canvas sizes
		c1w = self.canvas1.winfo_width() or self.canvas_width
		c1h = self.canvas1.winfo_height() or self.canvas_height
		c2w = self.canvas2.winfo_width() or self.canvas_width
		c2h = self.canvas2.winfo_height() or self.canvas_height
		img1 = self.load_image(self.images1, self.left_idx, c1w, c1h)
		img2 = self.load_image(self.images2, self.right_idx, c2w, c2h)
		self.tk_img1 = ImageTk.PhotoImage(img1)
		self.tk_img2 = ImageTk.PhotoImage(img2)
		self.canvas1.delete('all')
		self.canvas2.delete('all')
		self.canvas1.create_image(0, 0, anchor='nw', image=self.tk_img1)
		self.canvas2.create_image(0, 0, anchor='nw', image=self.tk_img2)
		name1 = self.files1[self.left_idx] if self.left_idx < len(self.files1) else '(black)'
		name2 = self.files2[self.right_idx] if self.right_idx < len(self.files2) else '(black)'
		self.label1.config(text=name1)
		self.label2.config(text=name2)

	def on_resize(self, event):
		# Redraw images on window resize
		if event.widget == self.root:
			self.show_images()

	def prev_img(self):
		# Move both left and right indices back
		if self.left_idx > 0:
			self.left_idx -= 1
		if self.right_idx > 0:
			self.right_idx -= 1
		self.show_images()

	def next_img(self):
		# Move both left and right indices forward
		if self.left_idx < len(self.files1) - 1:
			self.left_idx += 1
		if self.right_idx < len(self.files2) - 1:
			self.right_idx += 1
		self.show_images()

	def next_left_img(self):
		if self.left_idx < len(self.files1) - 1:
			self.left_idx += 1
			self.show_images()

	def next_right_img(self):
		if self.right_idx < len(self.files2) - 1:
			self.right_idx += 1
			self.show_images()

def main():
	if len(sys.argv) == 3:
		folder1, folder2 = sys.argv[1], sys.argv[2]
	else:
		root = Tk()
		root.withdraw()
		folder1 = filedialog.askdirectory(title="Select first image folder (left)")
		folder2 = filedialog.askdirectory(title="Select second image folder (right)")
		root.destroy()
	if not folder1 or not folder2:
		print("Both folders must be selected.")
		return
	ImageComparer(folder1, folder2)

if __name__ == "__main__":
	main()
