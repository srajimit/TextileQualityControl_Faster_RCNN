# TextileQualityControl_Faster_RCNN
This project uses Faster R-CNN for detecting defects in textile images. It includes training, evaluation, and a Streamlit web application for real-time predictions.
**Project Structure:**
/TextileQualityControl
	dataset.py # Custom PyTorch Dataset for Textile images
	split.py # Train/test split and DataLoader
	train.py # Model training script
	evaluate.py # Model evaluation and confusion matrix
	app.py # Streamlit app for real-time inference
	TextileQualityModel.pth # Pretrained model weights
	requirements.txt # Python dependencies
	README.md # Project documentation

**To install the dependencies:**
	pip install -r requirements.txt

Folder structure for Dataset/
/annotations
	/defect_free
	/stain
/images
	/defect_free
	/stain

**Each annotation file is a .txt in YOLO format:**
 class x_center y_center width height
Defect-free images may have empty .txt files.

Defect free instances are 68 whereas stain instances are 398. 

**To load the dataset and do preprocess:**
	python dataset.py

**To split the dataset:**
	python split.py

**To test whether images is loaded (optional):**
	python test_dataset.py

**To train the Model:**
	python train.py

**To evaluate the Model:**
	python evaluate.py

To run the application using streamlit:
	python -m streamlit run app.py

