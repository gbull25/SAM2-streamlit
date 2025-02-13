# SAM2-streamlit
Very simple Streamlit UI for labeling &amp; annotation of images with [Segment Anything 2](https://github.com/facebookresearch/sam2) model.


https://github.com/user-attachments/assets/a8cb07f9-7c56-4359-a05a-0f551f21f688


# Install

 1. Install segment-anything python package from Github: [Segment Anything 2](https://github.com/facebookresearch/sam2). Usually it is enough to run: ```pip install git+https://github.com/facebookresearch/sam2.git```.
 2. Download checkpoint [Checkpoint_Small](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) or [Checkpoint_Large](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) or [Checkpoint_Base+](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) and put it into checkpoints folder.

    - You can also use script /checkpoints/download_ckpts.sh to download all checkpoints.
    - Base+ version is used by default. You can change loaded model in app.py.

 3. Install requirements.txt. ```pip install -r requirements.txt```.
 4. Run ```streamlit run app.py```.


## All Checkpoints

### SAM2
- `sam2.1_hiera_l`: [SAM2.1 Hiera Small model](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- `sam2.1_hiera_b+`: [SAM2.1 Hiera Base+ model](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- `sam2.1_hiera_s`: [SAM2.1 Hiera Large model](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
- `sam2.1_hiera_t`: [SAM2.1 Hiera Tiny model](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)


# Usage

1. To switch between images, you can use the "Previous photo" \ "Next photo" buttons or select the desired image from the list.

2. You can select the object of interest in two ways: with single dots or a rectangular area (bounding box).
 
 - To select an object with dots, select "Point selection". Then click on the desired object in the photo. To improve the result, you can put several dots on the object, although most often one dot is enough.
 
 - To select an object with a rectangular area, select "Box selection". The first click in this case will select the upper left corner of the rectangle, the second - the lower right.

3. In test mode, automatic segmentation of all objects in the image is available

4. If the result is satisfactory, you can save it using the "Save Results" button. The mask in json format and the annotated image can be found in the output directory.
