# HHG_Simulation

**Python Simulation of High Harmonic Generation (HHG)**  
*(Pulse modeling, ADK ionization, electron trajectories, dipole moment, CCD analysis)*

This repository contains a Python script that simulates and analyzes High Harmonic Generation (HHG) through:

- **Pulse modeling**
- **Tunnel ionization using the ADK model**
- **Short and long electron trajectories**
- **Dipole moment calculation using the least squares method**
- **Analysis of experimental CCD images**

---

## File Description

- `1Final_code_arranged.py`  
  ----→ Complete simulation script combining theoretical modeling and image analysis.

---

##### How to Use

### 1. Set Up the Environment

1. Download the Python script from this repository.
2. Install Python 3.9+.
3. Install the required packages:
   ```bash
   pip install numpy matplotlib scipy
   ```

---

### 2. CCD Image Analysis (Optional but Important)

Some parts of the script analyze experimental CCD images. To run these:

1. Download the CCD image folders from this Google Drive link:  
   ---> [Download CCD Images](https://drive.google.com/drive/folders/1HNofMbrIi4xuuHdlKJX63B7u297RhJva)

2. After extracting the folders, you'll find multiple directories, each corresponding to a different set of measurements.

3. In the Python script, several code blocks require you to insert the path to the appropriate folder. These lines are clearly marked with comments like:

   ```python
   folder_path = "/your/local/path/to/folder_name"
   ```

4. Copy the path to each image folder and paste it into the corresponding section in the script.

5. Run the script as usual:
   ```bash
   python 1Final_code_arranged.py
   ```

> !!!!!!!   Without linking the image folders correctly, the CCD image analysis sections will not work. !!!!!!

---

## Author

Abdullah Abduljaleel  
Berlin, Germany
