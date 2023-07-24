# Silhouette Extraction - Computer Vision Course Project

This Python project was completed as part of a Computer Vision course at University of Tehran, focusing on the task of
silhouette extraction from images. The project's main objective was to process input video and produce their
corresponding silhouettes by separating the foreground objects from the background.

## Features

1. **Silhouette Extraction:** The project implements various image processing techniques to extract the silhouettes of
   objects from input video. It employs background subtraction algorithms to differentiate moving foreground objects
   from the static background.

2. **Image Input/Output:** The project allows users to input video in various formats and generates the corresponding
   silhouette images as output.

## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.x

## Usage

1. Clone the repository to your local machine or download the project files.
2. Install the required dependencies as specified in the 'requirements' file.
```bash
   pip install -r requirements.txt
```
3. Run the `md0.py` or `md1.py` Python scripts, providing the input video file and minimum area size (both optional).
```bash
   python3 (md0.py or md1.py) -v path/to/input_video -a 500
```

## Limitations

- The accuracy of silhouette extraction may be influenced by the complexity of the scene and the quality of the input
  video.
- The project may not handle videos with challenging lighting conditions or occlusions optimally.

## Contributions

Contributions to this project are not currently being accepted, as it was completed as part of a course project.

## License

This project is licensed under the [MIT License](LICENSE), allowing for open use and modification.

**Note:** The primary purpose of this project is for educational and learning purposes. It may require further
optimization and improvements to be suitable for production use.